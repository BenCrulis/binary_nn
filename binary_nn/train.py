import argparse
import random
from functools import partial

import numpy as np
import wandb.util
from keras.datasets import mnist, fashion_mnist, cifar10
import sklearn.metrics as metrics

from binary_nn.commons.model import Model
from binary_nn.commons.activations import Tanh, SignSTE, Identity, SignSTESat, SignSTETanh
from binary_nn.commons.layers import Reshape, Linear, BatchNormLayer, Scaler

from binary_nn.training.dfa import DFA
from binary_nn.training.drtp import DirectRandomTargetProjection
from binary_nn.training.gd import gradient_descent

from binary_nn.utils.binary import to_bin_targets
from binary_nn.utils.evaluation import accuracy
from binary_nn.utils.hsic import HSIC_Recorder
from binary_nn.utils.logging import ConsoleLogger, LoggerGroup, WandbLogger
from binary_nn.utils.model import clip_weights, measure_normal_and_packed_size, average_neuron_saturation


def build_model(output_size, input_shape, hidden, hidden_act, last_layer_act, binarize=False, binarize_last=False,
                weight_ste=None, initial_val=0.1, batchnorm=False, scale_norm=False):
    n_inputs = np.prod(input_shape)
    layers = [Reshape(input_shape, n_inputs)]
    for c1, c2 in zip([n_inputs] + hidden, hidden):
        layers.append(Linear(c1, c2, binarize=binarize,
                             weight_ste=weight_ste if binarize else None, init=initial_val))
        if scale_norm:
            layers.append(Scaler(c1))
        if batchnorm:
            layers.append(BatchNormLayer(c2))
        layers.append(hidden_act())
    layers.append(Linear(hidden[-1], output_size, binarize=binarize_last, init=initial_val))
    layers.append(last_layer_act())
    return Model(layers)


activations = {
    "tanh": Tanh,
    "sign": SignSTE,
    "signsat": SignSTESat,
    "signtanh": SignSTETanh,
    "id": Identity,
}


weight_stes = {
    "tanh": SignSTETanh,
    "sat": SignSTESat
}


def validate_weight_ste(weight_ste):
    if weight_ste == "identity":
        return None
    res = weight_stes.get(weight_ste, None)
    if res is not None:
        return res
    else:
        raise ValueError(f"cannot find STE for weights: {weight_ste}")


def parse_model(output_size, input_shape, hidden_str, hidden_act_str, last_act_str, binarize, binarize_last, weight_ste,
                initial_val, batchnorm=False, scale_norm=False):
    sizes = list(map(int, hidden_str.split(",")))
    hidden_act = activations.get(hidden_act_str.lower(), None)
    if hidden_act is None:
        raise ValueError(f"Could not find activation: {hidden_act_str}")
    last_act = activations.get(last_act_str.lower(), None)
    if last_act is None:
        raise ValueError(f"Could not find activation: {hidden_act_str}")
    return build_model(output_size, input_shape, sizes, hidden_act, last_act, binarize, binarize_last, weight_ste,
                       initial_val, batchnorm, scale_norm)


def parse_args():
    parser = argparse.ArgumentParser(description="Train neural networks with different methods")
    parser.add_argument("-m", "--method", type=str, default="GD",
                        help="available methods: GD (default), DFA, DRTP")
    parser.add_argument("-d", "--dataset", type=str, default="MNIST",
                        help="available datasets: MNIST, FASHION_MNIST, CIFAR10")
    parser.add_argument("-e", "--epochs", type=int, default=5,
                        help="how many epochs are used for training")
    parser.add_argument("-b", "--batchsize", type=int, default=128,
                        help="how many training examples are used for one iteration")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for continuous weights models")
    parser.add_argument("--hidden", type=str, default="700,500", help="hidden layer sizes")
    parser.add_argument("--init", type=float, default=0.1, help="weights are initialized in U(-init, init)")
    parser.add_argument("--bn", action="store_true", help="add batchnorm layers after linear layers")
    parser.add_argument("--sn", action="store_true", help="add scale normalization layers after linear layers")

    act_descriptions = ", ".join(activations.keys())

    parser.add_argument("--hidden-act", type=str, default="tanh",
                        help=f"hidden layers activations, available: {act_descriptions}")
    parser.add_argument("--last-act", type=str, default="tanh",
                        help=f"last layer activation function, available: {act_descriptions}")
    parser.add_argument("--binarize", action="store_true", help="binarize internal layers in forward pass")
    parser.add_argument("--weight-clipping", type=float, default=None,
                        help="clip weights between -value and value, no clipping if None")
    parser.add_argument("--weight-ste", type=str, default="identity",
                        help="STE for the weights one of: identity, sat, tanh")
    parser.add_argument("--binarize_last", action="store_true", help="binarize last linear layer")
    parser.add_argument('--verbose', '-v', action='count', default=0, help="verbosity")
    parser.add_argument("--seed", default=None, help="seed for the random number generator")

    # Wandb arguments
    parser.add_argument("--wandb", action="store_true", help="use Weights&Biases logging")
    parser.add_argument("--run-name", default=None,
                        help="custom name for the W&B experiment, is generated automatically otherwise")

    args = parser.parse_args()
    return args


def main():
    from hashlib import md5
    EXPERIMENT_VERSION = "0.1.5"  # remember to increment the version number according to changes made in the script
    with open(__file__) as file:
        md5_obj = md5(bytes(file.read(), "utf8"))
        SCRIPT_HASH = md5_obj.hexdigest()
    args = parse_args()

    print("EXPERIMENT VERSION", EXPERIMENT_VERSION)
    print("SCRIPT HASH:", SCRIPT_HASH)

    seed = np.random.randint(1000000)

    if args.seed:
        seed = int(args.seed)
    np.random.seed(seed)
    random.seed(seed)

    batch_size = args.batchsize
    n_epochs = args.epochs
    binarize_weights = args.binarize
    weight_clipping = args.weight_clipping
    if weight_clipping is not None:
        if weight_clipping <= 0.0:
            print("warning, weight clipping is negative, disabling weight clipping...")
            weight_clipping = None
    hidden_act = args.hidden_act.lower()
    weight_ste = args.weight_ste.lower()
    binarize_last = args.binarize_last
    lr = args.lr
    initial_val = args.init
    verbosity = args.verbose
    use_wandb = args.wandb
    batchnorm = args.bn
    scale_norm = args.sn
    dataset = args.dataset.lower()

    weight_type = "continuous"
    if binarize_weights:
        weight_type = "binarised"

    project_name = f"{dataset} custom"

    print(f"using dataset {dataset}")

    logger = LoggerGroup([ConsoleLogger(verbosity)])
    run_name = None
    if args.run_name:
        run_name = args.run_name
        if args.run_name.lower() == "auto":
            run_name = f"{args.method.upper()} {weight_type} {hidden_act}"
        print(f"starting run \"{run_name}\"")
    else:
        print("starting unammed run")
    if use_wandb:
        print("using Weights & Biases for logging")
        logger.add_logger(WandbLogger(project_name, run_name=run_name))

    w_ste_impl = validate_weight_ste(weight_ste)

    if dataset == "mnist":
        (x_train, y_train_), (x_test, y_test_) = mnist.load_data()

        x_train, x_test = (x_train / 127.5 - 1.0, x_test / 127.5 - 1.0)
        x_train, x_test = (x_train[..., None], x_test[..., None])

        n_classes = 10
        y_train, y_test = to_bin_targets(y_train_, 1, 10), to_bin_targets(y_test_, 1, n_classes)
    elif dataset == "fashion_mnist":
        (x_train, y_train_), (x_test, y_test_) = fashion_mnist.load_data()

        x_train, x_test = (x_train / 127.5 - 1.0, x_test / 127.5 - 1.0)
        x_train, x_test = (x_train[..., None], x_test[..., None])

        n_classes = 10
        y_train, y_test = to_bin_targets(y_train_, 1, 10), to_bin_targets(y_test_, 1, n_classes)
    elif dataset == "cifar" or dataset == "cifar10" or dataset == "cifar-10":
        (x_train, y_train_), (x_test, y_test_) = cifar10.load_data()
        y_train_ = y_train_.squeeze()
        y_test_ = y_test_.squeeze()

        x_train, x_test = (x_train / 127.5 - 1.0, x_test / 127.5 - 1.0)
        x_train, x_test = (x_train[..., None], x_test[..., None])

        n_classes = 10
        y_train, y_test = to_bin_targets(y_train_, 1, 10), to_bin_targets(y_test_, 1, n_classes)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    input_shape = x_train.shape[1:]

    model: Model = parse_model(n_classes, input_shape, args.hidden, hidden_act, args.last_act,
                               binarize=binarize_weights, binarize_last=binarize_last,
                               weight_ste=w_ste_impl, initial_val=initial_val, batchnorm=batchnorm, scale_norm=scale_norm)

    try:
        measure_normal_and_packed_size(model)
    except ImportError as e:
        print("install objsize to compute the model size")
        pass


    algo = args.method.lower()
    if algo == "gd":
        learning_algorithm = partial(gradient_descent, lr=lr)
    elif algo == "dfa":
        learning_algorithm = DFA(model, 10, lr=lr)
    elif algo == "drtp":
        learning_algorithm = DirectRandomTargetProjection(model, lr=lr)
    else:
        raise ValueError(f"Unknown algorithm: {args.method}")

    logs = {
        "algorithm": algo,
        "batch size": batch_size,
        "number of epochs": n_epochs,
        "weight type": weight_type,
        "weight clipping": "none" if weight_clipping is None else weight_clipping,
        "hidden activation function": hidden_act,
        "binarize weights": binarize_weights,
        "weight STE": weight_ste,
        "binarize last layer": binarize_last,
        "initial value": initial_val,
        "learning rate": lr,
        "seed": seed,
        "architecture": model.summary(),
        "batchnorm": batchnorm,
        "number of parameters": model.n_parameters(),
        "hidden sizes": args.hidden,
        "cmd args": str(args),
        "experiment verion": EXPERIMENT_VERSION,
        "script hash": SCRIPT_HASH,
    }

    hsic_recorder = HSIC_Recorder(model, use_wandb)

    row_limit = 2000
    hsic_res = hsic_recorder.update(x_train[:row_limit], y_train[:row_limit])
    logs.update(hsic_res)
    hsic_recorder.plot_wandb(0)

    logger.log(0, 0, logs, level=0)

    index = np.arange(len(x_train))
    iteration = 1

    for epoch in range(n_epochs):
        np.random.shuffle(index)
        nb_batches = len(x_train) // batch_size

        print()
        print(f"epoch {epoch}")

        for i in range(nb_batches):
            batch_idx = index[batch_size * i:batch_size * (i + 1)]
            batch_x = x_train[batch_idx]
            batch_y = y_train[batch_idx]

            logs = {}

            loss, xs = learning_algorithm(model, batch_x, batch_y)
            xs = xs[1:]
            avg_sat, var_acts, var_grads, max_abs_val = average_neuron_saturation(model, xs)
            logs["average saturation"] = avg_sat
            logs["activation variance"] = var_acts
            logs["gradient variance"] = var_grads
            logs["max absolute gradient value"] = max_abs_val
            if weight_clipping:
                clip_weights(model, weight_clipping)

            logs["batch/loss"] = loss

            v_level = 2

            if i == nb_batches - 1:
                print(f"------------------- end of epoch {epoch} -------------------")

                # evaluate train set
                pred_y = model(x_train, train=False)
                pred_label = pred_y.argmax(-1)
                acc = accuracy(pred_label, y_train_)
                prec = metrics.precision_score(y_train_, pred_label, average="macro", zero_division=0.0)
                f1 = metrics.f1_score(y_train_, pred_label, average="macro", zero_division=0.0)
                logs["train/accuracy"] = acc
                logs["train/precision"] = prec
                logs["train/f1"] = f1

                hsic_res = hsic_recorder.update(x_train[:row_limit], y_train[:row_limit])
                logs.update(hsic_res)
                #hsic_recorder.plot_wandb(iteration)

                # evaluate test set
                pred_y = model(x_test, train=False)
                pred_label = pred_y.argmax(-1)
                acc = accuracy(pred_label, y_test_)
                prec = metrics.precision_score(y_test_, pred_label, average="macro")
                f1 = metrics.f1_score(y_test_, pred_label, average="macro")
                logs["test/accuracy"] = acc
                logs["test/precision"] = prec
                logs["test/f1"] = f1

                if use_wandb:
                    logs["test/confusion"] = wandb.plot.confusion_matrix(probs=None,
                                                                         y_true=list(y_test_), preds=list(pred_label))

                v_level = 1

            logger.log(iteration, epoch, logs, level=v_level)
            iteration += 1

    print("finished training")


if __name__ == '__main__':
    main()
