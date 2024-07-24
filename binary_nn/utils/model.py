import numpy as np

from copy import deepcopy

from binary_nn.commons.activations import Tanh, SignSTE, SignSTESat
from binary_nn.commons.layers import Linear
from binary_nn.commons.model import Model


def clip_weights(model: Model, bound):
    """
    clip weights for continuous layers
    :param model:
    :param bound:
    :return:
    """
    for l in model.layers:
        if isinstance(l, Linear):
            l.w = np.clip(l.w, -bound, bound, out=l.w)


def pack_model(model: Model, dimension="biggest"):
    if not (isinstance(dimension, int) or dimension == "biggest"):
        raise ValueError("dimension should be an int or 'biggest'")

    packed_model = deepcopy(model)

    for layer in packed_model.layers:
        if isinstance(layer, Linear):
            if layer.binarize:
                bw = layer.w > 0.5
                if dimension == "biggest":
                    dimension = np.argmax(bw.shape)
                pbw = np.packbits(bw, axis=dimension)
                layer.w = pbw
    return packed_model


def measure_normal_and_packed_size(model):
    import objsize
    packed_model = pack_model(model)

    model_size = objsize.get_deep_size(model)
    packed_model_size = objsize.get_deep_size(packed_model)
    print(f"model size: {model_size}")
    print(f"packed model size: {packed_model_size}")


def average_neuron_saturation(model, xs, tanh_threshold=2.0):
    pre_acts = []
    activation_fn = None
    for l, x in zip(model.layers, xs):
        if isinstance(l, Linear):
            pre_acts.append(x)
        elif isinstance(l, Tanh):
            activation_fn = "tanh"
        elif isinstance(l, (SignSTE, SignSTESat)):
            activation_fn = "sign"

    pre_acts = pre_acts[:-1]  # remove last layer activations from the measurement

    if activation_fn == "tanh":
        threshold = tanh_threshold
    elif activation_fn == "sign":
        threshold = 1.0
    else:
        raise ValueError("could not find a valid activation function to measure saturation")

    n = 0
    v = 0.0
    for a in pre_acts:
        v += (np.abs(a) > threshold).sum()
        n += a.size

    return v/n
