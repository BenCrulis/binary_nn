## Dependencies

```
numpy
pyyaml
wandb
tqdm
scikit-learn
keras
tensorflow
objsize
```

`keras` and `tensorflow` are only used for loading datasets.

## Usage

`python -m binary_nn.train --help` prints all options.

Example of execution of the DFA algorithm on CIFAR10 with an entirely binarized model:

```sh
python -m binary_nn.train --dataset CIFAR10 --method DFA --bn --init 0.001 --lr 0.0001 --hidden-act signsat -v -b 128 --epochs 100 --hidden 700,500,300,200 --binarize --weight-clipping 1 --wandb --run-name auto
```

Remove the `--wandb` flag in order to disable Weights & Biases logging.

## Project structure

`binary_nn/commons/` contains the code for the common functions and classes (activation functions and layers)

`binary_nn/training/` contains the code for the different training algorithms: GD (BP), DFA, DRTP.

`binary_nn/utils/binary.py` contains binarization utilitaries

`experiments_results/` contains the raw results of the paper experiments extracted from the Weights & Biases logs.

`scripts/` contains diverse utilitary scripts

The `sweeps*.yaml` files are used for initializing grid searches in Weights & Biases.