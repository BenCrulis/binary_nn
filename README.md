# Apports des alternatives à la rétropropagation dans l'apprentissage des réseaux de neurones binaires

## Dépendances

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

`keras` et `tensorflow` sont utilisés uniquement pour
la partie chargement des jeux de données standard.

## Utilisation

`python -m binary_nn.train --help` affiche toutes les options.

Exemple d'éxécution de l'algorithme DFA sur le jeu de donnée CIFAR10 avec un modèle entièrement binarisé:

```sh
python -m binary_nn.train --dataset CIFAR10 --method DFA --bn --init 0.001 --lr 0.0001 --hidden-act signsat -v -b 128 --epochs 100 --hidden 700,500,300,200 --binarize --weight-clipping 1 --wandb --run-name auto
```

Retirez `--wandb` pour ne pas utiliser Weights & Biases.

## Structure du projet

`binary_nn/commons/` contient le code des fonctions et classes communes (fonctions d'activations et couches).

`binary_nn/training/` contient le code des différents algorithmes: GD (BP), DFA, DRTP.

`binary_nn/utils/binary.py` contient divers fonctions utilitaires pour la binarisation

`experiments_results/` contient les résultats bruts des expériences extraits depuis Weights & Biases.

`scripts/` contient divers scripts utilitaires

Les fichiers `sweeps*.yaml` servent à réeffectuer des grid search avec Weights & Biases.