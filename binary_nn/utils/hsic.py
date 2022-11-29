from binary_nn.commons.activations import Activation
from binary_nn.commons.model import Model
from .kernels import CosineSimilarityKernel, estimate_hsic_zy_objective, estimate_hsic_ker


class HSIC_Recorder:
    def __init__(self, model, use_wandb, x_ker=CosineSimilarityKernel(),
                z_ker=CosineSimilarityKernel(),
                y_ker=CosineSimilarityKernel(),
                normalised=True):
        self.model = model
        self.use_wandb = use_wandb
        self.layers = [i for i, l in enumerate(model.layers) if isinstance(l, Activation) or i == len(model.layers) - 1]
        self.x_ker = x_ker
        self.z_ker = z_ker
        self.y_ker = y_ker
        self.normalized = normalised
        self.layers_xz = [[] for i in self.layers]
        self.layers_zy = [[] for i in self.layers]
        self.layer_names = [f"layer {i}" for i in self.layers]

    def update(self, x, y):
        keys_xz, layers_xz, keys_zy, layers_zy = report_hsic(self.model, x, y, self.x_ker, self.z_ker, self.y_ker, self.normalized)
        hsic_str = "nHSIC" if self.normalized else "HSIC"
        results = {}
        for i in range(len(layers_xz)):
            self.layers_xz[i].append(layers_xz[i])
            self.layers_zy[i].append(layers_zy[i])
            results[f"layer_{i} {hsic_str} XZ"] = layers_xz[i]
            results[f"layer_{i} {hsic_str} ZY"] = layers_zy[i]
        return results

    def plot_wandb(self, step):
        if self.use_wandb:
            try:
                import wandb
                wandb.log({"HSIC": wandb.plot.line_series(self.layers_xz, self.layers_zy, self.layer_names,
                                                 "HSIC by layer", "HSIX XZ")}, step=step)
            except ImportError:
                pass


def report_hsic(model: Model, x, y,
                x_ker=CosineSimilarityKernel(),
                z_ker=CosineSimilarityKernel(),
                y_ker=CosineSimilarityKernel(),
                normalised=True):
    row_limit = 2000
    if len(x) > row_limit:
        raise ValueError(f"HSIC cannot be measured on more than {row_limit} instances")

    # reported = {}

    layers_xz = []
    keys_xz = []
    layers_zy = []
    keys_zy = []

    zs = [x]
    hsic_str = "nHSIC" if normalised else "HSIC"

    for i, l in enumerate(model.layers):
        z = l(zs[-1])
        zs.append(z)
        if isinstance(l, Activation) or i == len(model.layers) - 1:
            xz = estimate_hsic_ker(x, z, x_ker, z_ker, normalize=normalised)
            zy = estimate_hsic_ker(z, y, z_ker, y_ker, normalize=normalised)
            # reported[f"layer_{i} {hsic_str} XZ"] = xz
            # reported[f"layer_{i} {hsic_str} ZY"] = zy
            keys_xz.append(f"layer_{i} {hsic_str} XZ")
            keys_zy.append(f"layer_{i} {hsic_str} ZY")
            layers_xz.append(xz)
            layers_zy.append(zy)

    return keys_xz, layers_xz, keys_zy, layers_zy
