import numpy as np

from binary_nn.commons.activations import Activation
from binary_nn.commons.layers import HasGrad, HasBackward, Linear
from binary_nn.commons.losses import MSELoss
from binary_nn.commons.model import Model


class DirectRandomTargetProjection:
    def __init__(self, model, loss_fn=None, lr=1e-3):
        self.model = model
        self.loss_fn = loss_fn if loss_fn is not None else MSELoss()
        self.lr = lr

        model_n_outputs = [x for x in model.layers if isinstance(x, Linear)][-1].w.shape[1]

        self.dp = []

        nl = len(model.layers)
        for i, l in enumerate(model.layers[:-1]):
            if i < nl - 1 and isinstance(l, Activation):
                #_, layer_n_outputs = model.layers[i-1].w.shape
                layer_n_outputs = model.output_size_at(i-1)
                self.dp.append((i, np.random.uniform(-1.0, 1.0, (model_n_outputs, layer_n_outputs))))
                # self.dp.append((i, np.random.normal(.0, 1.0, (model_n_outputs, layer_n_outputs))))
        pass

    def __call__(self, model: Model, x, y):
        y_pred, xs = model(x, save_hidden_x=True, train=True)

        loss_fn = self.loss_fn
        err = loss_fn.error(y_pred, y)
        loss = loss_fn(y_pred, y)

        for j, (tl, rp) in enumerate(self.dp):
            ly = (y > 0.0) @ rp
            l_err = ly

            i = tl
            while i >= 0 and (i > self.dp[j - 1][0] if j > 0 else True):
                layer: HasBackward = model.layers[i]
                l_err = layer.backward(l_err)
                i -= 1

        i = len(model.layers) - 1
        l_err = err
        while i >= 0 and (i > self.dp[-1][0] if len(self.dp) > 0 else True):
            layer: HasBackward = model.layers[i]
            l_err = layer.backward(l_err)
            i -= 1

        for i, l in enumerate(model.layers):
            if isinstance(l, HasGrad):
                l.apply_grad(self.lr)

        return loss, xs
