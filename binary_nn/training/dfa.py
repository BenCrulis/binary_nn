import numpy as np

from binary_nn.commons.activations import Activation
from binary_nn.commons.layers import HasGrad, HasBackward
from binary_nn.commons.losses import MSELoss
from binary_nn.commons.model import Model


class DFA:
    def __init__(self, model, output_size, loss_fn=None, lr=1e-3):
        self.model = model
        self.loss_fn = loss_fn if loss_fn is not None else MSELoss()
        self.lr = lr

        self.bw = []

        bw_scale = 1.0

        nl = len(model.layers)
        for i, l in enumerate(model.layers[:-1]):
            if i < nl - 1 and isinstance(l, Activation):
                #layer_n_outputs = model.layers[i-1].w.shape[-1]
                layer_n_outputs = model.output_size_at(i-1)
                self.bw.append((i, np.random.uniform(-bw_scale, bw_scale, size=(output_size, layer_n_outputs))))
        pass

    def __call__(self, model: Model, x, y):
        y_pred, xs = model(x, train=True, save_hidden_x=True)
        loss_fn = self.loss_fn
        err = loss_fn.error(y_pred, y)
        loss = loss_fn(y_pred, y)

        for j, (tl, bw) in enumerate(self.bw + [(len(model.layers) - 1, None)]):
            l_err = err @ bw if bw is not None else err
            i = tl
            while i >= 0 and (i > self.bw[j-1][0] if j > 0 else True):
                layer: HasBackward = model.layers[i]
                l_err = layer.backward(l_err)
                i -= 1

        for i, l in enumerate(model.layers):
            if isinstance(l, HasGrad):
                l.apply_grad(self.lr)

        return loss, xs