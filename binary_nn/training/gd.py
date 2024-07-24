from binary_nn.commons.layers import HasGrad
from binary_nn.commons.losses import MSELoss
from binary_nn.commons.backprop import backprop


def gradient_descent(model, x, y, loss_fn=None, lr=1e-3):
    if loss_fn is None:
        loss_fn = MSELoss()
    y_pred, xs = model(x, train=True, save_hidden_x=True)
    err = loss_fn.error(y_pred, y)
    _ = backprop(model, err)
    loss = loss_fn(y_pred, y)

    for i, l in enumerate(model.layers):
        if isinstance(l, HasGrad):
            l.apply_grad(lr)

    return loss, xs