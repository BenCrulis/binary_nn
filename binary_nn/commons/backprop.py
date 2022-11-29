from binary_nn.commons.layers import HasBackward


def backprop(model, err):
    layer_err = err
    for i, l in reversed(list(enumerate(model.layers))):
        if hasattr(l, "backward"):
            layer_err = l.backward(layer_err)
        else:
            raise ValueError(f"layer {i}, {l.__class__.__name__} has no backward function")
    return layer_err
