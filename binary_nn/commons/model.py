from typing import List

from .layers import Linear, HasBackward, HasGrad, BatchNormLayer, Reshape


class Model(HasBackward, HasGrad):
    def __init__(self, layers: List[Linear]):
        super().__init__()
        self.layers = layers

    def __call__(self, x, save_hidden_x=False, train=True):
        last_layer_index = len(self.layers) - 1
        while last_layer_index > 0:
            if isinstance(self.layers[last_layer_index], Linear):
                break
            last_layer_index -= 1

        if save_hidden_x:
            xs = [x]

        for i, l in enumerate(self.layers):
            if isinstance(l, BatchNormLayer):
                x = l(x, train=train)
            else:
                x = l(x)
            if save_hidden_x:
                xs.append(x)
        if save_hidden_x:
            return x, xs
        else:
            return x

    def backward(self, grad_out):
        layer_err = grad_out
        for i, l in reversed(list(enumerate(self.layers))):
            if isinstance(l, HasBackward):
                layer_err = l.backward(layer_err)
            else:
                raise ValueError(f"layer {i}, {l.__class__.__name__} has no backward function")
        return layer_err

    def apply_grad(self, lr=1.0):
        for l in self.layers:
            if isinstance(l, HasGrad):
                l.apply_grad(lr)

    def summary(self):
        return "\n,".join(
            (f"{i}: {l.__class__.__name__}"+(f"[{l.w.shape[-1]}]" if isinstance(l, Linear) else "")
             for i, l in enumerate(self.layers))
        )

    def n_parameters(self):
        total = 0
        for l in self.layers:
            total += l.n_parameters()
        return total

    def output_size_at(self, layer_index):
        layer = self.layers[layer_index]
        if isinstance(layer, Linear):
            return layer.w.shape[-1]
        elif isinstance(layer, Reshape):
            return layer.output_shape
        elif isinstance(layer, BatchNormLayer):
            return layer.bias.shape[-1]
        if layer_index > 1:
            previous_layer = self.layers[layer_index-1]
            if isinstance(previous_layer, Linear):
                return previous_layer.w.shape[-1]
            elif isinstance(layer, Reshape):
                return layer.output_shape
            elif isinstance(layer, BatchNormLayer):
                return layer.bias.shape[-1]
        if layer_index < len(self.layers) - 1:
            next_layer = self.layers[layer_index+1]
            if isinstance(next_layer, Linear):
                return next_layer.w.shape[0]
            elif isinstance(layer, Reshape):
                return layer.output_shape
            elif isinstance(layer, BatchNormLayer):
                return layer.bias.shape[-1]
        raise ValueError(f"could not determine output size at layer {layer_index}")
