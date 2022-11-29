from abc import abstractmethod

import numpy as np

from binary_nn.utils.binary import binary_kwta
from binary_nn.utils.binary import kwta, binarize
from binary_nn.commons.layers import HasBackward


def tanh_adjoint(x):
    return 1.0 - np.square(np.tanh(x))


class Activation:
    @abstractmethod
    def __call__(self, x):
        pass

    def n_parameters(self):
        return 0


class SignSTE(Activation, HasBackward):
    def __call__(self, x):
        return binarize(x)

    def backward(self, grad_out):
        return grad_out


class SignSTESat(Activation, HasBackward):
    def __call__(self, x):
        self.x = x
        return binarize(x)

    def backward(self, grad_out):
        #return grad_out * ((self.x > -1.0) & (self.x < 1.0))
        out = grad_out.copy()
        c = np.empty(grad_out.shape, dtype=bool)
        np.greater_equal(self.x, 1, out=c)
        out[c] = 0.0
        np.less_equal(self.x, -1, out=c)
        out[c] = 0.0
        return out


class SignSTETanh(Activation, HasBackward):
    def __call__(self, x):
        self.x = x
        return binarize(x)

    def backward(self, grad_out):
        return grad_out * tanh_adjoint(self.x)


class Tanh(Activation, HasBackward):
    def __call__(self, x):
        self.x = x
        return np.tanh(x)

    def backward(self, grad_out):
        return (1.0 - np.square(np.tanh(self.x))) * grad_out


class Identity(Activation, HasBackward):
    def __call__(self, x):
        return x

    def backward(self, grad_out):
        return grad_out


class KWTA(Activation, HasBackward):
    def __init__(self, k=0.5):
        self.k = k

    def __call__(self, x):
        self.x = x
        self.y = kwta(x, self.k)
        return self.y

    def backward(self, grad_out):
        return grad_out * (self.y > 0.0)


class bKWTA(Activation):
    def __init__(self, k=0.5):
        self.k = k

    def __call__(self, x):
        return binary_kwta(x, self.k)