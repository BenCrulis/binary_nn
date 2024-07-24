from abc import abstractmethod, ABC

import numpy as np

from binary_nn.utils.binary import binarize

EPSILON = 10 ** -3


class HasClone(ABC):
    @abstractmethod
    def clone(self):
        pass


class HasBackward(ABC):
    @abstractmethod
    def backward(self, grad_out):
        pass


class HasGrad(ABC):
    def __init__(self):
        self.grad = None

    @abstractmethod
    def apply_grad(self, lr=1.0):
        pass


class Linear(HasGrad, HasBackward):
    def __init__(self, features_in, features_out, binarize=False, weight_ste=None, init=0.1):
        super().__init__()
        self.binarize = binarize
        self.weight_ste = weight_ste() if weight_ste is not None else None
        self.bw = None
        self.w = np.random.uniform(-1.0, 1.0, size=(features_in, features_out)) * init

    def __call__(self, x):
        x = x.astype(np.float)  # convert to float for faster matrix multiplication
        w = self.w
        if self.binarize:
            if self.bw is None:
                w = binarize(w, dtype=np.float)
                self.bw = w
            else:
                w = binarize(w, out=self.bw)
        self.x = x
        out = x @ w
        return out

    def backward(self, grad_out):
        self.grad = self.x.T @ grad_out
        if self.weight_ste is not None and self.binarize:
            self.weight_ste(self.w)
            self.grad = self.weight_ste.backward(self.grad)
        if self.binarize:
            return grad_out @ self.bw.T
        return grad_out @ self.w.T

    def apply_grad(self, lr=1.0):
        self.w -= lr*self.grad

    def n_parameters(self):
        return self.w.size


class Reshape(HasBackward):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        if not isinstance(output_shape, tuple):
            output_shape = (output_shape,)
        self.output_shape = output_shape

    def __call__(self, x: np.ndarray):
        return x.reshape((x.shape[0], *self.output_shape))

    def backward(self, grad_out):
        return grad_out.reshape((grad_out.shape[0], *self.input_shape))

    def n_parameters(self):
        return 0


class Scaler(HasBackward):
    def __init__(self, previous_layer_size):
        self.prev_size = previous_layer_size

    def __call__(self, x: np.ndarray):
        return x / self.prev_size

    def backward(self, grad_out):
        return grad_out / self.prev_size

    def n_parameters(self):
        return 0


class BatchNormLayer(HasGrad, HasBackward):
    """
    adapted from https://github.com/renan-cunha/BatchNormalization/blob/master/src/feed_forward/layers.py
    License:
    MIT License

    Copyright (c) 2019 Renan Cunha

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    def __init__(self, dims: int) -> None:
        super().__init__()
        self.gamma = np.ones((1, dims), dtype="float32")
        self.bias = np.zeros((1, dims), dtype="float32")

        self.running_mean_x = np.zeros(0)
        self.running_var_x = np.zeros(0)

        # forward params
        self.var_x = np.zeros(0)
        self.stddev_x = np.zeros(0)
        self.x_minus_mean = np.zeros(0)
        self.standard_x = np.zeros(0)
        self.num_examples = 0
        self.mean_x = np.zeros(0)
        self.running_avg_gamma = 0.9

        # backward params
        self.gamma_grad = np.zeros(0)
        self.bias_grad = np.zeros(0)

    def update_running_variables(self) -> None:
        is_mean_empty = np.array_equal(np.zeros(0), self.running_mean_x)
        is_var_empty = np.array_equal(np.zeros(0), self.running_var_x)
        if is_mean_empty != is_var_empty:
            raise ValueError("Mean and Var running averages should be "
                             "initialized at the same time")
        if is_mean_empty:
            self.running_mean_x = self.mean_x
            self.running_var_x = self.var_x
        else:
            gamma = self.running_avg_gamma
            self.running_mean_x = gamma * self.running_mean_x + (1.0 - gamma) * self.mean_x
            self.running_var_x = gamma * self.running_var_x + (1. - gamma) * self.var_x

    def __call__(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self.num_examples = x.shape[0]
        if train:
            self.mean_x = np.mean(x, axis=0, keepdims=True)
            self.var_x = np.mean((x - self.mean_x) ** 2, axis=0, keepdims=True)
            self.update_running_variables()
        else:
            self.mean_x = self.running_mean_x.copy()
            self.var_x = self.running_var_x.copy()

        self.var_x += EPSILON
        self.stddev_x = np.sqrt(self.var_x)
        self.x_minus_mean = x - self.mean_x
        self.standard_x = self.x_minus_mean / self.stddev_x
        return self.gamma * self.standard_x + self.bias

    def backward(self, grad_input: np.ndarray) -> np.ndarray:
        standard_grad = grad_input * self.gamma

        var_grad = np.sum(standard_grad * self.x_minus_mean * -0.5 * self.var_x ** (-3/2),
                          axis=0, keepdims=True)
        stddev_inv = 1 / self.stddev_x
        aux_x_minus_mean = 2 * self.x_minus_mean / self.num_examples

        mean_grad = (np.sum(standard_grad * -stddev_inv, axis=0,
                            keepdims=True) +
                            var_grad * np.sum(-aux_x_minus_mean, axis=0,
                            keepdims=True))

        self.gamma_grad = np.sum(grad_input * self.standard_x, axis=0,
                                 keepdims=True)
        self.bias_grad = np.sum(grad_input, axis=0, keepdims=True)

        return standard_grad * stddev_inv + var_grad * aux_x_minus_mean + mean_grad / self.num_examples

    def apply_grad(self, lr=1.0):
        self.gamma -= lr * self.gamma_grad
        self.bias -= lr * self.bias_grad

    def n_parameters(self):
        return self.gamma.size * 2
