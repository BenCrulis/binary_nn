from abc import ABC, abstractmethod

import numpy as np


class DifferentiableLoss(ABC):
    @abstractmethod
    def error(self, y_pred, y):
        pass


class Loss(ABC):
    @abstractmethod
    def __call__(self, y_pred, y):
        pass


class MSELoss(Loss, DifferentiableLoss):
    def __call__(self, y_pred, y):
        d = y_pred - y
        l = np.square(d).sum()/y.size
        return l

    def error(self, y_pred, y):
        return 2*(y_pred - y)