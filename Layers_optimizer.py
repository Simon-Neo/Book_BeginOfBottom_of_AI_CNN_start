
import numpy as np
from abc import *

class Layers(metaclass=ABCMeta):
    @abstractmethod
    def update(self, params, grads):
        pass

class SGD(Layers):
    def __init__(self, LEARN_RATE = 0.01):
        self.lr = LEARN_RATE

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum(Layers):
    def __init__(self, LEARN_RATE = 0.01, momentum=0.9):
        self.lr = LEARN_RATE
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

class AdaGrad(Layers):
    def __init__(self, LEARN_RATE=0.01):
        self.lr = LEARN_RATE
        self.h = None
    def update(self, params, grads):
        if self.h == None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1.0e-7)