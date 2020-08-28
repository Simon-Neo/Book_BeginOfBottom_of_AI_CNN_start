import numpy as np
import neuron_func as nr_func
from abc import *

class Layers(metaclass=ABCMeta):

    @abstractmethod
    def forward(self, x):
        pass
    @abstractmethod
    def backward(self, diff_out):
        pass

# ReLU
class Relu(Layers):
    def __init__(self):
        self.zero_mask = None
    def forward(self, x):
        # 이것도 그냥 0보다 작다면으로 하기.
        self.zero_mask = (x <= 0)
        # 이걸로 해보기 문제되는지x[self.zero_mask] = 0
        out = x.copy()
        out[self.zero_mask] = 0
        return out

    def backward(self, diff_out):
        diff_out[self.zero_mask] = 0
        # 이것도 바로 그냥 내보내 보기.
        dx = diff_out
        return dx


class Sigmoid(Layers):
    def __init__(self):
        self.out = None

    def forward(self, x):
        #sigmoid 이거 선생님이 한걸로 교체해보기
        self.out = nr_func.sigmoid(x)
        out = self.out
        return out

    def backward(self, diff_out):
        diff_val = diff_out * (1.0 - self.out) * self.out
        return diff_val


class Affine(Layers):
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None

        self.diff_W = None
        self.diff_b = None

    def getDiff_W(self):
        return self.diff_W
    def getDiff_b(self):
        return self.diff_b

    def forward(self, x):
        self.x = x
        return np.matmul(x, self.W) + self.b

    def backward(self, diff_out):
        self.diff_b = np.sum(diff_out, axis=0)
        self.diff_W = np.matmul(self.x.T, diff_out)
        diff_x = np.matmul(diff_out, self.W.T)
        return diff_x

class SoftWithLoss:
    def __init__(self):
        self.loss = None
        self.out = None
        self.labels = None

    def getLoss(self):
        return self.loss

    def forward(self, x , labels):
        self.labels = labels
        self.out = nr_func.softmax_v01(x)
        self.loss = nr_func.cross_entropy_error(self.out, labels)
        return self.loss

    def backward(self, dout = 1):
        batch_size = self.labels.shape[0]
        #print(np.prod(self.labels.shape))

        diff_out = (self.out - self.labels)/ batch_size
        return diff_out