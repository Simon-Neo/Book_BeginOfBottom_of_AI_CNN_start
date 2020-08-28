import numpy as np
import neuron_func as nr_func


class simpleNet():
    def __init__(self):
        self.W = np.array([
            [0.47355232, 0.9977393, 0.84668094],
            [0.85557411, 0.03563661, 0.69422093]
        ])

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, labels):
        outputs = self.predict(x)
        y = nr_func.softmax_v01(outputs)
        loss = nr_func.cross_entropy_error(y, labels)

        return loss
