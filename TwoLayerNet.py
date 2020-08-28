import numpy as np
import neuron_func as nr_func

from collections import OrderedDict as order_dic
import Layers_optimizer

import Layers

class TwoLayer():
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std = 0.01):
        # -------------------    PARAMETER   -------------------
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # -------------------    LAYERS   -------------------
        self.layers = order_dic()

        self.layers['Affine01'] = Layers.Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu01'] = Layers.Relu()
        self.layers['Affine02'] = Layers.Affine(self.params['W2'], self.params['b2'])
        self.last_layer = Layers.SoftWithLoss()

        self.optimizer = Layers_optimizer.AdaGrad(LEARN_RATE=0.01)

    def predict(self, x):
        for layer_key in self.layers.keys():
            x = self.layers[layer_key].forward(x)
        return x

    def loss(self, x, labels):
        outs = self.predict(x)
        return self.last_layer.forward(outs, labels)
        # return nr_func.cross_entropy_error(outs, labels)
    def loss_numeric(self, x, labels):
        outs = self.predict(x)
        return nr_func.cross_entropy_error(outs, labels)

    def my_predict(self, x):
        outs = self.predict(x)
        return nr_func.softmax(outs)

    def accuracy(self, x, labels):
        outs = self.predict(x)
        out_args = np.argmax(outs, axis=1)
        labels_args = np.argmax(labels, axis=1)

        accuracy = np.sum(out_args == labels_args) / float(x.shape[0])
        return accuracy

    def gradient(self, x, labels, LEARN_RATE):

        # -------------------    FORWARD   -------------------
        self.loss(x, labels)
        # -------------------    BACKWARD   -------------------
        diff_out = 1
        diff_out = self.last_layer.backward(diff_out)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            diff_out = layer.backward(diff_out)
        # -------------------    GRADIENT   -------------------
        grads = {}
        grads['W1'] = self.layers['Affine01'].getDiff_W()
        grads['b1'] = self.layers['Affine01'].getDiff_b()
        grads['W2'] = self.layers['Affine02'].getDiff_W()
        grads['b2'] = self.layers['Affine02'].getDiff_b()

        # -------------------    OPTIMIZER   -------------------
        self.optimizer.update(self.params, grads)

    def numerical_gradient(self, x, labels):
        loss_W = lambda W: self.loss_numeric(x, labels)

        # -------------------    UPDATE_GRADIENT   -------------------
        grads = {}
        grads['W1'] = nr_func.numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = nr_func.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = nr_func.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = nr_func.numerical_gradient(loss_W, self.params['b2'])

        return grads

#print('----------')
