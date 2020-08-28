import sys, os
import numpy as np

from mnist import load_mnist
import neuron_func as nr_func

from simpleNet import simpleNet
from TwoLayerNet import TwoLayer

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True,
                                                  one_hot_label=True,
                                                  normalize=True)
import matplotlib.pyplot as plt
import datetime

#print(x_train.shape) #(60000, 784)
x_size = x_train.shape[0]
batch_size = 150
LEARN_LATE = 0.01


net = TwoLayer(input_size=x_train.shape[1],
               hidden_size=80,
               output_size=10)


iters_num = 2500
loss_list = []

iter_per_epoch = iters_num// batch_size

for i in range(iters_num):

    batch_mask = np.random.choice(x_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    net.gradient(x_batch, y_batch, LEARN_LATE)

    # for key in ('W1', 'b1', 'W2','b1'):
    #     net.params[key] -= grads[key] * LEARN_LATE

    if i % iter_per_epoch == 0:
        # print(f'epoch = {i} ______ time = {datetime.datetime.now()}')
        loss = net.loss(x_batch, y_batch)
        print('loss = ', loss,'\t\t\tacc =', net.accuracy(x_batch, y_batch))

        loss_list.append(loss)


my_x = x_test[77]
print('-----------------------------   DEEP PREDICT')
print('neuron_asnwer = ', np.argmax(net.my_predict(my_x)))
print('y_test  = ', y_test[77])
plt.imshow(my_x.reshape(28, 28))
plt.show()


# 소프트맥스 함수 ver 1로 교체해서 해보기. (이걸로 하니까 정확히나옴..)

# 이걸로 해보기 문제되는지x[self.zero_mask] = 0 -------Layers

# 함수를 그냥 내보내도 되는데 그거 변수 만들어서 내보낸것 변수 없이 그냥 return 해보기
