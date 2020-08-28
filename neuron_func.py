
import numpy as np



def cross_entropy_error(predicts, labels):
    if predicts.ndim == 1:
        predicts = predicts.reshape(1, predicts.size)
        labels = labels.reshape(1, labels.size)

    batch_size = predicts.shape[0]
    return - (np.sum(labels * np.log(predicts + 1.0e-7)) / batch_size )

# --------------------- SOFT MAX
def softmax(inputs):
    max_val = np.max(inputs)

    minus_exp = np.exp(inputs - max_val)
    return minus_exp / np.sum(minus_exp)

def softmax_v01(inputs):
    # max 결과가 가로로 나오니 세로로 만든다
    maxs = np.max(inputs, axis=1).reshape(-1, 1)
    minus_exp = np.exp(inputs - maxs)
    # sum 결과가 가로로 나오니 세로로 만든다
    sum_exps = np.sum(minus_exp, axis=1).reshape(-1, 1)
    return minus_exp / sum_exps

# --------------------- SIGMOID

def sigmoid(inputs):
    return 1 / (1 + np.exp(inputs))

# --------------------- Numerical Gradient
def one_numerical_gradient(f, x):
    h = 1.0e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        temp = x[idx]

        x[idx] = temp + h
        fxh1 = f(x)

        x[idx] = temp - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = temp

    return grad



def numerical_gradient(f, X):
    if X.ndim == 1:
        return one_numerical_gradient(f, X)
    else:
        grads = np.zeros_like(X)
        for idx, data in enumerate(X):

            grads[idx] = one_numerical_gradient(f, data)
        return grads
