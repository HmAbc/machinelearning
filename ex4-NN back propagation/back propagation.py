#!/user/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import classification_report

datapath = './ex4data1.mat'
weightpath = './ex4weights.mat'

def load_data(datapath):
    data = loadmat(datapath)
    X = data['X']
    y = data['y']
    return X, y

def load_weight(weightpath):
    weight = loadmat(weightpath)
    theta1 = weight['Theta1']
    theta2 = weight['Theta2']
    return theta1, theta2

def print_image(data):
    sample_idx = np.random.choice(np.arange(data.shape[0]), 100)
    sample = data[sample_idx, :]

    fig, pic = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True)
    for x in range(10):
        for y in range(10):
            pic[x, y].matshow(np.array((sample[10*x + y]).reshape(20, 20)).T, cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_propagate(theta1, theta2, X, m):
    X = np.insert(X, 0, values=np.ones(m), axis=1)
    # theta1 = np.asmatrix(theta1)
    # theta2 = np.asmatrix(theta2)

    a1 = X
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)

    return a1, z2, a2, z3, h

# 将标签转换为 one-hot 编码
def onehot(y):
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y)
    return y_onehot

def cost(params, X, y, input_size, hidden_size, label_num, learning_rate, m):
    X = np.asmatrix(X)
    y = np.asmatrix(y)

    theta1 = np.asmatrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, input_size + 1)))
    theta2 = np.asmatrix(np.reshape(params[hidden_size * (input_size + 1):], (label_num, hidden_size + 1)))

    a1, z2, a2, z3, h = forward_propagate(theta1, theta2, X, m)

    J = 0
    for i in range(m):
        first_term = np.multiply(- y[i, :], np.log(h[i, :]))
        second_term = np.multiply(1 - y[i, :], np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m

    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
    return J

"""下面是反向传播算法代码"""

def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), 1 - sigmoid(z))

def backprop(params, X, y, input_size, hidden_size, label_num, learning_rate, m):
    X = np.asmatrix(X)
    y = np.asmatrix(y)

    theta1 = np.asmatrix(np.reshape(params[:hidden_size * (input_size + 1)],
                                    (hidden_size, input_size + 1)))
    theta2 = np.asmatrix(np.reshape(params[hidden_size * (input_size + 1):],
                                    (label_num, hidden_size + 1)))

    a1, z2, a2, z3, h = forward_propagate(theta1, theta2, X, m)

    J = 0
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)
    for i in range(m):
        first_item = np.multiply(-y[i, :], np.log(h[i, :]))
        second_item = np.multiply(1 - y[i, :], np.log(1 - h[i, :]))
        J += np.sum(first_item - second_item)

    J = J / m

    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) +np.sum(np.power(theta2[:, 1:], 2)))

    for t in range(m):
        a1t = a1[t, :]
        z2t = z2[t, :]
        a2t = a2[t, :]
        z3t = z3[t, :]
        ht = h[t, :]
        yt = y[t, :]

        d3t = ht - yt

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)

        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    # add the gradient regularization term
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    print(J)
    return J, grad

def evaluate(params, X, y, m):
    X = np.asmatrix(X)

    theta1 = np.asmatrix(np.reshape(params[:hidden_size * (input_size + 1)],
                                    (hidden_size, input_size + 1)))
    theta2 = np.asmatrix(np.reshape(params[hidden_size * (input_size + 1):],
                                    (label_num, hidden_size + 1)))

    a1, z2, a2, z3, h = forward_propagate(theta1, theta2, X, m)

    y_predict = np.array(np.argmax(h, axis=1) + 1)
    print(classification_report(y, y_predict))

    # 另一种评估方式

    # correct = [1 if a == b else 0 for (a, b) in zip(y_predict, y)]
    # accuracy = (sum(map(int, correct)) / float(len(correct)))
    # print('accuracy = {0}%'.format(accuracy * 100))

if __name__ == '__main__':
    X, y = load_data(datapath)
    input_size = 400
    hidden_size = 25
    label_num = 10
    learning_rate = 1
    m = X.shape[0]
    y_onehot = onehot(y)
    params = (np.random.random(size=hidden_size * (input_size + 1) + label_num * (hidden_size + 1)) - 0.5) * 0.25

    # J, grad = backprop(params, X, y_onehot, input_size, hidden_size, label_num, learning_rate, m)

    fmin = minimize(fun=backprop, x0=params, args=(X, y_onehot, input_size, hidden_size, label_num, learning_rate, m),
                    method='TNC', jac=True, options={'maxiter': 250})

    evaluate(fmin.x, X, y, m)