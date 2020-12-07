#!/user/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
import matplotlib
import matplotlib.pyplot as plt

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

def forward_propagate(theta1, theta2, X):
    X = np.asmatrix(np.insert(X, 0, values=np.ones(X.shape[0]), axis=1))
    theta1 = np.asmatrix(theta1)
    theta2 = np.asmatrix(theta2)

    a1 = X
    z2 = sigmoid(a1 * theta1.T)
    a2 = np.insert(z2, 0, values=np.ones(z2.shape[0]), axis=1)
    z3 = sigmoid(a2 * theta2.T)

    y_predict = np.argmax(z3, axis=1) + 1
    return y_predict

# 将标签转换为 one-hot 编码
def onehot(y):
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y)
    return y_onehot

def cost(X, y, y_predict):





if __name__ == '__main__':
    X, y = load_data(datapath)
    # print_image(X)
