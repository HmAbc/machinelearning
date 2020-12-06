#!/user/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import classification_report

weightpath = './ex3weights.mat'
datapath = './ex3data1.mat'

# 读取训练数据，手写数字图片
def load_data(datapath):
    data = loadmat(datapath)
    X = data['X']
    X = np.asmatrix(np.insert(X, 0, values=np.ones(data['X'].shape[0]), axis=1))
    y = np.asmatrix(data['y'])
    return X, y

# 从mat文件读取训练好的权重数据
def load_weight(weightpath):
    weight = loadmat(weightpath)
    theta1, theta2 = weight['Theta1'], weight['Theta2']
    print(theta1.shape, theta2.shape)
    return theta1, theta2

# 前馈神经网络
def neural_network(X, theta1, theta2):
    a1 = X
    z2 = a1 * theta1.T
    print(z2.shape)
    a2 = sigmoid(z2)

    a2 = np.asmatrix(np.insert(a2, 0, values=np.ones(a2.shape[0]), axis=1))

    z3 = a2 * theta2.T
    print(z3.shape)
    a3 = sigmoid(z3)
    y_predict = np.argmax(a3, axis=1) + 1
    return y_predict
#
# print(theta1.shape, theta2.shape)

# sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    X, y = load_data(datapath)
    theta1, theta2 = load_weight(weightpath)
    y_predict = neural_network(X, theta1, theta2)
    print(classification_report(y, y_predict))