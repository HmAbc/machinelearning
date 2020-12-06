#!/user/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import minimize
from sklearn.metrics import classification_report

data = loadmat('./ex3data1.mat')
m = data['X'].shape[0]

# print(data['X'].shape, data['y'].shape)
# 随机从数据中选择100个
def random_choice(data):
    sample_idx = np.random.choice(np.arange(data['X'].shape[0]), 100)
    sample_image = data['X'][sample_idx, :]
    return sample_image

# 画图展示随机选取的样本
def show_images(sample_image):
    fig, ax_array = plt.subplots(ncols=10, nrows=10, sharex=True, sharey=True, figsize=(12, 12))
    for x in range(10):
        for y in range(10):
            ax_array[x, y].matshow(np.array(sample_image[10*x + y].reshape(20, 20)).T, cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

# # 调用函数展示100张图像
# sample_image = random_choice(data)
# show_images(sample_image)
# plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义代价函数
def cost(theta, X, y, learnrating, m):
    theta = np.asmatrix(theta)
    X = np.asmatrix(X)
    y = np.asmatrix(y)

    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply(1 - y, np.log(1 - sigmoid(X * theta.T)))
    reg = learnrating * np.sum(np.power(theta[:, 1:], 2)) / (2 * m)
    return np.sum(first - second) / m + reg

# 定义梯度下降函数
def gradient(theta, X, y, learnrating, m):
    theta = np.asmatrix(theta)
    X = np.asmatrix(X)
    y = np.asmatrix(y)

    parameter = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y

    grad = ((X.T * error) / m).T + (learnrating / m) * theta
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / m
    return np.array(grad).ravel()

# 一对多分类
def one_vs_all(X, y, num_label, learningrate, m):
    rows = X.shape[0]
    params = X.shape[1]

    all_theta = np.zeros((num_label, params + 1))

    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    for i in range(1, num_label + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i,(rows, 1))

        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learningrate, m), method='TNC', jac=gradient)
        all_theta[i-1, :] = fmin.x

    return all_theta

all_theta = one_vs_all(data['X'], data['y'], 10, 1, m)


def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # convert to matrices
    X = np.asmatrix(X)
    all_theta = np.asmatrix(all_theta)

    # compute the class probability for each class on each training instance
    h = sigmoid(X * all_theta.T)

    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)

    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1

    return h_argmax

y_pred = predict_all(data['X'], all_theta)
print(classification_report(data['y'], y_pred))