#!/user/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

data = pd.read_csv('./ex2data1.txt', header=None, names=['exam1', 'exam2', 'y'])

# 将要分类的簇画出来
def plot_data(df):
    positive = df[df['y'].isin([1])]
    negative = df[df['y'].isin([0])]

    print(len(negative['exam1']), len(negative['exam2']))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(positive['exam1'], positive['exam2'], s=50, c='r', marker='+', label='Admitted')
    ax.scatter(negative['exam1'], negative['exam2'], s=50, c='b', marker='o', label='Not Admitted')
    ax.legend()
    ax.set_xlabel('Exam 1 score')
    ax.set_ylabel('Exam 2 score')
    plt.show()

# plot_data(data)

# 要拟合的曲线
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 代价函数
def cost(theta, X, y):
    X = np.asmatrix(X)
    y = np.asmatrix(y)
    theta = np.asmatrix(theta)
    htheta = sigmoid(X @ theta.T)
    temp = np.multiply(-y, np.log(htheta)) - np.multiply(1-y, np.log(1-htheta))
    return np.sum(temp) / X.shape[0]

# 返回特征值和监督信息
def get_X_y(df):
    df.insert(0, 'one', 1)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values
    return X, y

# 函数定义了一次梯度计算的值
def gradient(theta, X, y):
    theta = np.asmatrix(theta)
    X = np.asmatrix(X)
    y = np.asmatrix(y)
    return 1/len(X) * X.T @ (sigmoid(X @ theta.T) - y)

def predict(theta, X):
    prob = sigmoid(X @ theta.T)
    return [1 if x >= 0.5 else 0 for x in prob]

X, y = get_X_y(data)
theta = np.zeros(X.shape[1])

# 使用优化算法找出最优参数
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))

theta_min = result[0]

y_predit = predict(theta_min, X)

correct = [1 if ((a==1 and b==1) or (a==0 and b==0)) else 0 for (a, b) in zip(y, y_predit)]
accuracy = np.sum(correct)/len(correct)
print("accuracy = {}%".format(accuracy * 100))