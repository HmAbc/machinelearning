#!/user/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 这两句代码输出一个5阶单位矩阵
A = np.eye(5)
print(A)

# 这三句代码表示从文件读取数据保存为 dataFrame
path = './ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

# 这两句代码将数据在平面坐标系中画出来，x轴表示人口，y轴表示利润
# data.plot(kind='scatter', x='Population', y='Profit')
# plt.show()


# 定义代价函数
def compute_cost(X, y, theta):
    inner = X @ theta.T - y
    square_sum = inner.T @ inner
    return square_sum / (2*X.shape[0])

# 构造 X 矩阵
def get_X(df):
    ones = pd.DataFrame({'ones': np.ones(len(df))})
    data_new = pd.concat([ones, df], axis=1)
    return data_new.iloc[:, :-1].values

# 构造 y
def get_y(df):
    return np.array(df.iloc[:, -1:])

# 特征缩放 mean 函数求均值，std 函数求标准差
def normalize_feature(df):
    return df.apply(lambda column : (column - column.mean()) / column.std())

X = get_X(data)
y = get_y(data)
# X = np.asmatrix(get_X(data))
# y = np.asmatrix(get_y(data))
theta = np.zeros((1, X.shape[1]))

# print(compute_cost(X, y, theta))
# print(theta.ravel().shape[1])

# 梯度递减函数，迭代实现
def gradient_descent(X, y, theta, alpha, iters):
    temp = theta.copy()
    cost = [compute_cost(X, y, theta)[0]]
    m = X.shape[0]

    for i in range(iters):
        error = X.T @ (X @ temp.T - y) / m
        temp = temp - alpha * error.T
        cost.append(compute_cost(X, y, temp)[0])
    return temp, cost

alpha = 0.01
iters = 500
theta_, cost = gradient_descent(X, y, theta, alpha, iters)

print(np.array(cost))
# print(cost)
# x = np.linspace(data.Population.min(), data.Population.max(), 100)
# f = theta_[0] + (theta_[1] * x)


# plt.plot(x, f, 'r', label='Prediction')
# plt.scatter(data.Population, data.Profit, label='Traning Data')
# plt.legend(loc=2)
# plt.xlabel('Population')
# plt.ylabel('Profit')
# plt.title('Predicted Profit vs. Population Size')
# plt.show()

# fig, ax = plt.subplots()
# ax.plot(x, f, 'r', label='Prediction')
# ax.scatter(data.Population, data.Profit, label='Traning Data')
# ax.legend(loc=2)
# ax.set_xlabel('Population')
# ax.set_ylabel('Profit')
# ax.set_title('Predicted Profit vs. Population Size')
# plt.show()



# x = range(0,501)
# plt.plot(x, cost)
# plt.xlabel('iterations')
# plt.ylabel('cost')
# plt.show()