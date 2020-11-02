#!/user/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = './ex1data2.txt'
data = pd.read_csv(path, header=None, names=['square', 'bedrooms', 'pricw'])

# print(data.head())
# 参数正规化
def normalize_feature(df):
    return df.apply(lambda column : (column - column.mean())/column.std())

data = normalize_feature(data)

# 计算代价函数
def compute_cost(X, y, theta):
    inner = X @ theta.T - y
    square_sum = (inner.T @ inner)
    return square_sum[0]/(2*X.shape[0])

#构造 X
def get_X(df):
    ones = pd.DataFrame({'ones': np.ones(len(df))})
    data_new = pd.concat([ones, df], axis=1)
    return data_new.iloc[:, :-1].values

# 构造y
def get_y(df):
    return np.array(df.iloc[:, -1:])

# 梯度下降，迭代计算
def gradient_descent(X, y, theta, alpha, epoch):
    temp = theta.copy()
    cost = [compute_cost(X, y, theta)]
    m = X.shape[0]

    for i in range(epoch):
        error = X.T @ (X @ temp.T - y) / m
        temp = temp - alpha * error.T
        # print(compute_cost(X, y,temp))
        cost.append(compute_cost(X, y, temp))
    return temp, cost

X = get_X(data)
y = get_y(data)
theta = np.zeros((1, X.shape[1]))

theta_, cost = gradient_descent(X, y, theta, alpha=0.01, epoch=1500)

print(theta_)

def normal_equation(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta

print(normal_equation(X, y))

# plt.plot(np.arange(len(cost)), cost)
# plt.show()

#