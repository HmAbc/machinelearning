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
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2*len(X))

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

X = np.asmatrix(get_X(data))
y = np.asmatrix(get_y(data))
theta = np.asmatrix([0, 0])
# print(X.shape, y.shape)

print(compute_cost(X, y, theta))

def gradient_descent(X, y, theta, alpha, iters):
