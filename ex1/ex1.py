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
    return np.sum(inner) / 2*len(X)

# 构造 X 矩阵
def get_X(data):
    ones = pd.DataFrame('ones', np.ones(len(data)))
    data_new = pd.concat([ones, data], axis=1)
    return data_new.iloc[:, :-1].as_matrix()

# 构造 y
def get_y(data):
    return np.array(data.iloc[:, -1:])

# 特征缩放 mean 函数求均值，std 函数求标准差
def normalize_feature(data):
    return data.apply(lambda column : (column - column.mean()) / column.std())

