#!/user/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

theta = np.array([1,2,3])
print(theta)

a = np.matrix('1,2,3;1,2,3')
print(a)

b = a @ theta
print(b)

print(a.shape, b.shape, theta.shape)
print(type(theta), type(a), type(b))