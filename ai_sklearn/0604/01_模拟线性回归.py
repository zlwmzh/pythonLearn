#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/4 16:03
# @Author  : Micky
# @Site    : 
# @File    : 01_模拟线性回归.py
# @Software: PyCharm

import matplotlib.pylab as plt
import numpy as np

# 数据加载
X = np.random.randint(1,9,10)
Y = X + 1

plt.plot(X,Y,color = 'r')
plt.show()