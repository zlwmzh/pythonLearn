#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/21 23:00
# @Author  : Micky
# @Site    : 
# @File    : 03_基于NumPy随机数产生.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(214)
if __name__ == '__main__':
    N = 10
    # 0-6之间产生N个数，随机的
    x = np.linspace(0,6,N) + np.random.randn(N)
    # 产生y值，加入误差项
    y = 1.8 * x ** 3 + x ** 2 - 14 * x - 7 + np.random.randn(N)
    x.shape = -1,1  # 将x 的形状改为1列任意行，一行代表一个样本，一列代表一个特征属性
    y.shape = -1,1

    # 画点
    plt.plot(x,y,'ro')
    plt.show()