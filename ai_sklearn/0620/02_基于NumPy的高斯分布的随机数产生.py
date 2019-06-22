#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/21 22:46
# @Author  : Micky
# @Site    : 
# @File    : 02_基于NumPy的高斯分布的随机数产生.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    # 产生N个均值为loc，标准差为scale的正态分布随机数列
    N = 10000
    datas = np.random.normal(loc=100,scale=1.0,size=N)

    # 9-1 / 3
    # a = plt.hist(x=[1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9], bins=3)
    # print(a)
    # plt.show()
    plt.hist(x = datas,bins= 100)
    plt.show()
