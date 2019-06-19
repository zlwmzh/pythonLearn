#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/19 11:50
# @Author  : Micky
# @Site    : 
# @File    : AdaBoost算法Homework.py
# @Software: PyCharm



import numpy as np

"""
计算信息熵
"""
def entropy(p):
    return np.sum([-t * np.log2(t) for t in p])




"""
推导α求解过程中底数为2的情况下，该案例的最终参数情况
X  0  1  2  3  4  5  6  7  8  9
Y  1  1  1 - 1  -1  -1  1  1  1  -1
w1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
"""
if __name__ == '__main__':
    # 第一轮
    # 计算整个数据集的信息熵
    h = entropy([0.6,0.4])
    # 可知划分节点；2.5，5.5，8.5
    # 以2.5作为划分
    h1 = 0.3 * entropy([1]) + 0.7 * entropy([0.4/0.7 ,0.3/0.7])

