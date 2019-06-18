#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/18 23:16
# @Author  : Micky
# @Site    :
# @File    : 01_AdaBoost算法的过程理解代码.py
# @Software: PyCharm

import numpy as np

"""
计算信息熵
"""
def entropy(p):
    return np.sum([-t * np.log2(t) for t in p])

if __name__ == '__main__':
    # 计算所有数据的信息熵
    h = entropy([0.6,0.4])
    # 可知ppt上有三个划分节点2.5，5.5,8.5,分布计算其信息增益
    # 条件熵 2.5划分
    h1 = 0.3 * entropy([1]) + 0.7 * entropy([0.4/0.7,0.3/0.7])
    g1 = h - h1
    print(g1)
    # 5.5划分
    h2 = 0.6 * entropy([0.3/0.6,0.3/0.6]) + 0.4 * entropy([0.3/0.4,0.1/0.4])
    g2 = h - h2
    print(g2)
    # 8.5 划分
    h3 = 0.1 * entropy([1]) + 0.9 * entropy([0.6/0.9,0.3/0.9])
    g3 = h - h3
    print(g3)
    # 可知2.5划分最优
    # 如果以2.5划分的话，右子树有三个错误的预测，而每个预测的概率为0.1
    err1 = 0.3
    alpha = 0.5 * np.log((1-err1)/err1)
    print("第一个模型的权重系数：{}".format(alpha))
    # 正常样本权重变化（预测7个正常样本）
    w1 = 0.1 * np.e ** (-alpha)
    # 异常样本权重变化(预测失败3个样本)
    w2 = 0.2 * np.e ** alpha
    print(w1,w2)
    # 归一化