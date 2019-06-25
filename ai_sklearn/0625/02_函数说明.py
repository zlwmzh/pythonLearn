#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/25 19:31
# @Author  : Micky
# @Site    : 
# @File    : 02_函数说明.py
# @Software: PyCharm

from sklearn.metrics.pairwise import pairwise_distances_argmin

"""
  计算X和Y中的所有样本的距离，默认为欧式距离，然后选择距离最近的样本下标并返回
  下面X中第一个样本与Y中最近的为下标1，X中第二个样本与Y中最近的下标为0，X中的第三个样本与Y中最近的下标为2
"""
order = pairwise_distances_argmin(X=[[1.0], [2.0], [3.0]],
                                   Y=[[1.6], [1.4], [3.6]])
print(order)