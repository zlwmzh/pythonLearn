#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/3 17:28
# @Author  : Micky
# @Site    : 从文件加载数据
# @File    : 05_KNN鸾尾花数据分类2.py.py
# @Software: PyCharm
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# 1. 加载数据
iris = pd.read_csv(filepath_or_buffer='../datas/iris.data',header=None,names=['c1','c2','c3','c4','c5'])
X = iris[['c1','c2','c3','c4']]
Y = iris['c5']
X = np.asarray(X)
Y = np.asarray(Y)

# 2. 数据清洗、处理
label_name_indeo_dict = {'Iris-setosa':0,'Iris-virginica':1,'Iris-versicolor':2}
Y = map(lambda t : label_name_indeo_dict[t],Y)