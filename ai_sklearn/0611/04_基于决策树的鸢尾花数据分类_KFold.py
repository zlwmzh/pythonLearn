#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/12 13:56
# @Author  : Micky
# @Site    : 
# @File    : 04_基于决策树的鸢尾花数据分类_KFold.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    # 1. 数据加载
    iris = pd.read_csv(filepath_or_buffer='../datas/iris.data',header=None,names=['c1','c2','c3','c4','y'])
    X = iris[['c1','c2','c3','c4']]
    Y = iris['y']
    X = np.asarray(X)
    Y = np.asarray(Y)
    # 2. 数据清洗、处理
    # 将目标属性字符串转换为int
    label_encoder = LabelEncoder();
    Y = label_encoder.fit_transform(Y)

    # 采用最原始的交叉验证
    # 几折交叉验证
    n_splits = 3
    """
    n_splits：交叉验证的折数
    shuffle：是否打乱数据的顺序
    """
    kfold = KFold(n_splits=3,shuffle = True,random_state=214)
    score = 0
    for train_idx, test_idx in kfold.split(X,Y):
        # 这里说明下：kfold.split 返回特征属性和目标属性的下标，以3折交叉验证为例，没有打乱数据顺序
        # 0-49 50- 99 做为训练集  100-149 做为测试集
        # 0-49 100-149 做为训练集  50 - 99 做为测试集
        # 50 - 99 100-149 做为训练集  0-49 做为测试集
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        # 进行模型构建，分布评估效果
        algo = DecisionTreeClassifier(criterion='entropy', max_depth= 5, random_state= 214)
        algo.fit(X_train,Y_train)
        # 效果评估
        print('训练集上的准确率：{}'.format(algo.score(X_train,Y_train)))
        print('测试集上的准确率：{}'.format(algo.score(X_test, Y_test)))
        print('==='*50)
        score += algo.score(X_test, Y_test)
    print('测试集上准确率均值：{}'.format(score / n_splits))
