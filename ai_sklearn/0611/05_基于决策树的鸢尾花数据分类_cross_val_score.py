#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/12 14:28
# @Author  : Micky
# @Site    : 
# @File    : 05_基于决策树的鸢尾花数据分类_cross_val_score.py
# @Software: PyCharm

"""
cross_val_score：将测试集训练集划分，模型评估都封装起来了
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
    # 数据加载
    iris = pd.read_csv(filepath_or_buffer='../datas/iris.data',header=None,names=['c1','c2','c3','c4','y'])
    X = np.asarray(iris[['c1','c2','c3','c4']])
    Y = np.asarray(iris['y'])
    # 数据清洗、处理
    Y = LabelEncoder().fit_transform(Y)

    # 交叉验证的折数
    n_splite = 3
    # 创建模型对象
    algo = DecisionTreeClassifier(criterion = 'entropy' ,max_depth= 5, random_state= 214)

    # 返回几折对应的准确率
    result = cross_val_score(estimator=algo,X= X, y=Y, cv= n_splite)
    print(result)
    print('决策树算法在当前数据集上的评估效果为：{}'.format(result.mean()))

    # scoring 指定评估方式 模型为准确率和R2（分类/回归）<其实是模型对象的score方法>，可选值为:  https://scikit-learn.org/0.18/modules/model_evaluation.html#scoring-parameter
    result = cross_val_score(estimator=algo, X=X, y=Y, cv=n_splite,scoring='f1_macro')
    print(result)
    print('决策树算法在当前数据集上的评估效果为<F1值>：{}'.format(result.mean()))