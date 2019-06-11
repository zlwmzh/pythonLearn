#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/11 23:20
# @Author  : Micky
# @Site    :
# @File    : 03_基于决策树的鸢尾花数据分类_交叉验证.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    # 1. 数据加载
    iris = pd.read_csv('../datas/iris.data',header=None,names = ['c1','c2','c3','c4','t'])
    X = iris[['c1','c2','c3','c4']]
    Y = iris['t']
    X = np.asarray(X)
    Y = np.asarray(Y)
    # 2. 数据清洗，处理
    # 这里将目标属性字符串转换为int
    # LabelEncoder：将字符串类型的数据转为从0开始的序列
    # inverse_transform：反向转换，也就是数据恢复
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    # 3. 训练集、测试集划分
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 214)
    # 4. 特征工程
    # 5. 模型对象创建
    tree = DecisionTreeClassifier(random_state = 214)
    # 需要最优化的参数取值
    param_grid = {
        "criterion": ['gini', 'entropy'],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9]
    }
    """
     def __init__(self, 
                estimator, 分类器
                param_grid, 需要最优化的参数取值，传入值为字典或列表
                scoring=None, 模型评价标准
                fit_params=None,
                 n_jobs=1, 
                 iid=True, 
                 refit=True, 
                 cv=None, 交叉验证的折数
                 verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise',
                 return_train_score="warn"
    """
    algo = GridSearchCV(estimator=tree, param_grid= param_grid, cv=3, scoring=None)
    # 6. 模型训练
    algo.fit(X_train,Y_train)
    # 7. 模型评估
    print('训练集上的准确率：{}'.format(algo.score(X_train,Y_train)))
    print('测试集上的准确率：{}'.format(algo.score(X_test, Y_test)))
    # 8. 看下属性，API的信息
    print('当前最优参数：{}'.format(algo.best_params_))

