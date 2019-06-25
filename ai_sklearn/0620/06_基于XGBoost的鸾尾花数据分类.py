#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/23 10:13
# @Author  : Micky
# @Site    : 
# @File    : 06_基于XGBoost的鸾尾花数据分类.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    iris = pd.read_csv(filepath_or_buffer='../datas/iris.data',header=None,names=['c1','c2','c3','c4','y'])
    X = np.asarray(iris[['c1','c2','c3','c4']])
    Y = np.asarray(iris['y'])

    Y = pd.Categorical(Y).codes

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=214)

    """
    def __init__(self, max_depth=3, learning_rate=0.1,
                 n_estimators=100, silent=True,
                 objective="binary:logistic", booster='gbtree',
                 n_jobs=1, nthread=None, gamma=0, min_child_weight=1,
                 max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, random_state=0, seed=None, missing=None, **kwargs):
    max_depth：决策树深度，当booster = ‘gbtree’
    learning_rate：学习率，累加预测值的时候，每个子模型会先乘以这个值在累加
    n_estimators：子模型数目             
    objective：损失函数
    """
    algo = XGBClassifier(max_depth=10,learning_rate=0.1,n_estimators=10)

    algo.fit(X_train,Y_train)

    # 7. 模型效果评估
    print("训练数据上的准确率:{}".format(algo.score(X_train, Y_train)))
    print("测试数据上的准确率:{}".format(algo.score(X_test, Y_test)))

    # 8. 看一下属性、API的信息
    x_test = [
        [6.9, 3.1, 5.1, 2.3],
        [6.1, 2.8, 4.0, 1.3],
        [5.2, 3.4, 1.4, 0.2]
    ]
    print("样本的预测值:")
    print(algo.predict(x_test))
    print("样本的预测概率值:")
    print(algo.predict_proba(x_test))
    print("各个特征属性的重要性权重列表(要求booster为gbtree):{}".format(algo.feature_importances_))