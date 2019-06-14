#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/14 16:11
# @Author  : Micky
# @Site    : 
# @File    : 03_基于RF的鸾尾花数据分类.py
# @Software: PyCharm

import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    # 数据加载
    iris = pd.read_csv('../datas/iris.data',header=None,names=['c1','c2','c3','c4','y'])
    X = np.asarray(iris[['c1','c2','c3','c4']])
    y = np.asarray(iris['y'])
    # 数据清洗
    y = pd.Categorical(y).codes
    # 训练集测试集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2,random_state=214)
    # 创建模型对象
    """
    def __init__(self,
                 n_estimators=10,  给定子模型的数据
                 criterion="gini", 给定底层决策数据构建过程中的纯度的衡量指标，可选值gini和entropy
                 max_depth=None, 前置的剪枝参数，限制决策树的深度，None表示不限制
                 min_samples_split=2,剪枝参数，用于限制决策树划分的最低要求。对于某个节点进行划分时，要求该节点的样本数量至少为该值(要求去重后的数据，一个样本只计算一个)
                 min_samples_leaf=1,剪枝参数，用于限制叶子节点的样本数目，要求划分之后叶子节点的样本数目必须大于等于该值
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0., 只有当划分前后的纯度的增益值超过该值，当前划分才有效
                 min_impurity_split=None,
                 bootstrap=True,  给定每个数据子集的产生方式 True标识有放回的抽样产生，False表示没有放回的抽样产生
                 oob_score=False, 当进行有放回的抽样产生训练数据的时候，该参数才可以设置为True，标示计算袋外的评估指标
                 n_jobs=1, 启动几个线程
                 random_state=None,
                 verbose=0,是否打印日志 0 标示不打印
                 warm_start=False,
                 class_weight=None):
                 """
    algo = RandomForestClassifier(n_estimators=10,oob_score=True)
    # 模型训练
    algo.fit(X_train,y_train)
    # 7. 模型效果评估
    print('训练集上数据准确率：{}'.format(algo.score(X_train, y_train)))
    print('测试集上数据准确率：{}'.format(algo.score(X_test, y_test)))
    # 8. 看下属性API
    X_test = [
        [6.9, 3.1, 5.1, 2.3],
        [6.1, 2.8, 4.0, 1.3],
        [5.2, 3.4, 1.4, 0.2]
    ]
    print('样本预测值：')
    print(algo.predict(X_test))
    print('样本的预测概率：')
    print(algo.predict_proba(X_test))
    print('样本预测概率值得log转换值：')
    print(algo.predict_log_proba(X_test))

    print('训练好的所有子模型:\n{}'.format(algo.estimators_))
    X_test = [
        [6.9, 3.1, 5.1, 2.3],
        [6.1, 2.8, 4.0, 1.3],
        [5.2, 3.4, 1.4, 0.2]
    ]
    for k,estimators in enumerate(algo.estimators_):
        print('第{}个子模型对于数据的预测值为：{}'.format(k+1, estimators.predict(X_test)))
    print('各个特征属性的重要性权重列表：{}'.format(algo.feature_importances_))
    print('Bagging模型的袋外准确率：{}'.format(algo.oob_score_))