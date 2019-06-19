#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/19 19:21
# @Author  : Micky
# @Site    : 
# @File    : 04_基于AdaBoost的鸢尾花数据分类.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import warnings
import pydotplus
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    # 数据加载
    iris = pd.read_csv(filepath_or_buffer='../datas/iris.data',header=None,names=['c1','c2','c3','c4','y'])
    X = np.asarray(iris[['c1','c2','c3','c4']])
    y = np.asarray(iris['y'])
    # 数据处理、清洗
    label_encode = LabelEncoder()
    y = label_encode.fit_transform(y)
    # 训练集和测试集划分
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=214)

    # 模型对象
    dtree = DecisionTreeClassifier(max_depth=1)

    """
             def __init__(self, 
                loss='deviance', 
                learning_rate=0.1, 
                n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None,
                 random_state=None, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False,
                 presort='auto'):

    """
    algo = GradientBoostingClassifier(learning_rate=0.1,n_estimators=2,max_depth=1,criterion='friedman_mse')
    # 模型训练
    algo.fit(X_train,y_train)
    # 模型效果评估
    print('训练集上的准确率：{}'.format(algo.score(X_train,y_train)))
    print('测试集上的准确率：{}'.format(algo.score(X_test, y_test)))

    x_test = [
        [6.9, 3.1, 5.1, 2.3],
        [6.1, 2.8, 4.0, 1.3],
        [5.2, 3.4, 1.4, 0.2]
    ]
    print('样本预测值：')
    print(algo.predict(x_test))
    print("样本的预测概率值:")
    print(algo.predict_proba(x_test))
    print("样本的预测概率值的Log转换值:")
    print(algo.predict_log_proba(x_test))

    print("训练好的所有子模型:\n{}".format(algo.estimators_))
    x_test = [
        [6.9, 3.1, 5.1, 2.3],
        [6.1, 2.8, 4.0, 1.3],
        [5.2, 3.4, 2.9, 0.8]
    ]
    generator = algo.staged_predict(x_test)
    print('阶段预测值：')
    for i in generator:
        print(i)
    print('各特征属性权重列表：{}'.format(algo.feature_importances_))

    # 所有子模型可视化
    for k, estimators in enumerate(algo.estimators_):
        for j, estimator in enumerate(estimators):
            dot_data = tree.export_graphviz(decision_tree=estimator, out_file=None,
                                            feature_names=['f1', 'f2', 'f3', 'f4'],
                                            class_names=['A', 'B', 'C'],
                                            rounded=True, filled=True,
                                            special_characters=True)
            graph = pydotplus.graph_from_dot_data(dot_data)
            graph.write_png("gbdt_{}_{}.png".format(k, j))
        if k > 2:
            break