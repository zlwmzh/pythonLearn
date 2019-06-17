#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/17 21:21
# @Author  : Micky
# @Site    :
# @File    : 01_基于Baggin+KNN鸢尾花数据分类.py
# @Software: PyCharm

import warnings
import pandas as pd
import numpy as np
import pydotplus
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    # 数据加载：
    iris = pd.read_csv(filepath_or_buffer='../datas/iris.data',header=None,names=['c1','c2','c3','c4','y'])
    X = np.asarray(iris[['c1','c2','c3','c4']])
    Y = np.asarray(iris['y'])
    # 数据处理、清洗
    # 这里对Y值进行处理
    Y = pd.Categorical(Y).codes
    # 训练集和测试集划分
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=214)
    # 创建模型对象
    # KNN模型对象
    dtree = DecisionTreeClassifier(criterion='gini',max_depth=5)
    # Bagging对象
    """
    def __init__(self,
                 base_estimator=None, 基础子模型
                 n_estimators=10, 子模型个数
                 max_samples=1.0, 每个子模型对应训练数据的大小，该值为原始训练数据的占比。1.0表示每个子模型的训练数据样本数目和原始数据的样本数目是一样的
                 max_features=1.0, 每个子模型对应的特征属性占比，1.0表示每个子模型的特征属性和原始的特征属性是一样的
                                    如原始有4个特征属性，如果该值设置为0.5，则只有两个特征属性会参与模型训练
                 bootstrap=True, 每个子模型的训练数据的产生方式，True表示有放回的抽样，False表示不放回
                 bootstrap_features=False, 给定子模型的特征属性产生方式，True表示有放回采样，False表示不放回采样
                 oob_score=False, 是否计算袋外的评估值
                 warm_start=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
    """
    algo = BaggingClassifier(base_estimator= dtree,n_estimators=10,oob_score=True)
    # 模型训练
    algo.fit(X_train, Y_train)
    # 模型效果评估
    print('训练集上的准确率：{}'.format(algo.score(X_train,Y_train)))
    print('测试集上的准确率：{}'.format(algo.score(X_test, Y_test)))
    # 查看下API属性
    X_test = [
        [6.9, 3.1, 5.1, 2.3],
        [6.1, 2.8, 4.0, 1.3],
        [5.2, 3.4, 1.4, 0.2]
    ]
    print('样本的预测值：')
    print(algo.predict(X_test))
    print('样本预测值概率：')
    print(algo.predict_log_proba(X_test))
    print('样本预测概率值的Log转换：')
    print(algo.predict_log_proba(X_test))
  # print('训练好的所有子模型：{}'.format(algo.estimators_))

    for index,estimators in enumerate(algo.estimators_):
        print('第{}个子模型对于数据的预测值为：{}'.format(index+1,algo.predict(X_test)))
    # 就是有放回的抽样获取的数据子集
    print('每个子模型的训练数据：\n{}'.format(algo.estimators_samples_))
    print('每个子模型的训练数据使用的特征属性：\n{}'.format(algo.estimators_features_))
    print('Bagging模型的袋外准确率：\n{}'.format(algo.oob_score_))

    # 所有子模型可视化
    for index, estimators in enumerate(algo.estimators_):
        dot_data = tree.export_graphviz(decision_tree=estimators,out_file=None,
                                        feature_names=['c1','c2','c3','c4'],
                                        class_names=['A','B','C'],
                                        rounded=True,
                                        filled=True,
                                        special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_png('tree_bagging_{}.png'.format(index))
        if index > 2:
            break