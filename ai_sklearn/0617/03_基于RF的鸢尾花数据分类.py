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
from sklearn.ensemble import RandomForestClassifier

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
    # Bagging对象
    """
   def __init__(self,
                 n_estimators=10,  给定子模型的个数
                 criterion="gini", 给定底层决策树构建构成中纯度衡量指标，可选值gini和entropy
                 max_depth=None, 前置减枝参数，用于决策树的深度，None表示不限制
                 min_samples_split=2, 前置剪纸参数，用于限制决策树划分的最低要求，对于某个节点进行划分的时候，要求当前节点中的样本数据至少为该值
                                    (要求是去重后的数据，一个样本只计算一个)
                 min_samples_leaf=1,前置剪纸参数，用于限制叶子节点中的样本数目，要求划分子之后的叶子节点中的样本数目必须大于等于该值
                                    (要求是去重后的数据，一个样本只计算一个)
                 min_weight_fraction_leaf=0.,前置剪纸参数
                 max_features="auto", 给定决策树划分阶段选择的时候，具体从多少个特征属性中选择最优的特征属性
                 max_leaf_nodes=None, 剪纸参数，最多允许的叶子节点数目，None表示不限制
                 min_impurity_decrease=0.,前置剪纸参数，只有当前划分前后的纯度的增益值超过该值，当前划分才有效
                 min_impurity_split=None, 前置剪纸参数，如果一个节点要进行划分，要求gini系数或entropy至少为该值。None表示不限制
                 bootstrap=True, 给定每个数据子集产生方式，True表示有放回的抽样产生，False表示不放回的抽样产生
                 oob_score=False, 当进行有放回的抽样产生训练数据的时候，该参数才可以设置为True，表示计算袋外的误差
                 n_jobs=1, 启动几个线程
                 random_state=None, 随机数种子
                 verbose=0, 是否打印日志，0表示不打印
                 warm_start=False, 
                 class_weight=None)
    """
    algo = RandomForestClassifier(n_estimators=100,
                                  oob_score=False,
                                  criterion='entropy',
                                  max_depth=None,
                                  min_samples_split=2,
                                  min_samples_leaf=1,
                                  bootstrap=False,
                                  max_features='auto')
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
    print('各个特征属性的重要性权重列表：\n{}'.format(algo.feature_importances_))
    # print('Bagging模型的袋外准确率：\n{}'.format(algo.oob_score_))

    # 所有子模型可视化
    for index, estimators in enumerate(algo.estimators_):
        dot_data = tree.export_graphviz(decision_tree=estimators,out_file=None,
                                        feature_names=['c1','c2','c3','c4'],
                                        class_names=['A','B','C'],
                                        rounded=True,
                                        filled=True,
                                        special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_png('tree_rf_{}.png'.format(index))
        if index > 2:
            break