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
from sklearn.ensemble import ExtraTreesClassifier

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
    algo =ExtraTreesClassifier(n_estimators=100,
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
        graph.write_png('extral_trr_{}.png'.format(index))
        if index > 2:
            break