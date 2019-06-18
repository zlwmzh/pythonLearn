#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/18 11:46
# @Author  : Micky
# @Site    : 
# @File    : 05_基于RandomTreesEmbedding的鸢尾花数据增维.py
# @Software: PyCharm

import warnings
import pandas as pd
import numpy as np
import pydotplus
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.linear_model import LogisticRegression

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
                 n_estimators=10,  子模型个数
                 max_depth=5, 决策树最大深度
                 min_samples_split=2, 
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 sparse_output=True, 是否做稀疏矩阵输出
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
    """
    algo = RandomTreesEmbedding(n_estimators=100,max_depth=2,sparse_output=True)
    # 模型训练
    X_train2 = algo.fit_transform(X_train)
    print(X_train2)

    # 查看下API属性
    x_test2 = [
        [6.9, 3.1, 5.1, 2.3],
        [6.1, 2.8, 4.0, 1.3],
        [5.2, 3.4, 1.4, 0.2],
        [4.7, 3.2, 1.6, 0.2]
    ]
    print("样本的转换值:")
    print(algo.transform(x_test2))
    # # 模型效果评估
    # print('训练集上的准确率：{}'.format(algo.score(X_train, Y_train)))
    # print('测试集上的准确率：{}'.format(algo.score(X_test, Y_test)))

    print("训练好的所有子模型:\n{}".format(algo.estimators_))

    # 所有子模型可视化
    for k, estimator in enumerate(algo.estimators_):
        dot_data = tree.export_graphviz(decision_tree=estimator, out_file=None,
                                        feature_names=['f1', 'f2', 'f3', 'f4'],
                                        class_names=['A', 'B', 'C'],
                                        rounded=True, filled=True,
                                        special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_png("rte_{}.png".format(k))
        if k > 2:
            break

    # 可以基于扩维后的数据进行基础模型的构建
    lr = LogisticRegression()
    # TODO: 自己思考一下这里训练LR模型的时候，直接使用x_train2这个数据会不会存在某些问题？
    lr.fit(X_train2, Y_train)
    print("训练数据上的准确率:{}".format(lr.score(X_train2, X_train)))
    print("测试数据上的准确率:{}".format(lr.score(algo.transform(X_test), Y_test)))