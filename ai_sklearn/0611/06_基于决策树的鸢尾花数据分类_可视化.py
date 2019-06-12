#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/12 22:54
# @Author  : Micky
# @Site    :
# @File    : 06_基于决策树的鸢尾花数据分类_可视化.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import pydotplus
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
if __name__ == '__main__':
    # 1. 数据加载
    iris = pd.read_csv(filepath_or_buffer='../datas/iris.data',header=None,names=['c1','c2','c3','c4','y'])
    X = np.asarray(iris[['c1','c2','c3','c4']])
    Y = np.asarray(iris['y'])
    # 2. 数据清洗、处理
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    # 3. 训练数据和测试数据划分
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 214)
    # 4. 特征工程
    # 5. 模型对象创建
    algo = DecisionTreeClassifier(criterion='gini')
    # 6. 模型训练
    algo.fit(X_train, Y_train)
    # 7. 模型效果评估
    print('训练集上的准确率：{}'.format(algo.score(X_train,Y_train)))
    print('测试集上的准确率：{}'.format(algo.score(X_test, Y_test)))

    # 8. 看下属性，API的信息
    x_test = [
        [6.9, 3.1, 5.1, 2.3],
        [6.1, 2.8, 4.0, 1.3],
        [5.2, 3.4, 1.4, 0.2]
    ]
    print("样本的预测值:")
    print(algo.predict(x_test))
    print("样本的预测概率值:")
    print(algo.predict_proba(x_test))
    print("样本的预测概率值的Log转换值:")
    print(algo.predict_log_proba(x_test))
    print("各个特征属性的重要性权重系数:{}".format(algo.feature_importances_))

    # 可视化
    # 方式一：将模型对象输出为dot文件，然后进行转换可视化操作
    # dot -Tpng iris.dot -o iris.png
    # dot -Tpdf iris.dot -o iris.pdf
    with open('iris.dot','w') as writer:
        tree.export_graphviz(decision_tree=algo,out_file=writer)

    # 方式二：直接输出图像或者pdf文件
    """
    feature_names：特征属性命名
    class_names：分类命名
    rounded：是否圆角
    filled：不同分类不同颜色
    """
    dot_data = tree.export_graphviz(decision_tree= algo,out_file=None,
                                    feature_names=['f1','f2','f3','f4'],
                                    class_names=['A','B','C'],
                                    rounded=True,filled=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('iris3.png')
    graph.write_pdf('iris3.pdf')