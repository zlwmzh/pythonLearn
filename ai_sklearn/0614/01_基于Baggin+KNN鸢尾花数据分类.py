#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/14 13:41
# @Author  : Micky
# @Site    : 
# @File    : 01_基于Baggin+KNN鸢尾花数据分类.py
# @Software: PyCharm
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

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
    knn = KNeighborsClassifier(n_neighbors=5)
    algo = BaggingClassifier(base_estimator=knn,n_estimators=10,oob_score=True)
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
    print('各个子模型的训练数据使用的特征属性：{}'.format(algo.estimators_features_))
    print('Bagging模型的袋外准确率：{}'.format(algo.oob_score_))