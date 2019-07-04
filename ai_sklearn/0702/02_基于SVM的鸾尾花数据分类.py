#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/2 21:55
# @Author  : Micky
# @Site    : 
# @File    : 02_基于SVM的鸾尾花数据分类.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

if __name__ == '__main__':

    # 数据加载
    iris = pd.read_csv(filepath_or_buffer='../datas/iris.data',header=None,names=['c1','c2','c3','c4','y'])
    x = np.asarray(iris[['c1','c2','c3','c4']])
    y = np.asarray(iris['y'])
    # 数据清洗
    y = pd.Categorical(y).codes
    # 测试数据和训练数据划分
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=214)

    # 构建模型对象
    """
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=None):
                 C: 惩罚项系数，该值越大，表示模型要求训练集数据尽可能的预测正确，间隔要求尽可能小，容易过拟合；当该值比较小的时候
                    表示对于样本预测错误的惩罚比较小，间隔较大，容易欠拟合
                 kernel: 核函数。 一般选择linear、rbf和poly,最常用的linear、rbf
                 degree: 控制poly核函数的维度，相当于多项式扩展最多允许扩展多少阶。该值越大，相当于映射的维度越高，越容易过拟合   
                 gamma:  控制核函数的映射范围，gamma在rbf中用于控制每个样本对于周边多大范围的样本会存在影响；gamma比较大的时候，相当于
                        模型非常关注局部的细节特征，模型会学习到更多的细节信息；而gamma比较小的时候，相当于模型训练关注比较大范围的特征
                        信息。一般范围：1e-6/1.0,  而且一般建议设置为1/n_features
                 Note：如果过拟合，可以考虑降低c和gamma的值，如果欠拟合，可以考虑增加c和gamma的值       
    """
    algo = SVC(kernel='rbf',C = 1.0, gamma=0.2,probability=True)

    # 模型训练
    algo.fit(x_train,y_train)

    # 模型效果评估
    print('训练集上的模型效果：{}'.format(algo.score(x_train,y_train)))
    print('测试集上的模型效果：{}'.format(algo.score(x_test,y_test)))

    # 8. 看一下属性、API的信息
    x_test = [
        [6.9, 3.1, 5.1, 2.3],
        [6.1, 2.8, 4.0, 1.3],
        [5.2, 3.4, 1.4, 0.2]
    ]

    print('样本预测值：')
    print(algo.predict(x_test))
    print('样本预测概率值：')
    print(algo.predict_proba(x_test))
    print('支持向量在训练数据中对应的下标：{}'.format(algo.support_))
    print('支持向量的（样本向量）：\n{}'.format(algo.support_vectors_))
    print('各类别支持向量的数据：{}'.format(algo.n_support_))

