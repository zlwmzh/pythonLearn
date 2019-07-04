#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/4 10:51
# @Author  : Micky
# @Site    : 
# @File    : 01_基于OneClassSVM的异常点检测.py
# @Software: PyCharm


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM

np.random.seed(214)
if __name__ == '__main__':
    # 产生正太分布的数据
    x = np.random.normal(size=(100,2))
    # 做一个缩放操作
    x = x* 0.2

    x1 = x + 2
    x2 = x - 2

    # 产生训练数据
    x_train = np.vstack((x1,x2))


    # 产生测试数据
    x_test = 0.2 * np.random.normal(size=(20,2))
    x_test = np.vstack((x_test+2,x_test-2))

    # 产生异常数据
    x_outliers = np.random.uniform(low=-3.5,high=3.5,size=(20,2))


    """
        nu：错误分类百分比
    """
    algo = OneClassSVM(kernel='rbf',gamma=0.1,nu=0.01)
    # algo = OneClassSVM(kernel='linear', nu=0.01)
    algo.fit(x_train)


    # 做预测，正常样本返回1，异常样本返回-1
    predict_train = algo.predict(x_train)
    print('训练数据上的预测值：{}'.format(predict_train))
    print('训练数据上预测失败的样本个数(异常样本):{}/{}'.format(np.sum(predict_train == -1),np.size(predict_train)))
    predict_test = algo.predict(x_test)
    print('测试数据上的预测值：{}'.format(predict_test))
    print('测试数据上预测失败的样本个数(异常样本):{}/{}'.format(np.sum(predict_test == -1),np.size(predict_test)))
    predict_outliers = algo.predict(x_outliers)
    print('异常数据上的预测值：{}'.format(predict_outliers))
    print('异常数据上预测失败的样本个数(异常样本):{}/{}'.format(np.sum(predict_outliers == -1), np.size(predict_outliers)))

    # decision_test = algo.decision_function(x_test)
    # print(decision_test)
    # 做一个可视化的区域图
    t1 = np.linspace(-4, 4, 50)
    t2 = np.linspace(-4, 4, 50)
    x1, x2 = np.meshgrid(t1, t2)
    x_show = np.dstack((x1.flat, x2.flat))[0]
    z = algo.decision_function(x_show)
    z.shape = x1.shape

    # 做一个可视化操作
    # 等高线区域图
    plt.contourf(x1, x2, z, cmap=plt.cm.Blues_r)
    # 画出训练数据点
    plt.plot(x_train[:,0],x_train[:,1],'ro')
    plt.plot(x_test[:, 0], x_test[:, 1], 'go')
    plt.plot(x_outliers[:, 0], x_outliers[:, 1], 'bo')
    plt.plot(algo.support_vectors_[:, 0], algo.support_vectors_[:, 1], 'y*')
    try:
        w1, w2 = algo.coef_[0]
        b = algo.intercept_[0]
        print((w1, w2, b))
        k = -w1 / w2
        b = -b / w2
        plt.plot([-4, 4], [-4 * k + b, 4 * k + b], 'k-')
    except:
        pass
    plt.show()