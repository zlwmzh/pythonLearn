#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/2 16:34
# @Author  : Micky
# @Site    : 
# @File    : 01_Logistic和SVM效果比较.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.datasets import make_circles



def f1():
    # 1. 产生数据
    np.random.seed(214)
    x1 = np.random.rand(10, 2) * 5
    x2 = np.random.rand(10, 2) * -5

    # 对于x1的数据集产生对应的目标属性
    y1 = np.zeros(shape=x1.shape[0])
    # 对于x2的数据集产生对应的目标属性
    y2 = np.ones(shape=x2.shape[0])

    # 两个样本数据集合并
    x = np.vstack((x1, x2))
    y = np.hstack((y1, y2))

    # 2. 两个分类找一个分割线
    algo = LogisticRegression()
    algo.fit(x, y)

    # 获取模型参数
    coef_ = algo.coef_[0]
    # 获取截距项
    intercept_ = algo.intercept_

    # 可视化相关操作
    # 这里取x2作为y，x1做为x
    k1 = - coef_[0] / coef_[1]
    b1 = -intercept_[0] / coef_[1]

    print('线性回归的系数：{}'.format(coef_))
    print('线性回归的截距项：{}'.format(intercept_))
    print('模型效果：{}'.format(algo.score(x, y)))

    algo2 = LinearSVC()
    algo2.fit(x, y)
    coef2_ = algo2.coef_[0]
    intercept2_ = algo2.intercept_
    k2 = - coef2_[0] / coef2_[1]
    b2 = -intercept2_[0] / coef2_[1]

    print('LinearSVC的系数：{}'.format(coef2_))
    print('LinearSVC的截距项：{}'.format(intercept2_))
    print('LinearSVC模型效果：{}'.format(algo2.score(x, y)))

    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.plot([-6, 6], [-6 * k1 + b1, 6 * k1 + b1], 'r-')
    plt.plot([-6, 6], [-6 * k2 + b2, 6 * k2 + b2], 'y-')
    plt.show()

def f2():
    # 1. 产生数据
    np.random.seed(214)
    x1 = np.random.rand(10, 2) * 5
    x2 = np.random.rand(10000, 2) * -5

    # 对于x1的数据集产生对应的目标属性
    y1 = np.zeros(shape=x1.shape[0])
    # 对于x2的数据集产生对应的目标属性
    y2 = np.ones(shape=x2.shape[0])

    # 两个样本数据集合并
    x = np.vstack((x1, x2))
    y = np.hstack((y1, y2))

    # 2. 两个分类找一个分割线
    algo = LogisticRegression()
    algo.fit(x, y)

    # 获取模型参数
    coef_ = algo.coef_[0]
    # 获取截距项
    intercept_ = algo.intercept_

    # 可视化相关操作
    # 这里取x2作为y，x1做为x
    k1 = - coef_[0] / coef_[1]
    b1 = -intercept_[0] / coef_[1]

    print('线性回归的系数：{}'.format(coef_))
    print('线性回归的截距项：{}'.format(intercept_))
    print('模型效果：{}'.format(algo.score(x, y)))

    algo2 = SVC(kernel='linear',C = 0.1)
    algo2.fit(x, y)
    coef2_ = algo2.coef_[0]
    intercept2_ = algo2.intercept_
    k2 = - coef2_[0] / coef2_[1]
    b2 = -intercept2_[0] / coef2_[1]

    support_vectors = algo2.support_vectors_
    print('SVC的系数：{}'.format(coef2_))
    print('SVC的截距项：{}'.format(intercept2_))
    print('SVC模型效果：{}'.format(algo2.score(x, y)))
    print('SVC支持向量：{}'.format(support_vectors))


    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.plot([-6, 6], [-6 * k1 + b1, 6 * k1 + b1], 'r-')
    plt.plot([-6, 6], [-6 * k2 + b2, 6 * k2 + b2], 'y-')
    plt.plot(support_vectors[:,0],support_vectors[:,1],'b*')
    plt.show()

def f3():
    # 1. 产生数据
    x,y = make_circles(n_samples=100,random_state=214)

    algo2 = SVC(kernel='linear', C=0.1)
    algo2.fit(x, y)
    coef2_ = algo2.coef_[0]
    intercept2_ = algo2.intercept_
    k2 = - coef2_[0] / coef2_[1]
    b2 = -intercept2_[0] / coef2_[1]

    support_vectors = algo2.support_vectors_
    print('SVC的系数：{}'.format(coef2_))
    print('SVC的截距项：{}'.format(intercept2_))
    print('SVC模型效果：{}'.format(algo2.score(x, y)))
    print('SVC支持向量：{}'.format(support_vectors))

    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.plot([-6, 6], [-6 * k2 + b2, 6 * k2 + b2], 'y-')
    plt.plot(support_vectors[:, 0], support_vectors[:, 1], 'b*')
    plt.show()

def f4():
    x, y = make_circles(n_samples=100,random_state=214,noise=0.05,factor=0.5)

    poly = PolynomialFeatures(degree=2)
    # 获取扩展之后
    poly_x = poly.fit_transform(x,y)
    # 为了可视化，取x1^2，x2^2
    poly_x = poly_x[:,[3,5]]
    print("扩展的组合方式：{}".format(poly.get_feature_names()))

    algo2 = SVC(kernel='linear',C = 10000)
    algo2.fit(poly_x,y)
    coef_ = algo2.coef_[0]
    intercept_ = algo2.intercept_
    k1 = - coef_[0]/ coef_[1]
    b1 = -intercept_ / coef_[1]
    print("SVM的系数:{}".format(coef_))
    print("SVM的截距项:{}".format(intercept_))
    print("模型效果:{}".format(algo2.score(poly_x, y)))

    plt.plot([-0.5, 1], [-0.5 * k1 + b1, 1 * k1 + b1], 'b-')
    plt.scatter(poly_x[:,0],poly_x[:,1],c = y)
    plt.plot(algo2.support_vectors_[:,0],algo2.support_vectors_[:,1],'r*')
    plt.grid(True)
    plt.show()

def f5():
    # 1. 产生数据
    x, y = make_circles(n_samples=100,random_state=214,noise=0.05,factor=0.5)
    algo2 = SVC(kernel='poly',C = 1, degree=2,probability=True)
    algo2.fit(x,y)
    print("模型效果：{}".format(algo2.score(x,y)))
    print("预测值：{}".format(algo2.predict(x)))
    print("决策函数的值：\n{}".format(algo2.decision_function(x)))
    print("预测为各个类别的概率(要求probablity为True)：\n{}".format(algo2.predict_proba(x)))

    plt.scatter(x[:,0],x[:,1],c = y)
    plt.plot(algo2.support_vectors_[:,0],algo2.support_vectors_[:,1],'r*')
    plt.show()

if __name__ == '__main__':
    f5()
