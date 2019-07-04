#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/4 13:35
# @Author  : Micky
# @Site    : 
# @File    : 02_SVR回归效果查看.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as  plt
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

np.random.seed(214)
if __name__ == '__main__':
    # 产生数据
    # 产生20个在-5 ~ 5的一维数据
    x = np.random.rand(20,1) * 10 -5
    y = 1.5 * x + 0.4 + np.random.rand(20,1) * 5 - 1.5

    lr = LinearRegression(fit_intercept=True)
    lr.fit(x,y)

    # 查看线性回归的结果
    coef_ = lr.coef_[0]
    intercept_ = lr.intercept_
    print("查看线性回归的结果：")
    print('模型参数：{}'.format(coef_))
    print('截距项：{}'.format(intercept_))
    print('训练集上的模型效果：{}'.format(lr.score(x,y)))
    print('预测值：{}'.format(lr.predict(x).ravel()))
    print('实际值：{}'.format(y.ravel()))
    print('预测值与实际值之间的差值：\n{}'.format(lr.predict(x).ravel() - y.ravel()))

    algo = SVR(kernel='linear',C = 1.0,epsilon=2)
    algo.fit(x,y)

    # 查看SVR回归的结果
    coef2_ = algo.coef_[0]
    intercept2_ = algo.intercept_
    print("查看SVR的结果：")
    print('模型参数：{}'.format(coef2_))
    print('截距项：{}'.format(intercept2_))
    print('训练集上的模型效果：{}'.format(algo.score(x, y)))
    print('预测值：{}'.format(algo.predict(x).ravel()))
    print('实际值：{}'.format(y.ravel()))
    print('预测值与实际值之间的差值：\n{}'.format(algo.predict(x).ravel() - y.ravel()))

    plt.scatter(x,y)
    # 可视化线性回归
    plt.plot([-5,5],[-5 * coef_ + intercept_ , 5 * coef_ + intercept_],'r')
    plt.plot([-5, 5], [-5 * coef2_ + intercept2_, 5 * coef2_ + intercept2_], 'g')
    plt.plot(x[algo.support_],y[algo.support_],'r*')
    plt.show()
