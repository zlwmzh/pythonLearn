#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/22 10:24
# @Author  : Micky
# @Site    : 
# @File    : 05_other_person_线性回归代码实现.py
# @Software: PyCharm

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import r2_score

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


class MyLinearRegression(object):
    def __init__(self, alpha=0.01, tol=1e-8, max_iter=100):
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.coef_ = None
        self.intercept_ = None
        pass

    def fit(self, X, Y):
        """
        模型训练
        :param X:
        :param Y:
        :return:
        """
        # 1.将X和Y转换为NumPy数组的形式
        X = np.asarray(X)
        Y = np.asarray(Y)

        # 2. 强制将Y转换为数组的形式
        Y = np.reshape(Y, -1)

        # 3. 获取样本数目以及特征属性的数目
        m, n = np.shape(X)

        # 4. 要求X中的样本数目和Y的样数目必须一致
        if np.shape(Y)[0] != m:
            raise Exception("x和y的样本数目不一致！！！！")

        # 5. 随机产生模型参数
        coef_ = np.random.normal(loc=0.0, scale=1.0, size=n)
        intercept_ = 0.0

        # 6. 计算当前的损失和变化量
        # 计算当前的损失函数的值
        predict_Y = self.__internel_predict(X, coef_, intercept_)
        current_loss = np.mean(np.square(predict_Y - Y))
        change = abs(current_loss) + self.tol
        # 构建变量累计迭代次数
        num_iter = 0

        while change >= self.tol and num_iter < self.max_iter:
            num_iter += 1
            # 1. 计算m个样本，每个样本实际值和预测值之间的差值
            errors = Y - predict_Y

            # 2. 更新一个一个theta值
            for j in range(n):
                delta = 0
                for i in range(m):
                    delta += errors[i] * X[i][j]
                coef_[j] = coef_[j] + self.alpha * 1.0 / m * delta

            # 3. 更新截距项
            intercept_ = intercept_ + self.alpha * np.mean(errors)

            # 4. 计算当前的变化量
            pre_loss = current_loss
            predict_Y = self.__internel_predict(X, coef_, intercept_)
            current_loss = np.mean(np.square(predict_Y - Y))
            change = abs(pre_loss - current_loss)

        # 更新模型参数
        self.coef_ = coef_
        self.intercept_ = intercept_

    def predict(self, X):
        return self.__internel_predict(X, self.coef_, self.intercept_)

    def __internel_predict(self, X, coef, intercept):
        # 1. 将权重转换为矩阵的形式
        coef = np.reshape(coef, (-1, 1))
        # 2. 直接构造预测值
        predict_Y = np.dot(X, coef) + intercept
        # 3. 将预测值的形状转换为数组的形状
        return np.reshape(predict_Y, -1)

    def score(self, X, Y):
        # 1.将X和Y转换为NumPy数组的形式
        X = np.asarray(X)
        Y = np.asarray(Y)
        Y = np.reshape(Y, -1)

        # 2. 获取当前样本对应的预测值
        predict_Y = self.predict(X)

        # 3. 计算RSS
        rss = np.sum(np.square(Y - predict_Y))

        # 4. 计算实际Y中的均值
        y_ = np.average(Y)

        # 5. 计算tss
        tss = np.sum(np.square(Y - y_))

        # 6. 计算r2
        r2 = 1 - rss / tss
        return r2


def r2_score2(Y, predict_Y):
    Y = np.asarray(Y)
    Y = np.reshape(Y, -1)
    predict_Y = np.asarray(predict_Y)
    predict_Y = np.reshape(predict_Y, -1)

    rss = np.sum(np.square(Y - predict_Y))

    # 4. 计算实际Y中的均值
    y_ = np.average(Y)

    # 5. 计算tss
    tss = np.sum(np.square(Y - y_))

    # 6. 计算r2
    r2 = 1 - rss / tss
    return r2


if __name__ == '__main__':
    # TODO: SGD和MBGD怎么实现？logistic回归怎么实现？
    np.random.seed(214)
    N = 10
    x = np.linspace(0, 6, N) + np.random.randn(N)
    y = 1.8 * x ** 3 + x ** 2 - 14 * x - 7 + np.random.randn(N)
    x.shape = -1, 1
    y.shape = -1, 1
    # N = 10
    # d = 2
    # x = np.linspace(0, 6, N * d).reshape((N, d))
    # x = x + np.random.randn(N, d)
    # y = np.dot(x ** 3, [[1.0], [2.0]]) \
    #     + np.dot(x ** 2, [[-5.0], [-3.2]]) \
    #     + np.dot(x, [[7.0], [2.0]]) \
    #     + np.random.randn(N, 1)
    # x.shape = -1, d
    # y.shape = -1, 1

    # 1. 使用LinearRegression
    lr = LinearRegression()
    lr.fit(x, y)
    lr_coef_ = lr.coef_
    lr_intercept_ = lr.intercept_
    print("LinearRegression模型的评估指标R2:{}".format(lr.score(x, y)))
    print("LinearRegression模型的模型参数:{}".format(lr_coef_))
    print("LinearRegression模型的截距项:{}".format(lr_intercept_))

    # 2. 使用LinearRegression
    sgd = SGDRegressor(max_iter=1000)
    sgd.fit(x, y)
    sgd_coef_ = sgd.coef_
    sgd_intercept_ = sgd.intercept_
    print("SGDRegressor模型的评估指标R2:{}".format(sgd.score(x, y)))
    print("SGDRegressor模型的模型参数:{}".format(sgd_coef_))
    print("SGDRegressor模型的截距项:{}".format(sgd_intercept_))
    print(r2_score(y, sgd.predict(x)))
    print(r2_score2(y, sgd.predict(x)))

    # 2. 使用LinearRegression
    mlr = MyLinearRegression(max_iter=1000)
    mlr.fit(x, y)
    mlr_coef_ = mlr.coef_
    mlr_intercept_ = mlr.intercept_
    print("MyLinearRegression模型的评估指标R2:{}".format(mlr.score(x, y)))
    print("MyLinearRegression模型的模型参数:{}".format(mlr_coef_))
    print("MyLinearRegression模型的截距项:{}".format(mlr_intercept_))
    print(r2_score(y, mlr.predict(x)))
    print(r2_score2(y, mlr.predict(x)))
