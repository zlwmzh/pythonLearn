#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/21 23:10
# @Author  : Micky
# @Site    : 
# @File    : 04_基于梯度下降实现线性回归算法.py
# @Software: PyCharm

import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,SGDRegressor

warnings.filterwarnings('ignore')

"""
封装基于梯度下降的线性回归算法
提供BGD、SGD、MBGD方式进行迭代
"""
class MickyLinearRegression(object):
    """
    alpha ： 学习率/步长
    tol：预测值与实际值的最小误差，如果小于该值，函数收敛
    max_iter : 最大迭代次数
    iterta_type ： 迭代方式，可选值BGD，SGD，MBGD
    """
    def __init__(self,alpha=0.01,tol=1e-8,max_iter = 1000,iterat_type = 'BGD'):
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.iterat_type = iterat_type
        self.coef_ = None
        self.intercept_ = None
        self.coef_c = None

    """
    X：样本特征属性
    y：样本目标属性
    """
    def fit(self,X,y):
        # 1.转换为numpy数组形式
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        # 2. 将目标属性转为为1维的数组，方便后续操作
        self.y = np.reshape(self.y,-1)
        # 3. 判断目标属性的个数是否和样本个数一致，如果不一致抛出异常
        sample_count,feature_count = self.X.shape
        target_count = self.y.shape[0]
        # print('样本个数：{}'.format(sample_count))
        # print('样本特征属性：{}'.format(feature_count))
        # print('样本目标属性：{}'.format(target_count))
        if sample_count != target_count:
            raise Exception('x and y is no samle')
        # 模型参数初值,这是使用随机值,特征属性多少个就产生多少个模型参数
        self.coef_ = np.random.normal(loc=0.0,scale=1.0,size=feature_count)
        # 模型截距项,初始值为0
        self.intercept_ = 0.0

        # loss = (预测值- 实际值)**2
        predict_y = self._internel_predict(self.X,self.coef_,self.intercept_)
        # print(self.y)
        # print(predict_y - self.y)
        current_loss = np.sum(np.square(predict_y - self.y)) / 2*sample_count
        # current_loss = np.mean(np.square(predict_y - self.y))
        change_loss = np.abs(current_loss) + self.tol
        # print(change_loss)
        if self.iterat_type == 'BGD':
            self.__bgd(predict_y, current_loss, change_loss, feature_count, sample_count)
        elif self.iterat_type == 'SGD':
            self.__sgd(predict_y,current_loss, change_loss, feature_count, sample_count)
        else:
            self.__mbgd(predict_y,current_loss, change_loss, feature_count, sample_count)


    """
    BGD方式迭代
    predict_y : 预测值
    current_loss : 当前损失函数
    change_loss：损失函数变换量
    feature_count：特征属性数量
    sample_count：样本个数
    """
    def __bgd(self,predict_y,current_loss,change_loss,feature_count,sample_count):
        num_iter = 0
        while change_loss >= self.tol and num_iter < self.max_iter:
            self.__coef_intercept_bgd(predict_y,feature_count,sample_count)
            # 计算变化量
            pre_loss = current_loss
            predict_y = self._internel_predict(self.X, self.coef_, self.intercept_)
            current_loss = np.sum(np.square(predict_y - self.y)) / 2 * sample_count
            # current_loss = np.mean(np.square(predict_y - self.y))
            change_loss = np.abs(pre_loss - current_loss)
            num_iter += 1

    """
    MGD方式迭代
    predict_y : 预测值
    current_loss : 当前损失函数
    change_loss：损失函数变换量
    feature_count：特征属性数量
    sample_count：样本个数
    """
    def __sgd(self,predict_y,current_loss,change_loss,feature_count,sample_count):

        # 迭代次数
        num_iter = 0
        theta_iter = 0
        while change_loss >= self.tol and num_iter < self.max_iter:
            err = self.y - predict_y
            num_iter += 1
            for index in range(sample_count):
                theta_iter += 1
                x = self.X[index]
                predict_y = self.predict(x)
                delta = (self.y[index] - predict_y)
                # break
                for feature_index in range(feature_count):
                    self.coef_[feature_index] = self.coef_[feature_index] + self.alpha * delta * x[feature_index]
                    # 截距项
            self.intercept_ = self.intercept_ + self.alpha * np.mean(err)
            # 计算变化量
            pre_loss = current_loss
            predict_y = self._internel_predict(self.X, self.coef_, self.intercept_)
            # print(predict_y,self.y)
            current_loss = np.sum(np.square(predict_y - self.y)) / 2 * sample_count
            # current_loss = np.mean(np.square(predict_y - self.y))
            change_loss = np.abs(pre_loss - current_loss)

           # self.__coef_intercept_sgd(predict_y,feature_count,sample_count)

    def __mbgd(self,predict_y,current_loss,change_loss,feature_count,sample_count):
        num_iter = 0
        while change_loss >= self.tol and num_iter < self.max_iter:
            err = self.y - predict_y
            for index in range(sample_count // 10):
                if sample_count < 10:
                    sample_gradient = self.X
                    sample_gradient_err = err
                else:
                    sample_gradient = self.X[index:index+10]
                    sample_gradient_err = err[index:index+10]
                for feature_index in range(feature_count):
                    delta = 0
                    for sample_gradient_index in range(sample_gradient.shape[0]):
                        delta += sample_gradient_err[sample_gradient_index] * sample_gradient[sample_gradient_index][feature_index]
                    self.coef_[feature_index] = self.coef_[feature_index] + self.alpha * delta
                    # 截距项
            self.intercept_ = self.intercept_ + self.alpha * np.mean(err)
            pre_loss = current_loss
            predict_y = self.predict(self.X)
            current_loss = np.sum(np.square(predict_y - self.y)) / 2 * sample_count
            # current_loss = np.mean(np.square(predict_y - self.y))
            change_loss = np.abs(pre_loss - current_loss)
            num_iter +=1


    """
    模型参数和截距项计算bgd
    predict_y : 预测值
    y：预测值
    feature_count：特征属性数量
    sample_count：样本个数
    """
    def __coef_intercept_bgd(self,predict_y,feature_count,sample_count):
        err = self.y - predict_y
        # theta = theta - alpha * J的一阶导数
        # J的一阶导数又等于 实际值与预测值差值乘以当前样本特征属性求和
        for feature_index in range(feature_count):
            delta = 0
            # 求解变化的theta
            for sample_index in range(sample_count):
                delta += err[sample_index] * self.X[sample_index][feature_index]
            self.coef_[feature_index] = self.coef_[feature_index] + self.alpha * 1.0 / sample_count * delta
        # 截距项
        self.intercept_ = self.intercept_ + self.alpha * np.mean(err)

    """
    模型预测
    X: 样本
    y: 目标属性
    """
    def predict(self,X):
        return self._internel_predict(X,self.coef_,self.intercept_)

    """
    预测值
    X：预测样本
    coef：模型参数
    intercept：截距项
    """
    def _internel_predict(self,X,coef,intercept):
        # 将权重改为矩阵形式
        coef = np.reshape(coef,(-1,1))
        # 预测值就是X*coef
        predict_y = np.dot(X,coef) + intercept
        # 预测值修改为一维数组
        predict_y = np.reshape(predict_y,-1)
        return predict_y

    """
    评估模型效果
    X:样本
    y:样本真实的目标属性
    :return  r2
    """
    def score(self,X,y):
        # 将X和Y转换为NumPy数组的形式
        X = np.asarray(X)
        y = np.asarray(y)
        y = np.reshape(y, -1)
        predict_y = self.predict(X)
        rss = np.sum(np.square(predict_y - y))
        # 计算实际值均值
        y_average = np.average(y)
        # 实际值减去平均值
        tss = np.sum(np.square(y-y_average))
        r2 = 1 - rss/tss
        return r2






if __name__ == '__main__':
    np.random.seed(214)
    N = 10
    x = np.linspace(0,7,N) + np.random.randn(N)
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

    # algo = LinearRegression(fit_intercept=True)
    # algo.fit(x, y)
    # algo.score(x, y)
    # print("LinearRegression模型的评估指标R2:{}".format(algo.score(x, y)))
    # print("LinearRegression模型的模型参数:{}".format(algo.coef_))
    # print("LinearRegression模型的截距项:{}".format(algo.intercept_))

    algo = MickyLinearRegression(max_iter=1000,iterat_type='MBGD')
    algo.fit(x,y)
    print("Micky模型的评估指标R2:{}".format(algo.score(x, y)))
    print("Micky模型的模型参数:{}".format(algo.coef_))
    print("Micky模型的截距项:{}".format(algo.intercept_))




    sgd = SGDRegressor(max_iter=1000)
    sgd.fit(x, y)
    sgd_coef_ = sgd.coef_
    sgd_intercept_ = sgd.intercept_
    print("SGDRegressor模型的评估指标R2:{}".format(sgd.score(x, y)))
    print("SGDRegressor模型的模型参数:{}".format(sgd_coef_))
    print("SGDRegressor模型的截距项:{}".format(sgd_intercept_))
    plt.plot(x, y, 'ro')
    plt.plot(x, sgd.predict(x), 'g')
    plt.show()

