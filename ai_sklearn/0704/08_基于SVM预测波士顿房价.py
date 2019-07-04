#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/4 23:07
# @Author  : Micky
# @Site    : 
# @File    : 08_基于SVM预测波士顿房价.py
# @Software: PyCharm


import numpy as np
from sklearn.datasets import load_boston
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X, y = load_boston(return_X_y=True)
    print(np.shape(X))
    print(np.shape(y))

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=214)

    algo = SVR(kernel='rbf',C = 20,gamma=0.0004)
    algo.fit(X_train,y_train)

    print('训练集上的效果：{}'.format(algo.score(X_train,y_train)))
    print('测试集上的效果：{}'.format(algo.score(X_test, y_test)))
    predict_y_train = algo.predict(X_train)
    x_1 = range(len(X_train))
    x_2 = range(len(X_test))
    plt.plot(x_1,y_train,'r-')
    plt.plot(x_1,predict_y_train, 'g-')
    plt.show()
