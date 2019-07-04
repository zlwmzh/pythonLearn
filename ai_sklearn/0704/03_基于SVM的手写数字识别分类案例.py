#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/4 14:15
# @Author  : Micky
# @Site    : 
# @File    : 03_基于SVM的手写数字识别分类案例.py
# @Software: PyCharm

import os
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.externals import joblib

if __name__ == '__main__':
    # 数据加载
    digits = load_digits()

    # 测试集训练集划分
    x_train,x_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.2,random_state=214)

    print(x_train[1])
    # 模型对象构建
    algo = SVC(kernel='rbf',C = 1.0,gamma=0.001)
    algo.fit(x_train,y_train)

    # 模型结果预测
    predict_train = algo.predict(x_train)
    predict_test = algo.predict(x_test)
    # 混淆矩阵看下训练集预测值和测试集预测值
    print('训练集上的混淆矩阵：\n{}'.format(confusion_matrix(y_train,predict_train)))
    print('测试集上的混淆矩阵：\n{}'.format(confusion_matrix(y_test,predict_test)))
    print('训练集上的分类报告：\n{}'.format(classification_report(y_train,predict_train)))
    print('测试集上的分类报告：\n{}'.format(classification_report(y_test,predict_test)))
    print('训练集模型准确率：{}'.format(accuracy_score(y_train,predict_train)))
    print('测试集模型准确率：{}'.format(accuracy_score(y_test, predict_test)))

    # 模型持久化
    filename = './models/svm_digits.pk1'
    dirpath = os.path.dirname(filename)
    if not os.path.exists(dirpath):
        print("文件夹不存在，将创建")
        os.makedirs(dirpath)
    joblib.dump(algo,filename)