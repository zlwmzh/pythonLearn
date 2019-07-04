#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/4 16:04
# @Author  : Micky
# @Site    : 
# @File    : 06_基于SVM的中文字符识别.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import os
from scipy.misc import imresize,imread
from sklearn.svm import SVC

if __name__ == '__main__':
    # 加载数据
    dirpath_train = '../datas/训练数据/'
    dirpath_test = '../datas/验证数据/'
    list_path_train = []
    list_path_test = []
    for filename,filename2 in zip(os.listdir(dirpath_train),os.listdir(dirpath_test)):
        path_train = dirpath_train +filename+"/"
        path_test = dirpath_test +filename2+"/"
        list_path_train.append(path_train)
        list_path_test.append(path_test)
    # print(list_path_train)
    # print(list_path_test)
    x_train  = []
    y_train = []
    x_test = []
    y_test = []
    y_target = ['一','丁','七','万','丈','三','上','下','不','与']
    tartget_index = 0
    for train_path in list_path_train:
        length = len(os.listdir(train_path))
        count = 1
        for filename in os.listdir(train_path):
            path = train_path + filename
            img = imread(path)
            img = imresize(img,(16,16))
            # w,h,px = img.shape
            # img = np.reshape(img,(w * h, px))
            # print(img)
            # 遍历所有样本，每次使用一个样本，更新模型参数

            img = img.ravel()
            img = img[np.where(img < 230)]

            # img = img.ravel()

            # print(img)
            if(len(img)>100):
                # print(np.shape(img))
                x_train.append(img[:100])
                # 添加目标属性
                if count <= length:
                    # print('目标属性:{}'.format(y_target[tartget_index]))
                    y_train.append(y_target[tartget_index])
            else:
                print('不满足')

            count += 1
        tartget_index += 1
            # print(np.shape(img))
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    print(np.shape(x_train))
    print(np.shape(y_train))
    # print(x_train)
    # 测试集
    tartget_index = 0
    for test_path in list_path_test:
        length = len(os.listdir(test_path))
        count = 1
        for filename in os.listdir(test_path):
            path = test_path + filename
            img = imread(path)
            img = imresize(img,(16,16))
            img = img.ravel()

            img = img.ravel()
            img = img[np.where(img < 230)]

            # img = img.ravel()

            # print(img)
            if (len(img) > 100):
                print(np.shape(img))
                x_test.append(img[:100])
                # 添加目标属性
                if count <= length:
                    # print('目标属性:{}'.format(y_target[tartget_index]))
                    y_test.append(y_target[tartget_index])
            else:
                print('不满足')

            count += 1
        tartget_index += 1


    # y_test = np.asarray(y_test)
    # x_test = np.asarray(x_test)
    #
    # print(np.shape(x_test))
    # print(np.shape(y_test))

    algo = SVC(kernel='rbf',C=1,gamma=0.001)

    algo.fit(x_train,y_train)
    # # print(x_train)
    #
    # ipath = '../datas/验证数据/00000/72.png'
    # img = imread(ipath)
    # img = imresize(img,(16,16))
    # img = img.ravel()
    # img = img[img < 150]
    # print(img)
    # print(algo.predict([img.ravel()]))
    # print([x_test[0]])
    print('训练集上的预测值：{}'.format(algo.predict(x_train)))
    print('训练集模型效果：{}'.format(algo.score(x_train,y_train)))
    print('验证集上的预测值：{}'.format(algo.predict(x_test)))
    print('验证集上的预测值：{}'.format(algo.score(x_test,y_test)))

    # for idx in range(np.shape(x_train)[0]):
    #     if idx < 238:
    #         print('一')
    #     elif 238 <= idx < 474:
    #         print('丁')
    #     elif 474 <= idx < 714:
    #         print('七')
    #     elif 714 <= idx < 952:
    #         print('万')
    #     elif 952