#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/7 14:22
# @Author  : Micky
# @Site    : 
# @File    : 01_图片识别.py
# @Software: PyCharm


import os
import numpy as np
from scipy.misc import imresize,imread
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':

    # 数据加载
    absoluPath = '../datas/person_img/训练数据/'

    list_dir_path = []
    for path in os.listdir(absoluPath):
        filepath = absoluPath + path+"/"
        list_dir_path.append(filepath)


    x= []
    y= []
    y_target = ['刘诗诗','刘亦菲','杨幂','赵丽颖']
    target_index = 0
    for dir_path in list_dir_path:
        length = len(os.listdir(dir_path))
        count = 1
        for img_path in os.listdir(dir_path):
            # 读取图片
            full_path = dir_path + img_path
            img = imread(full_path)
            # print(np.shape(img))
            img = imresize(img,(16,16)).ravel()
            # 做一个归一化操作
            img = MinMaxScaler().fit_transform(img.reshape(-1,1)).ravel()
            # print(np.shape(img))
            x.append(img)
            if count <= length:
                # 添加目标属性
                y.append(y_target[target_index])
            # else:
            #     break
            count +=1
        target_index += 1
    print(np.shape(x),np.shape(y))

    absoluPathTest = '../datas/person_img/测试数据/'

    list_dir_path_test = []
    for path in os.listdir(absoluPathTest):
        filepath = absoluPathTest + path + "/"
        list_dir_path_test.append(filepath)

    x_test = []
    y_test = []
    y_target = ['刘亦菲', '杨幂']
    target_index = 0
    for dir_path in list_dir_path_test:
        length = len(os.listdir(dir_path))
        count = 1
        for img_path in os.listdir(dir_path):
            # 读取图片
            full_path = dir_path + img_path
            img = imread(full_path)
            # print(np.shape(img))
            img = imresize(img, (16, 16)).ravel()
            # 做一个归一化操作
            img = MinMaxScaler().fit_transform(img.reshape(-1, 1)).ravel()
            # print(np.shape(img))
            x_test.append(img)
            if count <= length:
                # 添加目标属性
                y_test.append(y_target[target_index])
            # else:
            #     break
            count += 1
        target_index += 1
    print(np.shape(x_test), np.shape(y_test))

    x = np.asarray(x)
    y = np.asarray(y)
    print('开始训练')
    algo = SVC(kernel='rbf',C = 20.0,gamma=0.01)
    algo.fit(x,y)

    print('训练集模型效果：{}'.format(algo.score(x,y)))
    print('测试集模型效果：{}'.format(algo.score(x_test, y_test)))
    print("训练集混淆矩阵：\n{}".format(confusion_matrix(y,algo.predict(x))))
    print("测试集混淆矩阵：\n{}".format(confusion_matrix(y_test, algo.predict(x_test))))
