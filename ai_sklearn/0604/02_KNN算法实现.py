#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/4 16:20
# @Author  : Micky
# @Site    : 
# @File    : 02_KNN算法实现.py
# @Software: PyCharm

"""
  -1. KNN算法原理：
      认为在空间上相近的样本具有相同的特征属性。“近朱者赤，近墨者黑”

  -2. KNN算法步骤：
      a. 从训练数据中获取和待预测样本最相似的K个样本
      b. 将K个样本的目标属性合并产生最终的预测值(多数投票)

  -3. KNN算法缺陷：计算待测样本与训练样本的距离计算量大。使用KDTree进行优化
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class MickyKNNClassic(object):
    def __init__(self, k = 3):
        self.k = k

    """
     拟合数据
     train_X : 训练特征属性集
     train_Y : 训练目标属性值   
    """
    def fit(self, train_X, train_Y):
        self.train_X = train_X
        self.train_Y = train_Y

    """
    预测目标属性
    predict_X : 
    """
    def predict(self,predict_X):
        # 遍历预测样本
        for x in predict_X:
            # 找到x的k个最近样本的目标属性
            pass

    """
    查找待测样本点的K个最近点
    """
    def fetch_k_neighbors(self,predict_x):
        dist_label_list = []
        for x,y in zip(self.train_X, self.train_Y):
            # 计算欧式距离
            dist = np.sqrt(np.sum((x - predict_x) ** 2))
            dist_label_list.append((dist,y))
        print (sorted(dist_label_list))

if __name__ == '__main__':
    # 1. 数据加载
    iris = pd.read_csv(filepath_or_buffer='../datas/iris.data', header=None, names = ['c1','c2','c3','c4','y'])
    X = np.asarray(iris[['c1','c2','c3','c4']])
    Y = np.asarray(iris['y'])
    # 2. 数据清洗、处理
    label_name_index = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
    Y = list(map(lambda name:label_name_index[name],Y))
    # 3. 训练数据和测试数据划分
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=214)
    # 4. 特征工程
    # 5. 模型对象构建
    knnC = MickyKNNClassic()
    # 6. 模型训练
    knnC.fit(x_train,y_train)
    # print(x_train)
    knnC.fetch_k_neighbors(x_train)

