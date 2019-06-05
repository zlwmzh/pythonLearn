#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/5 14:48
# @Author  : Micky
# @Site    : 
# @File    : 02_KNN算法代码实现.py
# @Software: PyCharm

import numpy as np
from sklearn.neighbors import KDTree

class KNN(object):

    def __init__(self, k_neighbors = 5, with_kdtree = True, max_count_type = 'max_count'):
        # K值(最相似的几个值)
        self.k_neighbors = k_neighbors
        # 是否使用kdtree
        self.with_kdtree = with_kdtree
        self.max_count_type = max_count_type

    """
     训练数据
     train_X : 训练特征属性集
     train_Y : 训练目标属性集
    """
    def fit(self, train_X, train_Y):
        self.train_X = train_X
        self.train_Y = train_Y
        if self.with_kdtree:
            # 创建KDTree  这里的叶子节点和计算方式可以自定义，我们这里用默认的叶子节点，以欧式距离进行计算
            self.tree = KDTree(self.train_X, leaf_size=40, metric='euclidean')
            print('采用KDTree的实现方式')
        else:
            print('采用暴力查找的实现方式')

    """
    预测样本
    predict_x : 待预测样本集
    分为两步：
            1. 找出K个最相似的样本(距离上最近)
            2. 计算出K个最相似样本出现最多的目标属性即为预测值
    """
    def predict(self, predict_X):
        predict_label = []
        # print(np.argwhere(np.logical_and.reduce([0,0] == train_X, axis=1) == True))
        for x in predict_X:
            # 0. 如果预测样本点就是训练集中的点，直接返回目标值即可，不必执行接下来的操作
            index = np.argwhere(np.logical_and.reduce(x == train_X, axis=1) == True)
            if len(index) > 0:
                print("测试样本中包含，直接返回目标值")
                predict_label.append(train_Y[index[0]])
                continue
            # if x in train_X:
            #     continue
            # 1. 找出K个最相似的样本(距离上最近)
            k_neighbors_dist, k_neighbors_index = self.fetch_k_neighbors(x, return_distance = True)
            # 2. 计算出K个最相似样本出现最多的目标属性即为预测值
            predict_label.append(self.calc_max_count_label(k_neighbors_dist, k_neighbors_index, self.max_count_type))
        return predict_label

    """
    找出待测样本点的K个最近邻
    predict_x：待测样本点
    # 两种方式：暴力查找，KDTree查找
    :return 
    """
    def fetch_k_neighbors(self, predict_x, return_distance= True):
        if self.with_kdtree:
            # kdtree实现
           return self.query_by_kdtree(predict_x, return_distance= return_distance)
        else:
            # 暴力查找
            pass

    """
    KDTree查找最相似的K个样本
    tree.query：返回值为对应测试集合中最近的K个样本的下标，和对应的距离
    """
    def query_by_kdtree(self, predict_x, return_distance= True):
       if return_distance:
           dist_L, label_index_L = self.tree.query([predict_x],k = self.k_neighbors,return_distance = return_distance)
           return dist_L[0], label_index_L[0]
       else:
           label_index_L = self.tree.query([predict_x], k=self.k_neighbors, return_distance=return_distance)[0]
           return label_index_L


    """
    找出待测点的目标属性
    k_neighbors_dist：最近的K个样本点与待测样本点的距离
    k_neighbors_index：最近的K个样本点的下标
    max_count_type：计算方式  多数表决法(max_count)、加权多数表决发(max_count_add_weight)
    """
    def calc_max_count_label(self,k_neighbors_dist, k_neighbors_index, max_count_type = 'max_count'):
        if max_count_type == 'max_count':
            # 多数表决法
            return self.calc_max_count_label_by_max_count(train_Y[k_neighbors_index])
        else:
            # 加权多数表决法
            self.calc_max_count_label_by_max_count_weight(k_neighbors_dist, k_neighbors_index)

    """"
    通过多数表决法找出
    k_neighbors_y ： K个最相似的目标属性
    """
    def calc_max_count_label_by_max_count(self, k_neighbors_y):
        label_count_dict = {}
        for y in k_neighbors_y:
            if y in label_count_dict:
                label_count_dict[y] += 1
            else:
                label_count_dict[y] = 1
        max_count = 0
        max_count_label = -1
        for label, count in label_count_dict.items():
            if count > max_count:
                max_count = count
                max_count_label = label
        return max_count_label

    """
    加权多数表决法：权重取距离的相反数
    k_neighbors_dist：最近的K个样本点与待测样本点的距离
    k_neighbors_index：最近的K个样本点的下标
    """
    def calc_max_count_label_by_max_count_weight(self, k_neighbors_dist, k_neighbors_index):
        # 计算权重
        k_neighbors_weight = [1/dist  for dist in k_neighbors_dist if dist != 0]
        # k个最相似样本的目标属性
        k_neighbors_y = train_Y[k_neighbors_index]
        # 统计目标属性个数
        label_count_dict = {}
        for y in k_neighbors_y:
            if y in label_count_dict:
                label_count_dict[y] += 1
            else:
                label_count_dict[y] = 1
        # 计算权重
        k_neighbors_label_weight = {}
        for y,index in zip(k_neighbors_y,k_neighbors_index):
            print(label_count_dict[y] * index)
        # print(k_neighbors_label_weight)

if __name__ == '__main__':
    # 1. 数据加载
    train_X = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [5, 8],
        [3, 6],
        [1, 4],
        [4, 7],
        [6, 9],
        [1, 9]
    ])
    train_Y = np.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 0])
    # 2. 数据清洗、处理(不需要)
    # 3. 训练数据和测试数据划分(创建数据时候已经划分)
    # 4. 特征工程
    # 5. 创建模型对象
    algo = KNN(k_neighbors = 4,with_kdtree = True,max_count_type='max_count_add_weight')
    # 6. 训练模型
    algo.fit(train_X, train_Y)

    X = np.asarray([
        [1, 3],
        [4, 6],
        [7, 9],
        [6, 9]
    ])
    algo.predict(X)

