#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/3 10:51
# @Author  : Micky
# @Site    : 算法原理步骤代码后面
# @File    : 01_KNN算法伪代码.py
# @Software: PyCharm

import numpy as np


class KNN:

    """
     train_x ：训练样本的特征属性
     train_y ：训练样本的目标属性
    """
    def fit(self,X,Y,k):
        # X此时是一个矩阵形式，有m行，n列, m行表示m个样本，n列表示每个样本有n个特征属性
        # Y此时是一个数组形式，中间有m个目标属性
        # k是一个数字
        self.train_x = X
        self.train_y = Y
        self.k = k

    """
    X：预测样本集
    """
    def predict(self,X):
        # 预测值放到列表中
        predict_labels = []
        # 遍历待预测样本
        for x in X:
            # 找出样本x的k个最相似的样本
            neighbors = self.fetch_k_neighbors(x)
            # 将这K个最相似样本中出现类别(目标属性)最多的最为样本的预测值
            predict_labels.append(self.calc_max_count_label(neighbors))
        return predict_labels

    """
    x：预测样本
    
    找出样本x的k个最相似样本
    Return：最相似的k个样本的目标属性
    """
    def fetch_k_neighbors(self,x):
        # 找k个最相似样本，我们需要计算待预测样本与整个测试集的距离，这里
        # 采用欧式距离计算.
        # 遍历测试集
        all_distant = []
        index = 0
        for tmp_x in self.train_x:
            # 计算欧式距离并添加到数组中。考虑到我们需要找到对应的目标属性
            # 每个数组存放对应的元祖
            distant = np.sqrt((tmp_x - x)**2)
            all_distant.append((distant,index))
            index += 1
        # 将x与测试样本集的距离进行排序
        all_distant = sorted(all_distant)
        # 取出距离最近的k个值
        top_k_index = list(map(lambda t:t[1],all_distant[:self.k]))
        # 获取k个值得目标属性
        neighbors = self.train_y[top_k_index]
        return neighbors

    """
    找到k个最相似样本出现类别最多的目标属性
    neighbors：k个最相似样本的目标属性
    Return：预测值/目标属性
    """
    def calc_max_count_label(self,neighbors):
        # 定义lable_count_dict = {} 统计每个lable出现的个数
        lable_count_dict = {}
        # 遍历neighbors
        for label in neighbors:
            if label not in lable_count_dict:
                # 如果之前没有添加，添加当前label，并设置个数为1
                lable_count_dict[label] = 1
            else:
                # 如果之前添加过，个数+1
                lable_count_dict[label] += 1
        # 获取出现次数最多的标签
        max_label = -1
        max_label_count = 0
        for label in lable_count_dict:
            # 当前目标属性的个数
            count = lable_count_dict[label]
            # 根据个数大小替换标签
            if count > max_label_count:
                max_label = label
                max_label_count = count
        return max_label

"""
    KNN算法原理：
       数据在空间上距离相近时，这些数据具有相似的特征属性。"近朱者赤，近墨者黑"
    分类预测：多数表决法，加权表决法
    回归预测：平均值法，加权平均值法           
    
    KNN算法三要素：K值得选择(K过小容易过拟合，K值过大容易欠拟合)[暴力查找、KDTree 查找k个样本 ]、距离的度量（一般都是欧式距离）
                   决策规则(多数表决法，加权表决法  平均值法，加权平均值法) 
                   
    KNN算法步骤：
    1. 找出待预测样本点最相似的K个样本
    2. 找出K个样本的目标属性中出现个数最大值(多数表决法)     
"""


