#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/25 19:39
# @Author  : Micky
# @Site    : 
# @File    : 03_KMeans聚类案例代码.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans



class KMeansByMicky(object):

    """
        k_calssic：k个簇
        max_iter : 最大迭代次数
    """
    def __init__(self,k_calssic,max_iter = 10):
        self.k_calssic = k_calssic
        self.max_iter = max_iter
        self.centers = []

    def fit(self,X,y = None):
        self.X =np.asarray(X)
        # 随机产生K个簇中心点
        sample_count = self.X.shape[0]
        random_Index = list(range(sample_count))
        np.random.shuffle(random_Index)
        self.centers =self.X[random_Index[:self.k_calssic]]
        num_iter = 1
        # 开始迭代
        while num_iter <= self.max_iter:
            # 用来存储每个簇中样本的下标
            center_idx_sample_index = {}
            for sample in range(sample_count):
                current_sample = self.X[sample]
                # 初始化一个最小值，无限大
                min_dist = np.inf
                # 用来存储当前样本属于哪个簇中心的下标
                center_idx = 0
                # 计算当前样本点与K个簇的距离并得到最小距离的簇的下标
                for index_k in range(self.k_calssic):
                    dist = self.min_dist(self.calc_dist(current_sample,self.centers[index_k]))
                    if dist < min_dist:
                        min_dist = dist
                        center_idx = index_k
                if center_idx not in center_idx_sample_index:
                    center_idx_sample_index[center_idx] = [sample]
                else:
                    center_idx_sample_index[center_idx].append(sample)
            #print(center_idx_sample_index)
            # 更新簇中心点
            for k in range(self.k_calssic):
                center_sample = self.X[center_idx_sample_index[k]]
                #print(center_sample)
                # 切第二维的第一列数
                #print(np.mean(center_sample[:,0]))
                #print(np.mean(center_sample,axis=0))
                self.centers[k] = np.mean(center_sample,axis=0)
            num_iter += 1
        print(self.centers)


    """
    计算两个点的距离，欧式距离
    """
    def calc_dist(self,x1,x2):
        return np.sqrt(np.square(x1-x2))

    """
    获取最小值
    """
    def min_dist(self,dist):
        return np.min(dist)

if __name__ == '__main__':
    # 产生数据
    N = 1000
    n_centers = 4
    x,y = make_blobs(n_samples= N,n_features=2,centers=[(5,5),(-5,5),(5,-5),(-5,-5)],
                     cluster_std=2.0,center_box=(-10,10))

    # 模型构建
    algo = KMeans(n_clusters=4)
    algo.fit(x)

    # 3. 对测试数据做一个测试
    x_test = [
        [-4, 8],
        [-5, 9],
        [1, 9],
        [4, 8],
        [-2, -5],
        [-8, 8]
    ]
    print("获取预测值:（预测属于那个簇，然后把簇下标返回）")
    print(algo.predict(x_test))
    print("簇中心点坐标:{}".format(algo.cluster_centers_))
    # print('训练数据所属的簇下标：{}'.format(algo.labels_))
    # print('所有训练数据上的距离的平方和：{}'.format(algo.inertia_))
    # algo.score：回归中，该API返回R2；分类中，该API返回准确率；功能：在交叉验证、网格交叉验证等操作的时候，
    # 需要获取最优模型，默认情况下，在这些交叉验证中会基于score API返回最优模型，score的值越大，表示模型越好
    # print('当前模型的评估指标：{}'.format(algo.score(x,y)))

    print('获取每个样本到簇中心的距离作为特征属性构建的矩阵:')
    # print(algo.transform(x_test))


    m = KMeansByMicky(k_calssic=3)
    m.fit(x)

    plt.scatter(x[:,0],x[:,1],c = y)
    plt.show()
