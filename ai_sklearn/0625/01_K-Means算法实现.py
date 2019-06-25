#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/25 9:51
# @Author  : Micky
# @Site    : 
# @File    : 01_K-Means算法实现.py
# @Software: PyCharm

"""
 对于给定的类别/簇数目K，首先给定初始簇中心，通过迭代改变样本和簇的归属关系，使得每次处理
 的划分方式比上次的好（总的数据集之间的距离变小了） 无监督算法

 执行过程：
 输入：训练数据X，簇中心数目K
 输出：K个簇中心点的坐标/向量
 a. 随机产生K个样本做为簇中心点
 b. 计算当前样本到K个簇中心点的距离/相似度，选择距离最近或者是相似度最高的簇做为当前样本的
 隶属簇，也就是当前样本距离最近的那个簇（划分）
 c. 按照上述b步骤，对所有样本进行计算，划分对应的簇隶属关系。（所有样本归属那个簇应该是确定的，
 每个样本都有对应的簇划分）
 d. 基于b、c的划分，更新K个簇中心点，新的簇中心点为当前簇所有样本的均值
 e. 重复执行上述b、c、d，直到达到某个收敛条件，结束迭代
 f. 输出最终的簇中心点

"""

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

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
            print(self.centers)
            num_iter += 1


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
    """
    产生服从高斯分布的数据，返回x，y  x：样本  y：类别
    def make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0,
               center_box=(-10.0, 10.0), shuffle=True, random_state=None):
               n_samples：数据的数量，可以是每个类型相同数量，也可以以列表的形式定义每个类别不同的数目
               n_feature：给定每个样本的特征属性数目
               centers：给定中心点的数目或中心点的坐标
               cluster_std：给定每个类别中数据服从的高斯分布的标准差 
               center_box：给定数据的取值范围      
    """

    # 第一种：仅定义100个样本，两个特征属性
    # x,y = make_blobs(n_samples=100,
    #                  n_features=2
    #                  )

    # 第二种：定义100个样本，两个特征属性,簇中心数目为3，每个类别的标准差为0.5,取值范围[-10,10]
    x, y = make_blobs(n_samples=100,
                      n_features=2,
                      centers=3,
                      cluster_std=0.5,
                      center_box=[-10,10]
                      )


    # x, y = make_blobs(n_samples=100,
    #                   n_features=2,
    #                   centers=3,
    #                   cluster_std=0.5,
    #                   center_box=[-100, 100]
    #                   )

    # x, y = make_blobs(n_samples=100,
    #                   n_features=2,
    #                   centers=3,
    #                   cluster_std=[1.0,0.5,0.5],
    #                   center_box=(-10, 10)
    #                   )

    # 设置特征属性的标准差，2，3分类的两个特征属性的标准差不一样
    # x, y = make_blobs(n_samples=100,
    #                   n_features=2,
    #                   centers=3,
    #                   cluster_std=[1.0, [2.0,0.5], [1.0,5.0]],
    #                   center_box=(-10, 10)
    #                   )

    # 设置k个簇中心点坐标
    # x, y = make_blobs(n_samples=100,
    #                   n_features=2,
    #                   centers=[(0,0),[50,50],(-50,50)],
    #                   cluster_std=[1.0, 0.5, 0.5],
    #                   center_box=(-100, 100)
    #                   )

    algo = KMeansByMicky(k_calssic=3)
    algo.fit(x)

    # a = [[1,2],[3,4],[5,6]]
    # a = np.mean(np.asarray(a))
    # print(a)
    # 可视化操作
    plt.subplot(1,2,1)
    plt.scatter(x[:,0],x[:,1],c=y)
    plt.show()

