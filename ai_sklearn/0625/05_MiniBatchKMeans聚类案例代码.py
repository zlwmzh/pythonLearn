#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/25 20:14
# @Author  : Micky
# @Site    : 
# @File    : 04_KMeans聚类案例代码02.py
# @Software: PyCharm

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':
    N = 1000
    n_center = 4
    x,y = make_blobs(n_samples=N,n_features=2,centers=[(5, 5), (-5, 5), (5, -5), (-5, -5)],
                     cluster_std=1.0,center_box=(-10.10))

    """
    def __init__(self, n_clusters=8, init='k-means++', max_iter=100,
                 batch_size=100, verbose=0, compute_labels=True,
                 random_state=None, tol=0.0, max_no_improvement=10,
                 init_size=None, n_init=3, reassignment_ratio=0.01):
                 n_clusters：簇的数目
                 init：给定初始的簇中心点的初始化方式，默认kmeans++，可选random
                 max_iter：最大迭代次数
                 batch_size：批次大小，使用fit方法进行模型训练的时候，每个批次使用样本数目
                 tol：损失函数的收敛阈值
                 compute_labels：是否计算训练数据对应的预测簇标签label，如果要调用labels_这个属性，必须设置为True
    """
    algo = MiniBatchKMeans(n_clusters = n_center,compute_labels=True)
    algo.fit(x)

    x_test = [
        [-4, 8],
        [-5, 9],
        [1, 9],
        [4, 8],
        [-2, -5],
        [-8, 8]
    ]
    print('获取预测值：(预测属于那个簇，然后把簇下标返回)')
    print(algo.predict(x_test))
    print('簇中心坐标：{}'.format(algo.cluster_centers_))


    # MiniBatchKMeans算法是支持基于之前模型训练的结果，使用新的数据继续更新模型参数
    """
    fit：从零开始模型的训练
    partial_fit：基于之前训练的结果/模型参数，使用新给定的数据，继续进行模型参数的优化(直接在原来
    之前训练出来的中心点的基础上，做一次数据的划分和中心点的更新操作)
    """
    algo.partial_fit(x_test)
    print('簇中心点坐标：{}'.format(algo.cluster_centers_))

