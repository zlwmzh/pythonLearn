#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/25 20:14
# @Author  : Micky
# @Site    : 
# @File    : 04_KMeans聚类案例代码02.py
# @Software: PyCharm

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':
    N = 1000
    n_center = 4
    x,y = make_blobs(n_samples=N,n_features=2,centers=[(5, 5), (-5, 5), (5, -5), (-5, -5)],
                     cluster_std=1.0,center_box=(-10.10))

    """
    def __init__(self, n_clusters=8, init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs=1, algorithm='auto'):
                n_clusters：簇的数目，也就是k
                init：给定初始的簇中心点的初始化方式
                max_iter：最大的迭代次数
                tol：损失函数收敛值 
    """
    km = KMeans(random_state=28)

    """
    这种选择最优参数的方法对k值的选择是没有效果的
    """
    # param_grid = {
    #     'n_clusters':[2,3,4,5,6,7,8,9,10,11,12]
    # }
    # algo = GridSearchCV(estimator=km,param_grid=param_grid,cv=3)
    # algo.fit(x)
    # print('最优参数：{}'.format(algo.best_params_))

    # 方式二：基于手肘法选择最优的K值
    inertias = []
    sss = []
    n_clusters = [2,3,4,5,6,7,8,9,10,11,12]
    for n_cluster in n_clusters:
        km = KMeans(random_state=28,n_clusters = n_cluster)
        km.fit(x)
        inertia_ = km.inertia_
        inertias.append(inertia_)
        ss = silhouette_score(x,km.labels_)
        sss.append(ss)

    # 做一个可视化
    plt.subplot(1, 3, 1)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=30)
    plt.subplot(1, 3, 2)
    plt.plot(n_clusters, inertias, 'ro--')
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(n_clusters, sss, 'ro--')
    plt.grid(True)
    plt.show()