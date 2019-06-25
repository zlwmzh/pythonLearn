#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/25 21:21
# @Author  : Micky
# @Site    : 
# @File    : 06_K-Means算法和MiniBatchK-Means算法比较.py
# @Software: PyCharm

import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    N = 100000
    # 三个簇中心点
    centers = [[1,1],[-1,-1],[1,-1]]
    # 簇为3个
    clusters = len(centers)
    x,y = make_blobs(n_samples=N,n_features=2,centers = centers,cluster_std=0.5,random_state=214)

    # KMeans算法
    km = KMeans(init='k-means++',n_clusters=clusters,random_state=214)
    t0 = time.time()
    km.fit(x)
    # 训练模型花费的时间
    km_batch = time.time() - t0
    print('K-Means算法模型训练消耗时间：%.4fs'%km_batch)

    # Mini-Batch-KMeans
    batch_size = 100
    mKm = MiniBatchKMeans(init='k-means++',n_clusters=clusters,batch_size=batch_size,random_state=214)
    t0 = time.time()
    mKm.fit(x)
    mKm_batch = time.time() - t0
    print("Mini Batch K-Means算法模型训练消耗时间:%.4fs" %  mKm_batch)

    # 预测结果
    print('K-Means：',end='')
    km_predict = km.predict(x)
    print(km_predict[:10])
    print('K-Means簇中心点坐标：{}'.format(km.cluster_centers_))
    print('Mini Batch K-Means：', end='')
    mkm_predict = mKm.predict(x)
    print(mkm_predict[:10])
    print('Mini Batch K-Means簇中心点坐标：{}'.format(mKm.cluster_centers_))

    k_means_centers = km.cluster_centers_
    min_centers = mKm.cluster_centers_

    order = pairwise_distances_argmin(k_means_centers,min_centers)
    print('order：{}'.format(order))

    cm = mpl.colors.ListedColormap(['#FFC2CC', '#C2FFCC', '#CCC2FF'])
    cm2 = mpl.colors.ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # 原数据图像
    plt.subplot(221)
    plt.title('原始分布')
    plt.scatter(x[:, 0], x[:, 1], c=y,cmap=cm)
    plt.grid(True)

    # K-Means算法聚类结果图
    plt.subplot(222)
    plt.title('K-Means图像分布')
    plt.scatter(x[:, 0], x[:, 1], c=km_predict, cmap=cm)
    plt.scatter(k_means_centers[:,0],k_means_centers[:,1],c=range(clusters),cmap=cm2)

    # K-Means算法聚类结果图
    plt.subplot(223)
    plt.title('Mini Batch K-Means分布')
    plt.scatter(x[:, 0], x[:, 1], c=mkm_predict, cmap=cm)
    plt.scatter(min_centers[:, 0], min_centers[:, 1], c=range(clusters), cmap=cm2)
    plt.show()