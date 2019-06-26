#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/26 20:29
# @Author  : Micky
# @Site    : 
# @File    : 04_层次聚类(AGNES)算法采用不同距离计算公式策略导致大的数据合并不同形式.py
# @Software: PyCharm

from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

## 设置属性防止中文乱码及拦截异常信息
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
if __name__ == '__main__':
    n_clusters = 4
    # 模拟产生数据
    N = 1000
    x,y = make_blobs(n_samples=N,n_features=2,centers=[[-1,-1],[-1,1],[1,-1],[1,1]],cluster_std=0.5,random_state=214)


    # 产生月牙数据
    x2,y2 = make_moons(n_samples=N,noise=0.05)

    # n_noise = int(0.1 * N)
    # r = np.random.rand(n_noise, 2)
    # min1, min2 = np.min(x2, axis=0)
    # max1, max2 = np.max(x2, axis=0)
    # r[:, 0] = r[:, 0] * (max1 - min1) + min1
    # r[:, 1] = r[:, 1] * (max2 - min2) + min2
    # data2_noise = np.concatenate((x2, r), axis=0)
    # y2_noise = np.concatenate((y2, [3] * n_noise))

    plt.subplot(2,2,1)
    plt.title('原始高斯分布数据')
    plt.scatter(x[:,0],x[:,1],c = y)

    plt.subplot(2, 2, 3)
    plt.title('原始月牙数据')
    # plt.scatter(data2_noise[:, 0], data2_noise[:, 1], c=y2_noise)
    plt.scatter(x2[:, 0], x2[:, 1], c=y2)
    plt.show()


    # ac = AgglomerativeClustering(n_clusters=n_clusters,affinity='euclidean',connectivity=co)