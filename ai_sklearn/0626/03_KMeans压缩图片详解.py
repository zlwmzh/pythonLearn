#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/26 19:30
# @Author  : Micky
# @Site    : 
# @File    : 03_KMeans压缩图片详解.py
# @Software: PyCharm

"""
    图片分为彩色图片、黑白图片；彩色图片每个像素存放rgb三个色，黑白图片每个像素只存放单色。
    我们图片存储到存储设备上时是按照像素点存放不同的颜色。我们如果要压缩图片，可以将相似色值的像素点
    调整为统一色值，这样在存储的时候就会少的占用空间，也就实现了我们的压缩效果。
    图片的话我们可以看出一个个的像素点，这些像素点我们可以认为是我们的样本数据。我们对这些样本数据进行聚类操作，找到
    相似像素点的中心点，用这个中心点当作这些同簇样本的色值，就可以实现压缩功能
"""

import numpy as  np
from PIL import Image
from scipy import misc
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import shuffle

if __name__ == '__main__':
    # 读取图片
    old_img = Image.open('../datas/24.jpg')
    # 转换为numpy数组，为三维数组：w,h,px
    old_img_data = np.asarray(old_img)
    print(old_img_data.shape)
    # 数据清洗、处理
    # 我们需要将图片的信息转换为样本，样本应该是个二维的数组，我们需要
    # 对图片转换后的三维数组进行处理
    # 分别获取长、宽、像素点
    w,h,px = old_img_data.shape[0],old_img_data.shape[1],old_img_data.shape[2]
    # 构建样本数据数组,转换为2维数组的话，我们的样本数据的特征属性为像素，总样本个数为w*h
    old_img_sample = np.reshape(old_img_data,(w*h,px))

    old_img_sample_random = shuffle(old_img_sample,random_state = 28)[:10000]

    n_clusters = 32
    # 模型对象创建
    km = KMeans(n_clusters= n_clusters,random_state=214)
    km.fit(old_img_sample_random)
    labels = km.predict(old_img_sample)
    # mkm = MiniBatchKMeans(n_clusters= n_clusters,batch_size=10000,random_state=214)
    # mkm.fit(old_img_sample)


    # 创建一个新的原始的图片数组
    new_img = np.zeros((w,h,px))
    idx = 0
    for i in range(w):
        for j in  range(h):
            new_img[i][j] = km.cluster_centers_[labels[idx]]
            idx += 1
    misc.imsave('../datas/miniImg3.jpg',new_img)



