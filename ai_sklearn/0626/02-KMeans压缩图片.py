#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/26 16:26
# @Author  : Micky
# @Site    : 
# @File    : 02-KMeans压缩图片.py
# @Software: PyCharm

from PIL import Image
from scipy import misc
from sklearn.cluster import KMeans
import numpy as np

if __name__ == '__main__':
    # 加载图片
    old_img = Image.open('../datas/xiaoren.png')
    # 数据处理
    old_img_px_data = np.asarray(old_img)
    print(old_img_px_data.shape)

    # 因为图片转换为数组后是一个三维数组，长、宽、像素
    # 我们需要将这些信息当作样本来处理，所以需要将三维数组转化为二维数组
    w, h, px = old_img_px_data.shape[0], old_img_px_data.shape[1], old_img_px_data.shape[2]
    old_img_px_v = np.reshape(old_img_px_data,(w*h,px))
    km = KMeans(n_clusters=64)
    km.fit(old_img_px_v)
    print('簇中心坐标：{}'.format(km.cluster_centers_))