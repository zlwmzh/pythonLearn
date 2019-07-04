#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/4 14:34
# @Author  : Micky
# @Site    : 
# @File    : 手写数据加载相关前置知识.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as  plt
from sklearn.datasets import load_digits
from scipy.misc import imsave,imread,imresize


if __name__ == '__main__':
    digits = load_digits()
    print(type(digits))
    print(digits.keys())
    # (1797,64) 1797个样本，64个特征属性
    print(np.shape(digits.data))
    # (1797,) 1797个目标属性
    print(np.shape(digits.target))
    # 目标属性的值
    print(digits.target_names)
    #(1797,8,8)  1797个图像，每个图像8*8大小
    print(np.shape(digits.images))

    print(digits.images[102])

    # 产生一个[0,1797)的随机数列
    random_index = np.random.permutation(np.shape(digits.data)[0])
    k = 1
    for idx in random_index[:25]:
        plt.subplot(5,5,k)
        # 黑白
        plt.imshow(digits.images[idx],cmap=plt.cm.gray_r)
        plt.title(digits.target[idx])

        # 把横纵坐标设置为空
        plt.xticks(())
        plt.yticks(())
        k += 1
    plt.show()

    # 保存图像
    idx = 1
    path = './digits_{}.png'.format(digits.target[idx])
    imsave(path,digits.images[idx])

    # 加载图像
    img = imread(path)
    print(img)
    plt.imshow(img,cmap=plt.cm.gray_r)
    plt.show()

    # 图像的resize操作
    img = imresize(img,(16,16))
    print(img)
    plt.imshow(img, cmap=plt.cm.gray_r)
    plt.show()


