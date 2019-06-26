#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/26 16:11
# @Author  : Micky
# @Site    : 
# @File    : 01_压缩相关知识.py
# @Software: PyCharm

import numpy as np
from PIL import Image
from scipy import misc

if __name__ == '__main__':
    # 图像加载
    image = Image.open('../datas/xiaoren.png')
    # 图像转换为numpy数组
    img = np.asarray(image)
    print(img.shape)

    # 构建一个新的图像
    imageNew = np.zeros((600,100,3))
    imageNew = imageNew.astype(np.uint8)
    misc.imsave('m.png',imageNew)
