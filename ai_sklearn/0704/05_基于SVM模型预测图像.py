#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/4 15:35
# @Author  : Micky
# @Site    : 
# @File    : 05_基于SVM模型预测图像.py
# @Software: PyCharm

import numpy as np
from sklearn.externals import joblib
from scipy.misc import imresize,imread

class ModelLoader(object):
    def __init__(self,model_file_path):
        # 模型恢复
        self.algo = joblib.load(model_file_path)

    def predict(self,x):
        return self.algo.predict(x)

if __name__ == '__main__':
    filename = './models/svm_digits.pk1'
    model = ModelLoader(filename)
    # 训练集中图像的数组最多16，而我们取得图像的数组最大255，所以这里我们需要转换下
    img = imread('./digits_1.png') / 16
    # 将图片转换为特征属性为64的数组
    img = np.reshape(img,(-1,64))
    # 图像预测值
    predict_y = model.predict(img)
    print('预测值：{}'.format(predict_y))