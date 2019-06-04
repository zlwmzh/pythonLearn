#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/3 16:58
# @Author  : Micky
# @Site    : 
# @File    : 04_持久化模型应用模板.py
# @Software: PyCharm
from sklearn.externals import joblib

class ModelLoader(object):
    def __init__(self, model_file_name):
        # 1. 恢复模型
        self.algo = joblib.load(model_file_name)

    def predict(self,x):
        return self.algo.predict(x)

if __name__ == '__main__':
    # 1. 构建模型恢复预测的对象
    filename = './model/knn.pkl'
    model = ModelLoader(filename)

    # 2. 对数据进行一个预测
    X_test1 = [
        [5.1, 3.5, 1.4, 0.2],
        [5.7, 2.6, 3.5, 1.0]
    ]
    print(model.predict(X_test1))


