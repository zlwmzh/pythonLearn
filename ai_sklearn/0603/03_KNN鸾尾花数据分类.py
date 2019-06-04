#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/3 16:26
# @Author  : Micky
# @Site    : 
# @File    : 03_KNN鸾尾花数据分类.py
# @Software: PyCharm
import  os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.externals import joblib

# 1. 数据加载
X,Y = load_iris(return_X_y=True)

# 2. 数据清洗、处理(此数据不需要此项操作)

# 3. 训练数据集和测试数据集划分
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=214)

# 4. 特征工程

# 5. 模型对象构建
algo = KNeighborsClassifier(n_neighbors=5)

# 6. 模型训练
algo.fit(x_train,y_train)

# 7. 模型效果评估
print('训练数据集的准确率:{}'.format(algo.score(x_train,y_train)))
print('测试数据集的准确率:{}'.format(algo.score(x_test,y_test)))

# 8. 模型持久化
filename = './model/knn.pkl'
dirPath = os.path.dirname(filename)
if not os.path.exists(dirPath):
    os.makedirs(dirPath)
joblib.dump(value=algo,filename = filename)
