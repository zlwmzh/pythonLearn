#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/18 22:26
# @Author  : Micky
# @Site    :
# @File    : 03_GBDT算法的过程理解代码.py
# @Software: PyCharm


import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# 设置随机数种子
np.random.seed(214)
if __name__ == '__main__':
   flag = 1
   if flag == 1:
       # 回归实现
       x = np.random.randn(10,2) * 5
       y = np.random.randn(10) * 3
       y_true = y
       # 使用单一的决策树模型拟合下数据
       algo = DecisionTreeRegressor(max_depth=1)
       algo.fit(x,y)
       print('单模型训练效果：{}'.format(r2_score(y_true,algo.predict(x))))
       print('实际y值：{}'.format(y_true))
       print('预测y值：{}'.format(algo.predict(x)))

       # GBDT回归代码构建过程
       # 存放每个子模型
       models = []
       # 构建第一个子模型,这里取均值，可以随意取值
       m1 = np.mean(y)
       # 添加到models中
       models.append(m1)
       # 学习步长
       learn_rate = 1.0
       # 保存当前的模型
       pred_m = m1
       # 总模型的数目(除第一个常熟模型外)
       n = 10
       for i in range(10):
           # 计算负梯度值，也就是更新y值
           # 因为第一项模型为常数项，区分开来
           if i == 0:
               y = y - learn_rate * pred_m
           else:
               # 计算当前模型的y的值
               y = y - pred_m.predict(x).reshape(y.shape)
           # print(y)
           # 构建当前子模型
           model = DecisionTreeRegressor(max_depth= 1)
           model.fit(x,y)
           models.append(model)
           pred_m = model
       print('模型构建完毕，总模型数目：{}'.format(len(models)))
       print('开始预测：')
       # zero_like：构建一个形状格式与y类似的numpy数组，但是填充值全部为0
       y_pred = np.zeros_like(y)
       # 因为总共有n+1个模型
       for i in  range(n+1):
           # 取出模型
           model = models[i]
           # 结果为所有模型预测结果之和
           if i == 0:
               y_pred = y_pred + learn_rate * model
           else:
               y_pred = y_pred + learn_rate * model.predict(x).reshape(y.shape)

       print('GBDT效果：{}'.format(r2_score(y_true,y_pred)))
       print('实际值：{}'.format(y_true))
       print('预测值：{}'.format(y_pred))