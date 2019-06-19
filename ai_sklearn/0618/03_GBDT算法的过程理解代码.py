#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/18 22:26
# @Author  : Micky
# @Site    :
# @File    : 03_GBDT算法的过程理解代码.py
# @Software: PyCharm


import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score,accuracy_score

# 设置随机数种子
np.random.seed(214)
if __name__ == '__main__':
   flag = 3
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
   elif flag == 2:
       # 二分类的实现
       x = np.random.randn(10, 2) * 5
       # 前6个数为1，后四个数为0，两个分类0，1
       y = np.array([1] * 6 + [0] * 4).astype(np.int)
       y_true = y

       # 使用单个决策树进行分类
       algo = DecisionTreeClassifier(max_depth=1)
       algo.fit(x,y)
       print('单模型训练集上的效果：{}'.format(r2_score(y_true,algo.predict(x))))

       # GBDT二分类问题构建过程
       models = []
       # 构建第一个模型,一般以ln(正样本个数/负样本个数)作为初始值
       m1 = np.log(6/4)
       models.append(m1)
       # 学习率
       learn_rate = 0.1
       pred_m = m1
       n = 1000
       for i in range(n):
           if i == 0:
               y = y - learn_rate * pred_m
           else:
               y = y - learn_rate * pred_m.predict(x).reshape(y.shape)
           model = DecisionTreeRegressor(max_depth=1)
           model.fit(x,y)
           pred_m = model
           models.append(model)
       print('二分类模型构建完毕，总模型数目：{}'.format(len(models)))
       print('开始预测')
       y_pred = np.zeros_like(y)
       for i in range(n+1):
           model = models[i]
           if i == 0:
               y_pred = y_pred + learn_rate * model
           else:
               y_pred = y_pred + learn_rate * model.predict(x).reshape(y.shape)
       y_hat = np.zeros_like(y_pred,np.int)
       y_hat[y_pred>0.5] = 1
       y_hat[y_pred<0.5] = 0
       print("GBDT效果:{}".format(accuracy_score(y_true, y_hat)))
       print("实际值:{}".format(y_true))
       print("预测值:{}".format(y_hat))
       print("决策树函数值（属于正例的概率值）:{}".format(y_pred))
   else:
       # 多分类器的实现(GBDT对每一个分类创建一组决策树)
       x = np.random.randn(10,2) * 5
       y = np.array([0] * 2 + [1] * 4 +[2] * 4).astype(np.int)
       y_true = y

       # 针对每个类别构建一个y,属于当前类别设置为1，不属于设置为0
       y1 = np.array([1]*2 +[0] * 8)
       y2 = np.array([0]*2 + [1] *4 + [0]*4)
       y3 = np.asarray([0]*6 + [1] * 4)

       # 使用简单的决策树模型看下效果
       algo = DecisionTreeClassifier(max_depth=1)
       algo.fit(x,y)
       print("单模型训练数据集上效果:{}".format(accuracy_score(y_true, algo.predict(x))))
       print("实际y值:{}".format(y_true))
       print("预测y值:{}".format(algo.predict(x)))

       # GBDT分类算法构建过程
       models = []
       m1 = 0
       models.append(m1)
       learn_rate = 0.1
       pred_m = m1
       n = 100
       for i in  range(len(n)):
           pass



