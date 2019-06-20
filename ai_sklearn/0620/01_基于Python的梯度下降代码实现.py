#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/20 19:48
# @Author  : Micky
# @Site    : 
# @File    : 01_基于Python的梯度下降代码实现.py
# @Software: PyCharm

import matplotlib as mpl
import matplotlib.pyplot as  plt
import numpy as np

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

"""
损失函数

"""
def j_function(bs,cs,theta):
    # 获取样本的个数
    sample_count = bs.shape[0]
    # 获取最终的损失函数
    j = 0
    for index in range(sample_count):
         b = bs[index]
         c = cs[index]
         j += (theta ** 2 - b * theta + c)/sample_count
    return j

"""
损失函数的导函数
"""
def derivative(bs,cs,theta):
    d = 0
    # 获取样本的个数
    if isinstance(bs,np.ndarray):
        sample_count = bs.shape[0]
        for index in range(sample_count):
            b = bs[index]
            d += (2 * theta - b) / sample_count
    else:
        d += (2 * theta - bs)
    return d

"""
求解最优的theta  BGD
b : 样本
c : 截距项
alpha ： 学习率/步长
tol：损失函数变化的插值最小值，小于这个值，则表明函数开始收敛了
max_iter：最大迭代次数
"""
def calc_min_value_v1_BGD(b,c,alpha,tol = 1e-32,max_iter = 100):
    # 当前b值均值：多个样本的时候，可知这个二次函数的最优解为均值一半
    print('当前b值均值：{}'.format(np.average(b)))
    # 给定一个初始值theta
    theta = 0
    # 当前损失函数的值
    current_loss = j_function(b,c,theta)
    # 损失函数变化值
    change_lose = abs(current_loss) + tol
    # 迭代次数
    num_iter = 0
    # 每次theta的变化情况
    delta_theta = 0
    # 存储可视化数据
    Loss_Function = []
    Thetas = []
    Delta_Thetas = []
    # 迭代更新theta
    while change_lose >= tol and num_iter <= max_iter:
        Loss_Function.append(current_loss)
        Thetas.append(theta)
        Delta_Thetas.append(delta_theta)
        # 沿梯度的方向迭代，损失函数会越来越大，我们希望损失函数原来越小
        # 所以我们让theta往损失函数的负梯度方向迭代，让损失函数更小
        delta_theta = alpha * derivative(b,c,theta)
        theta = theta - delta_theta
        # 记录当前损失函数
        pre_loss = current_loss
        # 更次更新后，计算下当前的函数值
        current_loss = j_function(b,c,theta)
        # 损失函数变化值
        change_lose = abs(current_loss - pre_loss)
        num_iter += 1
    print('最终迭代{}次后，模型参数theta：{}，损失函数值为：{}'.format(num_iter,theta,current_loss))
    # 可视化
    # 开启子图模式，总共三个子图
    plt.subplot(1,3,1)
    # 损失函数的变化情况
    plt.plot(range(num_iter),Loss_Function,'r-')
    plt.title('损失函数的变化情况')
    # theta的变化情况
    plt.subplot(1, 3, 2)
    plt.plot(range(num_iter),Thetas, 'g-')
    plt.title('Theta值的变化情况')

    # # 每次theta的变化大小情况
    # plt.subplot(1, 4, 3)
    # plt.plot(range(num_iter), Delta_Thetas, 'g-')
    # plt.title('每次theta的变化大小情况')

    # 构建原始函数图像看下梯度下降的效果
    X = np.arange(-3.5,3,0.5)
    Y = np.asarray(list(map(lambda t: j_function(b,c,t),X)))
    plt.subplot(1, 3, 3)
    plt.plot(X, Y, 'g-')
    # bo-- 蓝色--线
    plt.plot(Thetas,Loss_Function,'bo--')
    plt.title('每次theta的变化大小情况')
    plt.show()

"""
SGD
b : 样本
c : 截距项
alpha ： 学习率/步长
tol：损失函数变化的插值最小值，小于这个值，则表明函数开始收敛了
max_iter：最大迭代次数
"""
def calc_min_value_v1_SGB(bs,cs,alpha,tol = 1e-32,max_iter = 100):
    # 计算样本容量
    sample_count = bs.shape[0]
    theta = 5
    # 计算当前的损失函数
    current_loss = j_function(bs,cs,theta)
    change_loss = abs(current_loss) + tol

    # 迭代次数
    num_iter = 0
    theta_iter = 0
    delta_theta = 0
    # 存储可视化数据
    Loss_Function = []
    Thetas = []
    Delta_Thetas = []
    while change_loss >= tol and num_iter <= max_iter:
        Loss_Function.append(current_loss)
        # 遍历所有样本，每次使用一个样本，更新模型参数
        random_index = list(range(sample_count))
        np.random.shuffle(random_index)
        for index in random_index:
            theta_iter +=1
            Thetas.append(theta)
            Delta_Thetas.append(delta_theta)
            b = bs[index]
            c = cs[index]
            delta_theta = alpha * derivative(b,c,theta)
            theta = theta - delta_theta
        pre_loss = current_loss
        current_loss = j_function(bs,cs,theta)
        change_loss = abs(current_loss - pre_loss)
        num_iter += 1
    print('最终迭代{}次后，模型参数theta：{}，损失函数值为：{}'.format(num_iter, theta, current_loss))
 # 可视化
    # 开启子图模式，总共三个子图
    plt.subplot(1,3,1)
    # 损失函数的变化情况
    plt.plot(range(num_iter),Loss_Function,'r-')
    plt.title('损失函数的变化情况')
    # theta的变化情况
    plt.subplot(1, 3, 2)
    plt.plot(range(theta_iter),Thetas, 'g-')
    plt.title('Theta值的变化情况')

    # # 每次theta的变化大小情况
    # plt.subplot(1, 4, 3)
    # plt.plot(range(num_iter), Delta_Thetas, 'g-')
    # plt.title('每次theta的变化大小情况')

    # 构建原始函数图像看下梯度下降的效果
    # X = np.arange(-3.5,3,0.5)
    # Y = np.asarray(list(map(lambda t: j_function(b,c,t),X)))
    # plt.subplot(1, 3, 3)
    # plt.plot(X, Y, 'g-')
    # # bo-- 蓝色--线
    # plt.plot(Thetas,Loss_Function,'bo--')
    # plt.title('每次theta的变化大小情况')
    plt.show()

def calc_min_v1_MBGD(bs,cs,alpha,tol = 1e-32,max_iter = 100):
    sample_count = bs.shape[0]
    theta = 10
    current_loss = j_function(bs,cs,theta)
    change_loss = abs(current_loss) + tol

    num_iter = 0
    theta_iter = 0
    Thetas = []
    while change_loss >= tol and num_iter <= max_iter:
       # 将样本10个分一分，每10个
       for i in  range(sample_count // 10):
           sample_gradient = bs[i:i+10]
           delta_theta = alpha * derivative(sample_gradient,cs,theta)
           theta = theta - delta_theta
           Thetas.append(theta)
           theta_iter += 1
       pre_loss = current_loss
       current_loss = j_function(bs,cs,theta)
       change_loss = abs(current_loss - pre_loss)
       num_iter += 1
    print('最终迭代{}次后，模型参数theta：{}，损失函数值为：{}'.format(num_iter, theta, current_loss))
    # theta的变化情况
    plt.subplot(1, 3, 1)
    plt.plot(range(theta_iter), Thetas, 'g-')
    plt.title('Theta值的变化情况')
    plt.show()


if __name__ == '__main__':
    N = 100;
    b = np.random.normal(loc=-1.0,scale=1.0,size=N)
    c = np.random.normal(loc=5.0, scale=1.0, size=N)
    # calc_min_value_v1_BGD(b,c,alpha=0.1)
    # calc_min_value_v1_SGB(b,c,alpha=0.01,tol=1e-8)
    calc_min_v1_MBGD(b,c,alpha=0.8,tol=1e-8)
