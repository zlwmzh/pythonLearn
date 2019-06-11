#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/11 9:24
# @Author  : Micky
# @Site    : 
# @File    : 02_基于决策树的鸢尾花数据分类.py
# @Software: PyCharm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    # 1. 数据加载
    df = pd.read_csv('../datas/iris.data',header = None, names = ['f1','f2', 'f3','f4','f5'])
    X = df[['f1','f2','f3','f4']]
    Y = df['f5']
    # 2. 数据清洗、处理
    """
    LabelEncoder：将字符串类型的数据转换成从0开始的序列
    fit_transform(Y)：找到模型映射关系后转换
    inverse_transform：反向转换，也就是进行数据恢复操作
    """
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)

    # 3. 训练数据、测试数据划分
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 214)
    # 4. 特征工程

    # 5. 模型对象创建
    """
    def __init__(self,
                 criterion="gini",  给定纯度的衡量指标，可选值gini和entropy
                 splitter="best", 跟定划分方式，选择最优划分还是随机划分，一般不改动
                 max_depth=None,  剪枝参数，指定决策树的最大深度，None表示不限制
                 min_samples_split=2, 剪枝参数，用于限制决策树划分的最低要求。对于某个节点进行划分时，要求该节点的样本数量至少为该值
                 min_samples_leaf=1, 剪枝参数，用于限制叶子节点的样本数目，要求划分之后叶子节点的样本数目必须大于等于该值
                 min_weight_fraction_leaf=0., 
                 max_features=None, 用于防止过拟合用的模型参数
                 random_state=None, 随机数种子
                 max_leaf_nodes=None,剪枝参数，最多允许的叶子节点数目，None表示不限制
                 min_impurity_decrease=0.,
                 min_impurity_split=None, 剪枝参数，如果一个节点要进行划分，要求他的gini系数或者是信息熵至少为该值，None表示不限制
                 class_weight=None,
                 presort=False):
    """
    algo = DecisionTreeClassifier(criterion='gini')
    # 6. 模型训练
    algo.fit(X_train, Y_train)
    # 7. 模型效果评估
    print('训练集上数据准确率：{}'.format(algo.score(X_train, Y_train)))
    print('测试集上数据准确率：{}'.format(algo.score(X_test, Y_test)))
    # 8. 看下属性API
    X_test = [
        [6.9, 3.1, 5.1, 2.3],
        [6.1, 2.8, 4.0, 1.3],
        [5.2, 3.4, 1.4, 0.2]
    ]
    print('样本预测值：')
    print(algo.predict(X_test))
    print('样本的预测概率：')
    print(algo.predict_proba(X_test))
    print('样本预测概率值得log转换值：')
    print(algo.predict_log_proba(X_test))
    print('各个特征属性的权重系数：{}'.format(algo.feature_importances_))