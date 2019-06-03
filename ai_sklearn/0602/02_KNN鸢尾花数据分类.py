
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. 数据加载
iris  =  load_iris(return_X_y=True)
# print(iris)
X, Y = iris
# 2. 数据清洗、处理   此时数据没有必要做清洗和处理工作

# 3. 训练数据和测试数据的划分
# random_state 随机种子  如果不设置的话每次划分的数据不一样
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2,random_state=28)
# [0 2 1 0 2 1 2 1 1 0 2 0 1 1 2 0 2 2 2 1 0 0 1 1 1 0 2 2 0 1]
# [0 2 1 0 2 1 2 1 1 0 2 0 1 1 2 0 2 2 2 1 0 0 1 1 1 0 2 2 0 1]
print(y_test)
# 4. 特征工程

# 5. 模型对象构建
"""
def __init__(self, n_neighbors=5,
             weights='uniform', algorithm='auto', leaf_size=30,
             p=2, metric='minkowski', metric_params=None, n_jobs=1,
             **kwargs):
      参数：
       n_neighbors：邻居数目，也就是获取最近的样本的样本数目
       weights: 融合的方式，uniform 表示等权重的融合方式，distance 表示加权融合方式，权重系数为巨鹿的相反数
    （权重和距离成反比）
       algorithm ：求解方式，分为暴力计算和kdtree
       leaf_size：构建KDTree的时候，最多允许的叶子节点数目，
       p=2, metric='minkowski'：距离公式，可选值参考：https://scikit-learn.org/0.18/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric， 默认就是欧式距离。
"""
algo = KNeighborsClassifier(n_neighbors=5)
# 6. 模型训练
# fit : 进行模型训练
algo.fit(x_train,y_train)

# 7. 模型效果评估
# score:基于给定的数据获取评估指标，分类算法中为准确率，回归算法中为R2
print("训练数据上的准确率:{}".format(algo.score(x_train,y_train)))
print("测试数据上的准确率:{}".format(algo.score(x_test,y_test)))
# 8. 模型持久化
"""
持久化的方式主要三种：
-1. 将模型持久化为二进制的磁盘文件
-2. 将模型参数持久化到数据库中
-3. 使用模型对所哟数据进行预测，并将预测结果保存到数据库中
"""