from sklearn.externals import joblib


class ModelLoader(object):
    def __init__(self, model_file_path):
        # 1 恢复模型
        self.algo = joblib.load(model_file_path)

    def predict(self, x):
        return self.algo.predict(x)


if __name__ == '__main__':
    # 1 构建模型恢复预测对象
    filename = './models/knn.pk1'
    model = ModelLoader(filename)

    # ctrl + alt + L  自动规整代码
    # 2 对数据进行预测
    x_test1 = [
        [5.7,4.4,1.5,0.4],
        [5.0,3.4,1.6,0.4]
    ]
    print(model.predict(x_test1))
