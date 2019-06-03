import  numpy as np
import pickle

# arr = np.array([
#     [1,3,5,7],
#     [2,4,6,8],
#     [1,5,9,8],
#     [2,5,8,9]
# ])

# # 将arr矩阵使用numpy的API保存
# np.save('./arr.npy',arr)

# 使用numpy的API加载npy的文件数据

# arr = np.load('./arr.npy')
# print(arr)

class Person(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age


xiaoming = Person('小明',15)
print((xiaoming,xiaoming.name,xiaoming.age))

# 任意对象持久化为磁盘文件
pickle.dump(xiaoming,open('./xiaoming.pkl','wb'))

#
xiaoming2 = pickle.load(open('./xiaoming.pkl','rb'))
print((xiaoming2,xiaoming2.name,xiaoming2.age))