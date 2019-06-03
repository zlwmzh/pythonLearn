
import pandas as pd
file_path = '../datas/iris.data'
df = pd.read_csv(filepath_or_buffer=file_path, header = None, names = ['f1','f2','f3','f4','f5'])
df.info()

