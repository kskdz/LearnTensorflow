# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 11:28:11 2020
    数据加载总结：
        加载和预处理数据  tf.data.Dataset
        数据：泰坦尼克号 生还预测
        模型会根据乘客的年龄、性别、票务舱和是否独自旅行等特征来预测乘客生还的可能性。
@author: kdz-pc
"""
import functools

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

file = "C:\\Users\\kdz-pc\\mygit\\tensorflow2.0\\dataset\\income.csv"
#这个是 下载之后 得到数据所在的地址
TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

'''
precision: 保留几位小数，后面不会补0
supress: 对很大/小的数不使用科学计数法 (true)
formatter: 强制格式化，后面会补0
import numpy as np
a = np.random.random(3)
print('before set precision: \n',a)

np.set_printoptions(precision=3, suppress=True)
print('after set precision: \n',a)

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print('after set formatter: \n',a)

'''
np.set_printoptions(precision=3, suppress=True)  #格式化打印：设置输出数字精确度

CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']
LABEL_COLUMN = 'survived'
LABELS = [0, 1]

def get_dataset(file_path):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=12, # 为了示例更容易展示，手动设置较小的值
      column_names=CSV_COLUMNS,         #自定义读取列
      #select_columns = columns_to_use,    指定选择的列
      label_name= LABEL_COLUMN,         #指定作为标签的列
      na_value="?",
      num_epochs=1,
      ignore_errors=True)
  return dataset

raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

'''
dataset 中的每个条目都是一个批次，用一个元组（多个样本，多个标签）表示。
样本中的数据组织形式是以列为主的张量（而不是以行为主的张量），每条数据中包含的元素个数就是批次大小（这个示例中是 12）。
'''
examples, labels = next(iter(raw_train_data)) # 第一个批次
print("EXAMPLES: \n", examples, "\n")
print("LABELS: \n", labels)

'''
EXAMPLES: 
 OrderedDict([('sex', <tf.Tensor: shape=(12,), dtype=string, numpy=
array([b'male', b'male', b'female', b'female', b'male', b'male', b'male',
       b'male', b'female', b'female', b'male', b'male'], dtype=object)>), ('age', <tf.Tensor: shape=(12,), dtype=float32, numpy=
array([33., 28., 38., 28., 28., 28., 52., 45., 30., 24., 28., 17.],
      dtype=float32)>), ('n_siblings_spouses', <tf.Tensor: shape=(12,), dtype=int32, numpy=array([1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1], dtype=int32)>), ('parch', <tf.Tensor: shape=(12,), dtype=int32, numpy=array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1], dtype=int32)>), ('fare', <tf.Tensor: shape=(12,), dtype=float32, numpy=
array([20.525,  7.25 , 80.   , 15.5  ,  7.896, 56.496, 30.5  , 26.55 ,
       24.15 , 26.   ,  7.896,  7.229], dtype=float32)>), ('class', <tf.Tensor: shape=(12,), dtype=string, numpy=
array([b'Third', b'Third', b'First', b'Third', b'Third', b'Third',
       b'First', b'First', b'Third', b'Second', b'Third', b'Third'],
      dtype=object)>), ('deck', <tf.Tensor: shape=(12,), dtype=string, numpy=
array([b'unknown', b'unknown', b'B', b'unknown', b'unknown', b'unknown',
       b'C', b'B', b'unknown', b'unknown', b'unknown', b'unknown'],
      dtype=object)>), ('embark_town', <tf.Tensor: shape=(12,), dtype=string, numpy=
array([b'Southampton', b'Southampton', b'unknown', b'Queenstown',
       b'Southampton', b'Southampton', b'Southampton', b'Southampton',
       b'Southampton', b'Southampton', b'Cherbourg', b'Cherbourg'],
      dtype=object)>), ('alone', <tf.Tensor: shape=(12,), dtype=string, numpy=
array([b'n', b'y', b'y', b'n', b'y', b'y', b'y', b'y', b'n', b'n', b'y',
       b'n'], dtype=object)>)]) 

LABELS: 
 tf.Tensor([0 0 1 1 0 1 1 0 0 1 0 0], shape=(12,), dtype=int32)
'''

'数据预处理'
'CSV 数据中的有些列是分类的列。也就是说，这些列只能在有限的集合中取值'
'使用 tf.feature_column API 创建一个 tf.feature_column.indicator_column 集合，每个 tf.feature_column.indicator_column 对应一个分类的列'
CATEGORIES = {
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']
}

categorical_columns = []
for feature, vocab in CATEGORIES.items():
  cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
  categorical_columns.append(tf.feature_column.indicator_column(cat_col))

# 你刚才创建的内容
categorical_columns

'1\数据标准化'
#连续数据需要标准化,先将连续数据标准化然后转化为2维张量
def process_continuous_data(mean, data):
  # 标准化数据
  data = tf.cast(data, tf.float32) * 1/(2*mean)
  return tf.reshape(data, [-1, 1])

MEANS = {
    'age' : 29.631308,
    'n_siblings_spouses' : 0.545455,
    'parch' : 0.379585,
    'fare' : 34.385399
}

numerical_columns = []

for feature in MEANS.keys():
  num_col = tf.feature_column.numeric_column(feature, normalizer_fn=functools.partial(process_continuous_data, MEANS[feature]))
  numerical_columns.append(num_col)
'现在创建一个数值列的集合。tf.feature_columns.numeric_column API 会使用 normalizer_fn 参数。在传参的时候使用 functools.partial，functools.partial 由使用每个列的均值进行标准化的函数构成。'

# 你刚才创建的内容。
numerical_columns

'神经网络 构建'
#预处理层
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numerical_columns)

model = tf.keras.Sequential([
  preprocessing_layer,
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

train_data = raw_train_data.shuffle(500)
test_data = raw_test_data

model.fit(train_data, epochs=20)

test_loss, test_accuracy = model.evaluate(test_data)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

predictions = model.predict(test_data)

# 显示部分结果
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
  print("Predicted survival: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("SURVIVED" if bool(survived) else "DIED"))





