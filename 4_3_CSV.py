# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 19:26:30 2020
    数据来自CSV文件
    使用泰坦尼克号的数据：模型会根据乘客的年龄、性别、票务舱和是否独自旅行等特征来预测乘客生还的可能性。

    这是一个经典的离散非数值进行深度学习预测的举例
@author: 67443
"""

import functools

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

'数据下载'
TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

# 让 numpy 数据更易读。
np.set_printoptions(precision=3, suppress=True)

'''
正如你看到的那样，CSV 文件的每列都会有一个列名。dataset 的构造函数会自动识别这些列名。
如果你使用的文件的第一行不包含列名，那么需要将列名通过字符串列表传给 make_csv_dataset 函数的 column_names 参数。

注意：对于包含模型需要预测的值的列是你需要显式指定的。
'''
CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']

LABEL_COLUMN = 'survived'
LABELS = [0, 1]

def get_dataset(file_path):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=12, # 为了示例更容易展示，手动设置较小的值
      label_name=LABEL_COLUMN,   #预测目标，也就是Label是必须指定的
      na_value="?",
      num_epochs=1,
      ignore_errors=True)
  return dataset

raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

examples, labels = next(iter(raw_train_data)) # 第一个批次
print("EXAMPLES: \n", examples, "\n")
print("LABELS: \n", labels)

'数据分类'
'''
使用 tf.feature_column API 创建一个 tf.feature_column.indicator_column 集合，
每个 tf.feature_column.indicator_column 对应一个分类的列。
本身数据是离散的，并非数值，但是我们需要输入神经网络是需要数值的
'''
#对于非数值化的项目，构建一个查询字典
'分类离散数据的处理'
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

#创建的内容
categorical_columns  

'连续数据:数据标准化'
'将离散的，非数值的数据转换为数值'
def process_continuous_data(mean, data):
  # 标准化数据
  data = tf.cast(data, tf.float32) * 1/(2*mean)
  return tf.reshape(data, [-1, 1])
'''
现在创建一个数值列的集合。tf.feature_columns.numeric_column API 会使用 normalizer_fn 参数。
在传参的时候使用 functools.partial，functools.partial 由使用每个列的均值进行标准化的函数构成。
'''
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

#创建的内容。
numerical_columns

'神经网络的创建'
#预处理层
#将这两个特征列的集合相加，并且传给 tf.keras.layers.DenseFeatures 从而创建一个进行预处理的输入层。
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numerical_columns)
#构建模型
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
#训练、评估和预测
train_data = raw_train_data.shuffle(500)
test_data = raw_test_data

model.fit(train_data, epochs=20)

#测试正确性
test_loss, test_accuracy = model.evaluate(test_data)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

#使用 tf.keras.Model.predict 推断一个批次或多个批次的标签。
predictions = model.predict(test_data)

# 显示部分结果
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
  print("Predicted survival: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("SURVIVED" if bool(survived) else "DIED"))
'''
对于离散数据和连续数据：
    对于属性数据的处理一般方案：tf.feature_column API
    对于分类数据：
        得到所有取值的字典：特征名和特征取值范围
        用tf.feature_column.categorical_column_with_vocabulary_list创建列
        然后合并即可categorical_columns.append(tf.feature_column.indicator_column(cat_col))
    对于连续数据：
        得到连续数据特征名和特征均值
        num_col = tf.feature_column.numeric_column(feature, normalizer_fn=functools.partial(process_continuous_data, MEANS[feature]))
        通过上面方式创建列，之后合并，numerical_columns.append(num_col)
    最后得到两个list表，再之后
    在神经网络第一层创建特征转化层
    preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numerical_columns)
'''


