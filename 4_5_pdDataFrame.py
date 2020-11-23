# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 16:01:30 2020
    从pandas的DataFrame中创建一个模型
    
    本教程使用了一个小型数据集，由克利夫兰诊所心脏病基金会（Cleveland Clinic Foundation for Heart Disease）提供. 此数据集中有几百行CSV。
    每行表示一个患者，每列表示一个属性（describe）。我们将使用这些信息来预测患者是否患有心脏病，这是一个二分类问题。
    
@author: 67443
"""
import pandas as pd
import tensorflow as tf

csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/applied-dl/heart.csv')

df = pd.read_csv(csv_file)
df.head()
df.dtypes

'非数值类转化为离散数值'
df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes
df.head()

'转化为tf.data.Dataset'
'''
使用 tf.data.Dataset.from_tensor_slices 从 pandas dataframe 中读取数值。
使用 tf.data.Dataset 的其中一个优势是可以允许您写一些简单而又高效的数据管道（data pipelines)。
从 loading data guide 可以了解更多。
https://tensorflow.google.cn/guide/data?hl=zh-CN
'''
target = df.pop('target')
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))


for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))

'由于 pd.Series 实现了 __array__ 协议，因此几乎可以在任何使用 np.array 或 tf.Tensor 的地方透明地使用它。'
tf.constant(df['thal'])

train_dataset = dataset.shuffle(len(df)).batch(1)


'创建并训练模型'
def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  return model

model = get_compiled_model()
model.fit(train_dataset, epochs=15)


'代替特征列的方法处理'
'将字典作为输入传输给模型就像创建 tf.keras.layers.Input 层的匹配字典一样简单，应用任何预处理并使用 functional api。 您可以使用它作为 feature columns 的替代方法。'
'字典输入的经典例子'
inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
x = tf.stack(list(inputs.values()), axis=-1)

x = tf.keras.layers.Dense(10, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model_func = tf.keras.Model(inputs=inputs, outputs=output)

model_func.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

'与 tf.data 一起使用时，保存 pd.DataFrame 列结构的最简单方法是将 pd.DataFrame 转换为 dict ，并对该字典进行切片。'
dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), target.values)).batch(16)

for dict_slice in dict_slices.take(1):
  print (dict_slice)

model_func.fit(dict_slices, epochs=15)


