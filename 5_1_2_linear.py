# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:44:42 2020
    创建一个线性模型:
        Build a linear model with Estimators
        This end-to-end walkthrough trains a logistic regression model using the tf.estimator API. 
        The model is often used as a baseline for other, more complex, algorithms.
@author: 67443
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

'Load the titanic dataset'
import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

#预览数据：
dftrain.head()
dftrain.describe()
dftrain.shape[0], dfeval.shape[0]

#图标反应数据特征：
dftrain.age.hist(bins=20)   #年龄特征
dftrain.sex.value_counts().plot(kind='barh') #性别比例
dftrain['class'].value_counts().plot(kind='barh') #主要乘客都是第三类乘客
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')  #女性生存几率更高

#构建模型特征工程

#基本要素列
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']
'''
两类特征：tf.feature_column  特征名，特征类型（分类可选集合）
    数值特征列    tf.feature_column.categorical_column_with_vocabulary_list
    分类特征列    tf.feature_column.numeric_column
'''
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

#设定输入函数
#通过tf.data.Dataset构建输入流
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function
#构建输入输出流
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

#查看数据集
ds = make_input_fn(dftrain, y_train, batch_size=10)()
for feature_batch, label_batch in ds.take(1):
  print('Some feature keys:', list(feature_batch.keys()))
  print()
  print('A batch of class:', feature_batch['class'].numpy())
  print()
  print('A batch of Labels:', label_batch.numpy())

#查看特定功能列的结果 tf.keras.layers.DenseFeatures图层检查特定功能列的结果：
age_column = feature_columns[7]
tf.keras.layers.DenseFeatures([age_column])(feature_batch).numpy()

#DenseFeatures仅接受密集张量，要检查分类列，您需要先将其转换为指标列：
gender_column = feature_columns[0]
tf.keras.layers.DenseFeatures([tf.feature_column.indicator_column(gender_column)])(feature_batch).numpy()

#在将所有基本特征添加到模型后，让我们训练模型。 训练模型只是使用tf.estimator API的单个命令：
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(result)

#派生特征列
#组合多个特征列进行预测
age_x_gender = tf.feature_column.crossed_column(['age', 'sex'], hash_bucket_size=100)

derived_feature_columns = [age_x_gender]
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns+derived_feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(result)

#正确率进一步提升
pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')

#查看ROC曲线     更好地了解真阳性率和假阳性率之间的折衷
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

fpr, tpr, _ = roc_curve(y_eval, probs)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0,)
plt.ylim(0,)



