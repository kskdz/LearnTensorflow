# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 17:01:04 2020
    通过一个Keras来创建Estimator
@author: 67443
"""
import tensorflow as tf

import numpy as np
import tensorflow_datasets as tfds

'创建一个Keras模型'

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3)
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam')
model.summary()

'创建输入函数'
'''
Estimator 需要控制构建输入流水线的时间和方式。为此，它们需要一个“输入函数”或 input_fn。
Estimator 将不使用任何参数调用此函数。input_fn 必须返回 tf.data.Dataset。
'''
def input_fn():
  split = tfds.Split.TRAIN
  dataset = tfds.load('iris', split=split, as_supervised=True)
  dataset = dataset.map(lambda features, labels: ({'dense_input':features}, labels))
  dataset = dataset.batch(32).repeat()
  return dataset

for features_batch, labels_batch in input_fn().take(1):
  print(features_batch)
  print(labels_batch)
  
'构造Estimator'
import tempfile
model_dir = tempfile.mkdtemp()
keras_estimator = tf.keras.estimator.model_to_estimator(
    keras_model=model, model_dir=model_dir)

keras_estimator.train(input_fn=input_fn, steps=500)
eval_result = keras_estimator.evaluate(input_fn=input_fn, steps=10)
print('Eval result: {}'.format(eval_result))