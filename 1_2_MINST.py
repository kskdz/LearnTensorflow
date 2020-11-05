# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 17:39:16 2020
    MINST数据集合尝试：
        我的MINST数据集合保存位置：
        D:\BaiduNetdiskDownload\mygit\tensorflow2.0\dataset\MINST
        MINST
@author: kdz-pc
"""
import tensorflow as tf

#导入minst数据集合
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0       #数据归一化

'''
#如果数据无法下载；
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
#指定下载位置
'''
#顺序叠加神经网络
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

