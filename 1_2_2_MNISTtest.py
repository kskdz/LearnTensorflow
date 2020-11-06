# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:18:45 2020
    在默认的MINST数据集合无法下载的情况下,试着自己处理一下
@author: 67443
"""
import os
import struct
import numpy as np
import tensorflow as tf

#读取已经下载好的数据
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    
    return images, labels

#数据读取和简单处理
path = "D:/BaiduNetdiskDownload/mygit/tensorflow2.0/dataset/MINST/"
x_train,y_train = load_mnist(path , kind = 'train')
x_test,y_test = load_mnist(path , kind = 't10k')   #测试集

x_train, x_test = x_train / 255.0, x_test / 255.0       #数据归一化

#神经网络搭建
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(784),           #问题:Flatten(28,28)是否和这个等价？
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)