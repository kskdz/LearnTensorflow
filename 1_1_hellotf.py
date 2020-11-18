# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 21:08:40 2020
    tf.keras是tensorflow的核心
    简单实现,hello keras
@author: kdz-pc
"""
import tensorflow as tf
#from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('D:/BaiduNetdiskDownload/mygit/tensorflow2.0/dataset/income.csv')
plt.scatter(data.Education,data.Income) #绘制散点图
#data.plot.scatter('Education','Income')
plt.show()

x = data.Education
y = data.Income

model = tf.keras.Sequential()   #创建顺序框架，为空模型
model.add(tf.keras.layers.Dense(1, input_shape = (1,))) #Dense层就是所谓的全连接神经网络层
model.summary()  #反应模型整体信息 #ax+b

model.compile(optimizer = 'adam',  #优化算法：梯度下降算法
              loss = 'mse')         #损失函数
              
history = model.fit(x,y,epochs = 5000) #循环次数

model.predict(x)

