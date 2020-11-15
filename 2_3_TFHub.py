# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:24:37 2020
     keras基础知识：TFHub
         文本评论是积极的还是消极的 采用迁移学习的方法进行文本分类
         
         说白了，就是联系下载好的一个已经预训练好的隐藏层插入网络然后进行训练
@author: kdz-pc
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

#环境确定
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

#数据载入与探索
imdb = tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
train_examples_batch = train_data[:10]       #这样下载的数据类型是一个array类型不是Tensor类型

train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

'''
#这个数据是从tensorflow_datasets中下载数据
#
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

#然后把下载训练过的层进行配置
embedding = "https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1"  #外网需要VPN,直接下会崩溃
hub_layer = hub.KerasLayer(embedding, input_shape=[],  
                           dtype=tf.string, trainable=True) #设置迁移学习的层

hub_layer(train_examples_batch[:3]) #简单输出实例
'''
'''
一般来讲：神经网络的输入应当样式统一，也就是说需要把数据规模统一
但是，可以采用把句子变成嵌入向量的模式来规避这个过程。
我们可以使用一个预先训练好的文本嵌入（text embedding）作为首层，这将具有三个优点：

我们不必担心文本预处理
我们可以从迁移学习中受益
嵌入具有固定长度，更易于处理

针对此示例我们将使用 TensorFlow Hub 中名为 google/tf2-preview/gnews-swivel-20dim/1 的一种预训练文本嵌入（text embedding）模型 。

为了达到本教程的目的还有其他三种预训练模型可供测试：

google/tf2-preview/gnews-swivel-20dim-with-oov/1 ——类似 google/tf2-preview/gnews-swivel-20dim/1，但 2.5%的词汇转换为未登录词桶（OOV buckets）。如果任务的词汇与模型的词汇没有完全重叠，这将会有所帮助。
google/tf2-preview/nnlm-en-dim50/1 ——一个拥有约 1M 词汇量且维度为 50 的更大的模型。
google/tf2-preview/nnlm-en-dim128/1 ——拥有约 1M 词汇量且维度为128的更大的模型。
让我们首先创建一个使用 Tensorflow Hub 模型嵌入（embed）语句的Keras层，并在几个输入样本中进行尝试。请注意无论输入文本的长度如何，嵌入（embeddings）输出的形状都是：(num_examples, embedding_dimension)。
'''
#配置迁移学习层
embedding = "https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1"  #外网需要VPN,下有可能会崩溃
hub_layer = hub.KerasLayer(embedding, input_shape=[],  
                           dtype=tf.string, trainable=True)     #输入必须为一个Tensor类型
hub_layer(train_examples_batch[:3])

#构建函数
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()


#配置训练
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
#训练
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

#评估模型
results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))


