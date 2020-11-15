# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 19:56:41 2020
     keras基础知识：基本文本分类
     
         文本评论是记得还是消极的
         我们将使用来源于网络电影数据库（Internet Movie Database）的 IMDB 数据集（IMDB dataset），
         其包含 50,000 条影评文本。从该数据集切割出的 25,000 条评论用作训练，另外 25,000 条用作测试。
         训练集与测试集是平衡的（balanced），意味着它们包含相等数量的积极和消极评论。
数据下载部分：
'Fashion MNIST数据集下载'
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

'IMDB数据集下载'
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

'IMDB数据集'
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

'Auto MPG数据集'
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path

'希格斯数据集'
gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')

数据集下载一次就会保存在 C:/Users/kdz-pc/.keras/datasets 中
#(unicode error) 'unicodeescape' codec can't decode bytes in position 2428-2429: truncated \\UXXXXXXXX escape
#即使在注释中，也不能有 \\U 的形式
@author: kdz-pc
"""

import tensorflow as tf
from tensorflow import keras

import numpy as np

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#数据显示：
'评论是以单词的数字形式显示的'
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])

# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()

# 保留第一个索引
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) #构建字典

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

#数据预测处理
'''
神经网络的输入必须为：等长的，所以需要转换
    思路一：设置一个足够高长度的张量，以0和1反映出这个单词是否会出现
    思路二：将所有的句子都填充到最长的一个句子的状态
'''
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

#构建模型
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))    #将数字ID转换为-1到1之间的数字
model.add(keras.layers.GlobalAveragePooling1D())     #全局平均池化
model.add(keras.layers.Dense(16, activation='relu')) #relu层
model.add(keras.layers.Dense(1, activation='sigmoid')) #映射到01

model.summary()  #显示网络层

'''
第一层是嵌入（Embedding）层。该层采用整数编码的词汇表，并查找每个词索引的嵌入向量（embedding vector）。这些向量是通过模型训练学习到的。向量向输出数组增加了一个维度。
得到的维度为：(batch, sequence, embedding)。

接下来，GlobalAveragePooling1D 将通过对序列维度求平均值来为每个样本返回一个定长输出向量。这允许模型以尽可能最简单的方式处理变长输入。
该定长输出向量通过一个有 16 个隐层单元的全连接（Dense）层传输。

最后一层与单个输出结点密集连接。使用 Sigmoid 激活函数，其函数值为介于 0 与 1 之间的浮点数，表示概率或置信度。
'''

#模型设置：
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#设置验证集合与训练
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

#评估模型
results = model.evaluate(test_data,  test_labels, verbose=2)
print(results)

'model.fit() 返回一个 History 对象，该对象包含一个字典，其中包含训练阶段所发生的一切事件'
history_dict = history.history
history_dict.keys()

'显示训练效果'
import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# “bo”代表 "蓝点"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b代表“蓝色实线”
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

'显示为准确率'
plt.clf()   # 清除数字

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()







