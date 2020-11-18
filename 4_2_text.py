# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 20:11:18 2020
    文本数据处理：
        以荷马的伊利亚特为例子 来进行实验
        William Cowper — text
https://storage.googleapis.com/download.tensorflow.org/data/illiad/cowper.txt
        Edward, Earl of Derby — text
https://storage.googleapis.com/download.tensorflow.org/data/illiad/derby.txt
        Samuel Butler — text
https://storage.googleapis.com/download.tensorflow.org/data/illiad/butler.txt
#本教程中使用的文本文件已经进行过一些典型的预处理，主要包括删除了文档页眉和页脚，行号，章节标题。请下载这些已经被局部改动过的文件。

        通过单行本确定翻译者
@author: 67443
"""
import tensorflow as tf

import tensorflow_datasets as tfds
import os

'数据下载'
DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

for name in FILE_NAMES:
  text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL+name)

parent_dir = os.path.dirname(text_dir)

parent_dir

'文本加载到数据集中'
def labeler(example, index):
  return example, tf.cast(index, tf.int64)  

labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):   #枚举很好用啊
  lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
  #这个接口会自动构造一个dataset，类中保存的元素：文中一行，就是一个元素，是string类型的tensor
  labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
  #匿名函数 冒号前是参数，冒号后是返回值
  labeled_data_sets.append(labeled_dataset)
  #3合成1

#合并数据集 并进行随机化操作
BUFFER_SIZE = 50000   #缓冲区大小
BATCH_SIZE = 64       #训练batch
TAKE_SIZE = 5000      #测试集合大小

#为什么要再合成一遍？没有复制data功能吗？
all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
  all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
  #concatenate 两个Dataset对象进行合并或连接

#随机化
all_labeled_data = all_labeled_data.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=False)
'''
shuffle(buffer_size, seed=None,reshuffle_each_iteration=True)
buffer_size参数，指元素的个数，最完美的shuffle是所有数据一起shuffle,但是避免内存不够，每次选buffer_size个数据进行shuffle
默认reshuffle_each_iteration=True,即下次对dataset操作（如切片、复制、取batch）时都会再次执行洗牌操作。
    则表示每次迭代时都应对数据集进行伪随机重组
'''

for ex in all_labeled_data.take(5):
  print(ex)
  
'文本编码转化为数字'
#创建词汇表  通过将文本标记为单独的单词集合来构建词汇表
tokenizer = tfds.features.text.Tokenizer()
#其中 tokenizer = tfds.features.text.Tokenizer() 的目的是实例化一个分词器，tokenizer.tokenize 可以将一句话分成多个单词。
vocabulary_set = set()
for text_tensor, _ in all_labeled_data:
  some_tokens = tokenizer.tokenize(text_tensor.numpy())
  vocabulary_set.update(some_tokens)  #合并两个集合

vocab_size = len(vocabulary_set)
vocab_size

'样本编码'
#创建一个编码器
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

#编码器效果举例：
example_text = next(iter(all_labeled_data))[0].numpy()
print(example_text)
encoded_example = encoder.encode(example_text)
print(encoded_example)

#在数据集上运行编码器
def encode(text_tensor, label):
  encoded_text = encoder.encode(text_tensor.numpy())
  return encoded_text, label

def encode_map_fn(text, label):
  # py_func doesn't set the shape of the returned tensors.
  encoded_text, label = tf.py_function(encode, 
                                       inp=[text, label], 
                                       Tout=(tf.int64, tf.int64))
  '''
  Python函数包装成一个tensorflow的operation操作
  不管函数func返回的是什么内容，py_function的返回值总是会自动将其包装成与之对应的匹配的Tensor再返回
  tensorflow为数据提供更高效的处理方法封装
  '''
  # `tf.data.Datasets` work best if all components have a shape set
  #  so set the shapes manually: 
  encoded_text.set_shape([None])
  label.set_shape([])
  
  return encoded_text, label


all_encoded_data = all_labeled_data.map(encode_map_fn)

'''
将数据集拆分为测试集和训练集进行分支：
使用 tf.data.Dataset.take 和 tf.data.Dataset.skip 来建立一个小一些的测试数据集和稍大一些的训练数据集。

在数据集被传入模型之前，数据集需要被分批。最典型的是，每个分支中的样本大小与格式需要一致。
但是数据集中样本并不全是相同大小的（每行文本字数并不相同）。因此，使用 tf.data.Dataset.padded_batch（而不是 batch ）将样本填充到相同的大小。
'''
# tf.TensorShape([])     表示长度为单个数字
# tf.TensorShape([None]) 表示长度未知的向量
padded_shapes = (
    tf.TensorShape([None]),
    tf.TensorShape([])
)

train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes = padded_shapes)
'''
batch_size：在单个批次中合并的此数据集的连续元素个数。
padded_shapes：tf.TensorShape或其他描述tf.int64矢量张量对象，表示在批处理之前每个输入元素的各个组件应填充到的形状。如果参数中有None，则表示将填充为每个批次中该尺寸的最大尺寸。
padding_values：要用于各个组件的填充值。默认值0用于数字类型，字符串类型则默认为空字符。
drop_remainder：如果最后一批的数据量少于指定的batch_size，是否抛弃最后一批，默认为False，表示不抛弃。

主要问题在于：train_data是由一个数组和一个标签组成的，对齐只需要对其数组，需要在padded_shapes中表达出来
'''
test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes = padded_shapes)

#现在test_data 和 train_data 不是（ example, label ）对的集合，而是批次的集合，输出实例
sample_text, sample_labels = next(iter(test_data))
sample_text[0], sample_labels[0]

vocab_size += 1 #引入0最为填充，词汇表增加

'''
建立模型开始训练
'''
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 64))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

# 一个或多个紧密连接的层
# 编辑 `for` 行的列表去检测层的大小
for units in [64, 64]:
  model.add(tf.keras.layers.Dense(units, activation='relu'))

# 输出层。第一个参数是标签个数。
model.add(tf.keras.layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, epochs=3, validation_data=test_data)

eval_loss, eval_acc = model.evaluate(test_data)

print('\nEval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))
