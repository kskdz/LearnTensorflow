# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 16:58:41 2020
    2_5_过拟合与欠拟合 
    除了要在训练集合上达到比较高的进度，测试集合上的提高很大程度上需要一定房子过拟合的技巧
    需要希格斯数据集合 明天
@author: kdz-pc
"""
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import regularizers

print(tf.__version__)

'''
#!pip install -q git+https://github.com/tensorflow/docs
python下载第三方库的方法
从git上下载包的方法：
在git上下载zip解压之后，文件中有setup.py文件
cmd cd到文件夹所在的位置
然后激活对应环境,activate tensorflow-gpu
然后激活 python setup.py install 
文件最终会添加在C:\Users\kdz-pc\Anaconda3\Lib\site-packages 中
'''
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from  IPython import display
from matplotlib import pyplot as plt

import numpy as np

import pathlib
import shutil
import tempfile

'''
首先介绍希格斯数据集合：
    并不是要研究量子物理，所以只需要知道数据集合
     它包含11 000 000个示例，每个示例具有28个功能以及一个二进制类标签。
     
     csv文件查看头尾，可以用pandas
     import pandas as pd
     filename = './data/iris.data.csv'
     names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
     dataset = pd.read_csv(filename, names=names) # 这个数据集没有头部，手动指定即可
     dataset.head()
     
     极大批量操作，还是tf设计的数据结构快一点
'''
logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

'CSV转化为数据集合:由于没有默认文件头，所以会导致如果首行有重复数据就会报错'
LABEL_COLUMN = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24',
                '25','26','27','28','29']
file_path = 'D:\\DataSet\\HIGGS.csv'
def get_dataset(file_path):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=12, # 为了示例更容易展示，手动设置较小的值
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True)
  return dataset
HIGGSset = get_dataset(file_path)

'采用gz格式打开'
gz = 'D:\\DataSet\\HIGGS.csv.gz'
FEATURES = 28
ds = tf.data.experimental.CsvDataset(gz,[float(),]*(FEATURES+1), compression_type="GZIP")
#该csv阅读器类为每个记录返回一个标量列表。 
#以下函数将该标量列表重新打包为（feature_vector，label）对。
def pack_row(*row):
  label = row[0]
  features = tf.stack(row[1:],1)
  return features, label

#重新打包:返回一个数据集合
packed_ds = ds.batch(10000).map(pack_row).unbatch()

for features,label in packed_ds.batch(1000).take(1):
  print(features[0])
  plt.hist(features.numpy().flatten(), bins = 101)

'创建训练集合和测试集合：10000个数据进行训练1000个数据进行测试'
N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE
'Dataset.skip和Dataset.take方法使此操作变得容易。'
'同时，使用Dataset.cache方法来确保加载器不需要在每个时期重新从文件中读取数据：'
validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()
train_ds

validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

'''
展示过拟合现象:
    问题的关键在于如何
'''

'Training procedure 训练预处理'
'如果在训练过程中逐渐降低学习率，许多模型的训练效果会更好。' 
'使用optimizers.schedules可以随着时间的推移降低学习率：'
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)
'上面的代码设置了一个schedules.InverseTimeDecay，以双曲线的方式将学习速率在1000个时代降低到基本速率的1/2，'
'在2000个时代降低1/3，依此类推。以图像显示学习率'
step = np.linspace(0,100000)
lr = lr_schedule(step)
plt.figure(figsize = (8,6))
plt.plot(step/STEPS_PER_EPOCH, lr)
plt.ylim([0,max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('Learning Rate')
'''
本教程中的每个模型都将使用相同的训练配置。 因此，从回调列表开始，以可重用的方式设置它们。

本教程的培训会持续很短的时间。 为了减少记录噪音，请使用tfdocs.EpochDots，它仅显示一个。 
每个时期，以及每100个时期的完整指标。

接下来包括callbacks.EarlyStopping以避免冗长和不必要的培训时间。 
请注意，此回调设置为监视val_binary_crossentropy，而不是val_loss。 这种差异稍后将变得很重要。

使用callbacks.TensorBoard生成用于训练的TensorBoard日志。
'''
def get_callbacks(name):
  return [
    tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]

def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
  if optimizer is None:
    optimizer = get_optimizer()
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[
                  tf.keras.losses.BinaryCrossentropy(
                      from_logits=True, name='binary_crossentropy'),
                  'accuracy'])

  model.summary()

  history = model.fit(
    train_ds,
    steps_per_epoch = STEPS_PER_EPOCH,
    epochs=max_epochs,
    validation_data=validate_ds,
    callbacks=get_callbacks(name),
    verbose=0)
  return history

'极小的模型'
tiny_model = tf.keras.Sequential([
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(1)
])
size_histories = {}
size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')
#查看极小模型效果
plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])

'较小的模型'
small_model = tf.keras.Sequential([
    # `input_shape` is only required here so that `.summary` works.
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(16, activation='elu'),
    layers.Dense(1)
])
size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')

'中等模型'
medium_model = tf.keras.Sequential([
    layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(64, activation='elu'),
    layers.Dense(64, activation='elu'),
    layers.Dense(1)
])
size_histories['Medium']  = compile_and_fit(medium_model, "sizes/Medium")   

'较大的模型'
large_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(1)
])
size_histories['large'] = compile_and_fit(large_model, "sizes/large")

'所有的模型比较:验证集损失和测试集损失'
'''
实线表示训练集损失，虚线是测试集合损失;虚线越低越好

虽然构建较大的模型可以提供更多功能，但是如果不以某种方式限制此功能，则可以轻松地将其过度拟合至训练集。

在此示例中，通常，只有“ Tiny”模型设法避免完全过拟合，并且每个较大的模型都更快地过拟合数据。 对于“大型”模型而言，这变得如此严重，以至于您需要将绘图切换为对数比例才能真正看到正在发生的事情。

差异很小是正常的。
如果两个指标都朝着同一方向发展，那么一切都很好。
如果在培训指标继续提高的同时，验证指标开始停滞不前，那么您可能已经过拟合了。
如果验证指标的方向错误，则表明该模型过度拟合。
'''
plotter.plot(size_histories)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5, 0.7])
plt.xlabel("Epochs [Log Scale]")

'防止过度拟合'
'将tiny模型的日志作为比较标准'
shutil.rmtree(logdir/'regularizers/Tiny', ignore_errors=True)
shutil.copytree(logdir/'sizes/Tiny', logdir/'regularizers/Tiny')

regularizer_histories = {}
regularizer_histories['Tiny'] = size_histories['Tiny']

'方法1：增加权重调节'
'将参数规模限制，避免参数规模无限制增长,L1和L2正则化'
l2_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001),
                 input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])

regularizer_histories['l2'] = compile_and_fit(l2_model, "regularizers/l2")
#绘制图像
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
#可以明显看到L2正则化有更好的泛化能力，相比较能更好的修正正确率
result = l2_model(features)
regularization_loss=tf.add_n(l2_model.losses)
'''
这种正则化有两种问题：
首先：如果您正在编写自己的训练循环，则需要确保向模型询问其正则化损失。
第二：此实现方式是通过对模型的损失添加权重惩罚，然后再应用标准优化程序来实现的。
还有第二种方法，它仅对原始损耗运行优化器，然后在应用计算出的步骤时，优化器还应用一些权重衰减。 这种“解耦的权重衰减”可在诸如Optimizer.FTRL和Optimizers.AdamW之类的优化器中看到。
'''

'增加dropout方法'
'dropout是非常流行的正则化技术，让我们试一试增加两个dropout层'
dropout_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

regularizer_histories['dropout'] = compile_and_fit(dropout_model, "regularizers/dropout")

plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])

'dropout + L2'
combined_model = tf.keras.Sequential([
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

regularizer_histories['combined'] = compile_and_fit(combined_model, "regularizers/combined")

plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])




