# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 19:33:07 2020
    2_6_SaveLoad
    模型的保存与载入
    保存和加载    
@author: kdz-pc
"""

import os

import tensorflow as tf
from tensorflow import keras

#pip install -q pyyaml h5py  # 以 HDF5 格式保存模型所必须
print(tf.version.VERSION)

#取MINST数据集合前1000个例子

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
#数据归一化
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

#构建模型
# 定义一个简单的序列模型
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model

# 创建一个基本的模型实例
model = create_model()

# 显示模型的结构
model.summary()

#在训练期间保存模型（以 checkpoints 形式保存）
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 创建一个保存模型权重的回调
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# 使用新的回调训练模型
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images,test_labels),
          callbacks=[cp_callback])  # 通过回调训练

# 这可能会生成与保存优化程序状态相关的警告。
# 这些警告（以及整个笔记本中的类似警告）
# 是防止过时使用，可以忽略。
#这将会创建一个 TensorFlow checkpoint 文件集合，这些文件在每个 epoch 结束时更新：

# 创建一个基本模型实例
model = create_model()

# 评估模型
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

#加载模型后 重新评估
# 加载权重
model.load_weights(checkpoint_path)

# 重新评估模型
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

#每5个创建一个回调
# 在文件名中包含 epoch (使用 `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 创建一个回调，每 5 个 epochs 保存模型的权重
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=5)

# 创建一个新的模型实例
model = create_model()

# 使用 `checkpoint_path` 格式保存权重
model.save_weights(checkpoint_path.format(epoch=0))

# 使用新的回调训练模型
model.fit(train_images, 
          train_labels,
          epochs=50, 
          callbacks=[cp_callback],
          validation_data=(test_images,test_labels),
          verbose=0)




