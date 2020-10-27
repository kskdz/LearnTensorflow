# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:17:52 2020
    keras基础知识：基本图像分类
    
@author: kdz-pc
"""
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

'''
#数据需要从谷歌上下载，所以需要翻墙
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
'''
#mnist = tf.keras.datasets.mnist
#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#class_names = ['0','1','2','3','4','5','6','7','8','9']

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#灰度查看图像
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()  #色阶的颜色栏
plt.grid(False) #显示网格线
plt.show()

#显示训练集合前25图像
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#归一化处理
train_images = train_images / 255.0
test_images = test_images / 255.0

#创建模型
'''
Flatten:将图像的格式从二维数组（28 x 28像素）转换为一维数组（28 * 28 = 784像素）。
  可以将这一层看作是堆叠图像中的像素行并将它们排成一行。该层没有学习参数。它只会重新格式化数据。
Dense:这些是紧密连接或完全连接的神经层。第一Dense层具有128个节点（或神经元）。第二层（也是最后一层）返回长度为10的logits数组。
  每个节点包含一个得分，该得分指示当前图像属于10个类之一。
'''
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

#编译模型
'''
损失函数 -衡量训练期间模型的准确性。您希望最小化此功能，以在正确的方向上“引导”模型。
优化器 -这是基于模型看到的数据及其损失函数来更新模型的方式。
指标 -用于监视培训和测试步骤。以下示例使用precision ，即正确分类的图像比例。
'''
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#训练模型
model.fit(train_images, train_labels, epochs=10)
    
#评估准确性
'''
测试数据集的准确性略低于训练数据集的准确性。训练准确性和测试准确性之间的差距代表过度拟合 。
当机器学习模型在新的，以前看不见的输入上的表现比训练数据上的表现差时，就会发生过度拟合。
过度拟合的模型“记忆”训练数据集中的噪声和细节，从而对新数据的模型性能产生负面影响。
'''
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)


#做出预测：
'预测模型'
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
'应用预测模型'
predictions = probability_model.predict(test_images)   
    
'图像绘制函数'
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:  #设定下标颜色
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10)) #x轴标签
  plt.yticks([])        #y轴省略
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#查看单幅图的置信程度
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

#查看整体的置信程度
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

#使用训练好的模型
'单个图像'
# Grab an image from the test dataset.
img = test_images[1]
print(img.shape)
'因此，即使您使用的是单个图像，也需要将其添加到列表中：'
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(img.shape)
'图像预测正确的标签'
predictions_single = probability_model.predict(img)
print(predictions_single)
'keras.Model.predict返回一个列表列表-数据批次中每个图像的一个列表。批量获取我们（仅）图像的预测'
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
np.argmax(predictions_single[0])





    


