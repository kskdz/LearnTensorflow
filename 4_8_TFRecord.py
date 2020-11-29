# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:54:37 2020
    TFRecord 和 tf.Example        protocol buffers 协议缓冲区
    为了高效地读取数据，比较有帮助的一种做法是对数据进行序列化并将其存储在一组可线性读取的文件（每个文件 100-200MB）中。
    这尤其适用于通过网络进行流式传输的数据。这种做法对缓冲任何数据预处理也十分有用。
    TFRecord 格式是一种用于存储二进制记录序列的简单格式。
    
    协议缓冲区是一个跨平台、跨语言的库，用于高效地序列化结构化数据。
    协议消息由 .proto 文件定义，这通常是了解消息类型最简单的方法。

    tf.Example 消息（或 protobuf）是一种灵活的消息类型，表示 {"string": value} 映射。它专为 TensorFlow 而设计，并被用于 TFX 等高级 API。

    本笔记本将演示如何创建、解析和使用 tf.Example 消息，以及如何在 .tfrecord 文件之间对 tf.Example 消息进行序列化、写入和读取。
@author: 67443
"""
import tensorflow as tf

import numpy as np
import IPython.display as display

'tf.Example 的数据类型'
'''
从根本上讲，tf.Example 是 {"string": tf.train.Feature} 映射。

tf.train.Feature 消息类型可以接受以下三种类型（请参阅 .proto 文件）。大多数其他通用类型也可以强制转换成下面的其中一种：

tf.train.BytesList（可强制转换自以下类型）
    string
    byte
tf.train.FloatList（可强制转换自以下类型）
    float (float32)
    double (float64)
tf.train.Int64List（可强制转换自以下类型）
    bool
    enum
    int32
    uint32
    int64
    uint64
为了将标准 TensorFlow 类型转换为兼容 tf.Example 的 tf.train.Feature，可以使用下面的快捷函数。
请注意，每个函数会接受标量输入值并返回包含上述三种 list 类型之一的 tf.train.Feature：
'''
# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

'''
注：为了简单起见，本示例仅使用标量输入。
要处理非标量特征，最简单的方法是使用 tf.io.serialize_tensor 将张量转换为二进制字符串。
在 TensorFlow 中，字符串是标量。使用 tf.io.parse_tensor 可将二进制字符串转换回张量。
'''
print(_bytes_feature(b'test_string'))
print(_bytes_feature(u'test_bytes'.encode('utf-8')))

print(_float_feature(np.exp(1)))

print(_int64_feature(True))
print(_int64_feature(1))

'可以使用 .SerializeToString 方法将所有协议消息序列化为二进制字符串：'
feature = _float_feature(np.exp(1))

feature.SerializeToString()

'创建 tf.Example 消息'
'''
假设您要根据现有数据创建 tf.Example 消息。
在实践中，数据集可能来自任何地方，但是从单个观测值创建 tf.Example 消息的过程相同：
    在每个观测结果中，需要使用上述其中一种函数，将每个值转换为包含三种兼容类型之一的 tf.train.Feature。
    创建一个从特征名称字符串到第 1 步中生成的编码特征值的映射（字典）。
    将第 2 步中生成的映射转换为 Features 消息。
在此笔记本中，您将使用 NumPy 创建一个数据集。
此数据集将具有 4 个特征：
    具有相等 False 或 True 概率的布尔特征
    从 [0, 5] 均匀随机选择的整数特征
    通过将整数特征作为索引从字符串表生成的字符串特征
    来自标准正态分布的浮点特征
请思考一个样本，其中包含来自上述每个分布的 10,000 个独立且分布相同的观测值：
'''
# The number of observations in the dataset.
n_observations = int(1e4)

# Boolean feature, encoded as False or True.
feature0 = np.random.choice([False, True], n_observations)

# Integer feature, random from 0 to 4.
feature1 = np.random.randint(0, 5, n_observations)

# String feature
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]

# Float feature, from a standard normal distribution
feature3 = np.random.randn(n_observations)
'''
您可以使用 _bytes_feature、_float_feature 或 _int64_feature 将下面的每个特征强制转换为兼容 tf.Example 的类型。
然后，可以通过下面的已编码特征创建 tf.Example 消息：
'''
def serialize_example(feature0, feature1, feature2, feature3):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  feature = {
      'feature0': _int64_feature(feature0),
      'feature1': _int64_feature(feature1),
      'feature2': _bytes_feature(feature2),
      'feature3': _float_feature(feature3),
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

'''
例如，假设您从数据集中获得了一个观测值 [False, 4, bytes('goat'), 0.9876]。
您可以使用 create_message() 创建和打印此观测值的 tf.Example 消息。
如上所述，每个观测值将被写为一条 Features 消息。请注意，tf.Example 消息只是 Features 消息外围的包装器：
'''
# This is an example observation from the dataset.

example_observation = []

serialized_example = serialize_example(False, 4, b'goat', 0.9876)
serialized_example
'要解码消息，请使用 tf.train.Example.FromString 方法。'
example_proto = tf.train.Example.FromString(serialized_example)
example_proto

'TFRecords 格式详细信息'
'''
TFRecord 文件包含一系列记录。该文件只能按顺序读取。
每条记录包含一个字节字符串（用于数据有效负载），外加数据长度，
以及用于完整性检查的 CRC32C（使用 Castagnoli 多项式的 32 位 CRC）哈希值。
每条记录会存储为以下格式：
    uint64 length uint32 masked_crc32_of_length byte   data[length] uint32 masked_crc32_of_data
将记录连接起来以生成文件。此处对 CRC 进行了说明，且 CRC 的掩码为：
    masked_crc = ((crc >> 15) | (crc << 17)) + 0xa282ead8ul
'''
'''
注：不需要在 TFRecord 文件中使用 tf.Example。
tf.Example 只是将字典序列化为字节字符串的一种方法。
文本行、编码的图像数据，或序列化的张量（使用 tf.io.serialize_tensor，或在加载时使用 tf.io.parse_tensor）。
有关更多选项，请参阅 tf.io 模块。
'''


'使用 tf.data 的 TFRecord 文件'
'tf.data 模块还提供用于在 TensorFlow 中读取和写入数据的工具。'
tf.data.Dataset.from_tensor_slices(feature1)
#若应用于数组的元组，将返回元组的数据集：
features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
features_dataset

# Use `take(1)` to only pull one example from the dataset.
for f0,f1,f2,f3 in features_dataset.take(1):
  print(f0)
  print(f1)
  print(f2)
  print(f3)

'''
使用 tf.data.Dataset.map 方法可将函数应用于 Dataset 的每个元素。
映射函数必须在 TensorFlow 计算图模式下进行运算（它必须在 tf.Tensors 上运算并返回）。可以使用 tf.py_function 包装非张量函数（如 serialize_example）以使其兼容。
使用 tf.py_function 需要指定形状和类型信息，否则它将不可用：
'''
def tf_serialize_example(f0,f1,f2,f3):
  tf_string = tf.py_function(
    serialize_example,
    (f0,f1,f2,f3),  # pass these args to the above function.
    tf.string)      # the return type is `tf.string`.
  return tf.reshape(tf_string, ()) # The result is a scalar

tf_serialize_example(f0,f1,f2,f3)
#应用于数据集每一个元素
serialized_features_dataset = features_dataset.map(tf_serialize_example)
serialized_features_dataset

def generator():
  for features in features_dataset:
    yield serialize_example(*features)

serialized_features_dataset = tf.data.Dataset.from_generator(
    generator, output_types=tf.string, output_shapes=())

serialized_features_dataset
#并将它们写入 TFRecord 文件：
filename = 'test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)

'读取TFRecord文件'
'''
读取 TFRecord 文件
您还可以使用 tf.data.TFRecordDataset 类来读取 TFRecord 文件。

有关通过 tf.data 使用 TFRecord 文件的详细信息，请参见此处。

使用 TFRecordDataset 对于标准化输入数据和优化性能十分有用。
'''
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset

for raw_record in raw_dataset.take(10):
  print(repr(raw_record))

#可以使用以下函数对这些张量进行解析。请注意，这里的 feature_description 是必需的，因为数据集使用计算图执行，并且需要以下描述来构建它们的形状和类型签名：
# Create a description of the features.
feature_description = {
    'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
}

def _parse_function(example_proto):
  # Parse the input `tf.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)

'或者，使用 tf.parse example 一次解析整个批次。使用 tf.data.Dataset.map 方法将此函数应用于数据集中的每一项：'
parsed_dataset = raw_dataset.map(_parse_function)
parsed_dataset

'使用 Eager Execution 在数据集中显示观测值。此数据集中有 10,000 个观测值，但只会显示前 10 个。数据会作为特征字典进行显示。每一项都是一个 tf.Tensor，此张量的 numpy 元素会显示特征的值：'
for parsed_record in parsed_dataset.take(10):
  print(repr(parsed_record))

'Python 中的 TFRecord 文件'
#tf.io 模块还包含用于读取和写入 TFRecord 文件的纯 Python 函数。
'写入 TFRecord 文件'
# Write the `tf.Example` observations to the file.
with tf.io.TFRecordWriter(filename) as writer:
  for i in range(n_observations):
    example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
    writer.write(example)
    
'读取 TFRecord 文件'
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset

for raw_record in raw_dataset.take(1):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  print(example)
  
'举例：读取和写入图像数据'
'''
下面是关于如何使用 TFRecord 读取和写入图像数据的端到端示例。您将使用图像作为输入数据，将数据写入 TFRecord 文件，然后将文件读取回来并显示图像。

如果您想在同一个输入数据集上使用多个模型，这种做法会很有用。您可以不以原始格式存储图像，而是将图像预处理为 TFRecord 格式，然后将其用于所有后续的处理和建模中。

首先，让我们下载雪中的猫的图像，以及施工中的纽约威廉斯堡大桥的照片。
'''
cat_in_snow  = tf.keras.utils.get_file('320px-Felis_catus-cat_on_snow.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')
williamsburg_bridge = tf.keras.utils.get_file('194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')

#显示
display.display(display.Image(filename=cat_in_snow))
display.display(display.HTML('Image cc-by: &lt;a "href=https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg"&gt;Von.grzanka&lt;/a&gt;'))
display.display(display.Image(filename=williamsburg_bridge))
display.display(display.HTML('&lt;a "href=https://commons.wikimedia.org/wiki/File:New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg"&gt;From Wikimedia&lt;/a&gt;'))

'写入 TFRecord 文件'
'''
和以前一样，将特征编码为与 tf.Example 兼容的类型。
这将存储原始图像字符串特征，以及高度、宽度、深度和任意 label 特征。
后者会在您写入文件以区分猫和桥的图像时使用。将 0 用于猫的图像，将 1 用于桥的图像：
'''
image_labels = {
    cat_in_snow : 0,
    williamsburg_bridge : 1,
}
# This is an example, just using the cat image.
image_string = open(cat_in_snow, 'rb').read()

label = image_labels[cat_in_snow]

# Create a dictionary with features that may be relevant.
def image_example(image_string, label):
  image_shape = tf.image.decode_jpeg(image_string).shape

  feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

for line in str(image_example(image_string, label)).split('\n')[:15]:
  print(line)
print('...')
#写文件
# Write the raw image files to `images.tfrecords`.
# First, process the two images into `tf.Example` messages.
# Then, write to a `.tfrecords` file.
record_file = 'images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
  for filename, label in image_labels.items():
    image_string = open(filename, 'rb').read()
    tf_example = image_example(image_string, label)
    writer.write(tf_example.SerializeToString())

#读文件
raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

# Create a dictionary describing the features.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
parsed_image_dataset
#恢复图像
for image_features in parsed_image_dataset:
  image_raw = image_features['image_raw'].numpy()
  display.display(display.Image(data=image_raw))




