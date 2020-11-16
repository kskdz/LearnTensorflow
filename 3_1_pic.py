# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 20:22:42 2020
    加载和预处理数据
        图像：以大量花卉图片为例子
@author: 67443
"""
import tensorflow as tf
import pathlib
import os
import random
import IPython.display as display
import matplotlib.pyplot as plt

'数据下载：花卉图像识别'
AUTOTUNE = tf.data.experimental.AUTOTUNE

#tensorflow中大量data数据放在keras中
data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         fname='flower_photos', untar=True)
#得到文件路径
data_root = pathlib.Path(data_root_orig)
print(data_root)
#文件下图片
for item in data_root.iterdir():
  print(item)

all_image_paths = list(data_root.glob('*/*'))  #获取指定目录下所有二级子文件
'''
glob.glob() 返回所有匹配的文件路径列表
#获取指定目录下的所有图片
print (glob.glob(r"/home/qiaoyunhao/*/*.png"),"\n") #加上r让字符串不转义
#获取上级目录的所有.py文件
print (glob.glob(r'../*.py')) #相对路径
'''
all_image_paths = [str(path) for path in all_image_paths]   #转化为字符串
random.shuffle(all_image_paths)  #随机排序

image_count = len(all_image_paths)
image_count   #总计3670张图片

all_image_paths[:10]

'检查图片'
#创建所有图片的索引字典:所有文件名都在LICENSE.txt下
attributions = (data_root/"LICENSE.txt").open(encoding='utf-8').readlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)

def caption_image(image_path):
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    image_rel = str(image_rel)
    image_rel = image_rel.replace("\\","/")
    return "Image (CC BY 2.0) " + ' - '.join(attributions[image_rel].split(' - ')[:-1])
    '''
    字符串处理：
        a[:-1] #到最后一个元素之前的所有元素
        a = ['a','b','c','d']
        a[:-1] = ['a', 'b', 'c']
        
        ' - '.join(stra) #在字符串a中的所有元素间加' - '字符合成一个新字符串
    '''    

    
for n in range(3):
  image_path = random.choice(all_image_paths)
  display.display(display.Image(image_path))
  print(caption_image(image_path))
  print()
#keyError：表示字典中没有这个key值

'确定每张图片的标签'
#列出可用标签
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
'''
    sorted(iterable, cmp=None, key=None, reverse=False)
    #sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
    #iterable -- 可迭代对象。
    #cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
    #key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
    #reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
'''
label_names

#为每一个标签分配索引
label_to_index = dict((name, index) for index, name in enumerate(label_names))
label_to_index

#创建一个列表，包含每个文件的标签索引：
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]
#all_image_labels 是字符串列表，用pathlib转化为路径，用parent.name选择最后一个
print("First 10 labels indices: ", all_image_labels[:10])

#加载和格式化数据：
img_path = all_image_paths[0]
img_path
img_raw = tf.io.read_file(img_path)
print(repr(img_raw)[:100]+"...")

#解码为图像张量
img_tensor = tf.image.decode_image(img_raw)

print(img_tensor.shape)
print(img_tensor.dtype)

#根据模型调整大小，并包装在一个简单的函数中，以备后用
img_final = tf.image.resize(img_tensor, [192, 192])
img_final = img_final/255.0
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)

image_path = all_image_paths[0]
label = all_image_labels[0]

plt.imshow(load_and_preprocess_image(img_path))
plt.grid(False)
plt.xlabel(caption_image(img_path))
plt.title(label_names[label].title())
print()

'''
创建一个 tf.data.Dataset： from_tensor_slices方法
'''
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
print(path_ds)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

plt.figure(figsize=(8,8))
for n, image in enumerate(image_ds.take(4)):
  plt.subplot(2,2,n+1)
  plt.imshow(image)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.xlabel(caption_image(all_image_paths[n]))
  plt.show()

#创建图像标签对 数据集
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
for label in label_ds.take(10):
  print(label_names[label.numpy()])

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
print(image_label_ds)

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

# 元组被解压缩到映射函数的位置参数中
def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

image_label_ds = ds.map(load_and_preprocess_from_path_label)
image_label_ds
'''
训练的基本方法：
    将数据打包为tf.data之后就可以利用其api快速实现
        充分打乱。
        分割为 batch。
        永远重复。
        尽快提供 batch。
'''
BATCH_SIZE = 32

# 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
# 被充分打乱。
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds
'''
这里有一些注意事项：

顺序很重要。
    在 .repeat 之后 .shuffle，会在 epoch 之间打乱数据（当有些数据出现两次的时候，其他数据还没有出现过）。
    在 .batch 之后 .shuffle，会打乱 batch 的顺序，但是不会在 batch 之间打乱数据。

你在完全打乱中使用和数据集大小一样的 buffer_size（缓冲区大小）。较大的缓冲区大小提供更好的随机化，但使用更多的内存，直到超过数据集大小。

在从随机缓冲区中拉取任何元素前，要先填满它。所以当你的 Dataset（数据集）启动的时候一个大的 buffer_size（缓冲区大小）可能会引起延迟。

在随机缓冲区完全为空之前，被打乱的数据集不会报告数据集的结尾。Dataset（数据集）由 .repeat 重新启动，导致需要再次等待随机缓冲区被填满。
    最后一点可以通过使用 tf.data.Dataset.apply 方法和融合过的 tf.data.experimental.shuffle_and_repeat 函数来解决:
'''
ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds

'传递数据集给模型'
#从 tf.keras.applications 取得 MobileNet v2 副本。简单的迁移学习
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False

#你需要将其范围从 [0,1] 转化为 [-1,1]
def change_range(image,label):
  return 2*image-1, label

keras_ds = ds.map(change_range)

# 数据集可能需要几秒来启动，因为要填满其随机缓冲区。
image_batch, label_batch = next(iter(keras_ds))

feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)

model = tf.keras.Sequential([
  mobile_net,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(len(label_names), activation = 'softmax')])

logit_batch = model(image_batch).numpy()

print("min logit:", logit_batch.min())
print("max logit:", logit_batch.max())
print()

print("Shape:", logit_batch.shape)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

len(model.trainable_variables)
model.summary()

steps_per_epoch=tf.math.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
steps_per_epoch

model.fit(ds, epochs=1, steps_per_epoch=3)

'性能分析'
#上面使用的简单 pipeline（管道）在每个 epoch 中单独读取每个文件。
#在本地使用 CPU 训练时这个方法是可行的，但是可能不足以进行 GPU 训练并且完全不适合任何形式的分布式训练。
import time
default_timeit_steps = 2*steps_per_epoch+1

def timeit(ds, steps=default_timeit_steps):
  overall_start = time.time()
  # 在开始计时之前
  # 取得单个 batch 来填充 pipeline（管道）（填充随机缓冲区）
  it = iter(ds.take(steps+1))
  next(it)

  start = time.time()
  for i,(images,labels) in enumerate(it):
    if i%10 == 0:
      print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(steps, duration))
  print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))
  print("Total time: {}s".format(end-overall_start))

#当前数据集的性能是：
ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
ds
timeit(ds)

#缓存：
#使用 tf.data.Dataset.cache 在 epoch 之间轻松缓存计算结果。这是非常高效的，特别是当内存能容纳全部数据时。
ds = image_label_ds.cache()
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
ds
timeit(ds)
#使用内存缓存的一个缺点是必须在每次运行时重建缓存，这使得每次启动数据集时有相同的启动延迟：
timeit(ds)

#如果内存不够容纳数据，使用一个缓存文件：
ds = image_label_ds.cache(filename='./cache.tf-data')
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(1)
ds
timeit(ds)
#这个缓存文件也有可快速重启数据集而无需重建缓存的优点
timeit(ds)

#TFRecord 文件
'''
原始图片数据
TFRecord 文件是一种用来存储一串二进制 blob 的简单格式。通过将多个示例打包进同一个文件内，
TensorFlow 能够一次性读取多个示例，当使用一个远程存储服务，如 GCS 时，这对性能来说尤其重要。
'''
#从原始图片数据中构建出一个 TFRecord 文件
image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.io.read_file)
tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
tfrec.write(image_ds)
#构建一个从 TFRecord 文件读取的数据集，并使用你之前定义的 preprocess_image 函数对图像进行解码/重新格
image_ds = tf.data.TFRecordDataset('images.tfrec').map(preprocess_image)
#压缩该数据集和你之前定义的标签数据集以得到期望的 (图片,标签) 对：
ds = tf.data.Dataset.zip((image_ds, label_ds))
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds=ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
ds
timeit(ds)  #这比 缓存 版本慢，因为你还没有缓存预处理。

#序列化的 Tensor（张量） 为 TFRecord 文件省去一些预处理过程
paths_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = paths_ds.map(load_and_preprocess_image)
image_ds

ds = image_ds.map(tf.io.serialize_tensor)
ds

tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
tfrec.write(ds)
#有了被缓存的预处理，就能从 TFrecord 文件高效地加载数据——只需记得在使用它之前反序列化：
ds = tf.data.TFRecordDataset('images.tfrec')

def parse(x):
  result = tf.io.parse_tensor(x, out_type=tf.float32)
  result = tf.reshape(result, [192, 192, 3])
  return result

ds = ds.map(parse, num_parallel_calls=AUTOTUNE)
ds

#像之前一样添加标签和进行相同的标准操作
ds = tf.data.Dataset.zip((ds, label_ds))
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds=ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
ds
timeit(ds)