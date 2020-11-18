# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 17:02:29 2020
    关于 tf.data.Dataset 的学习：
        Dataset可以看作是相同类型“元素”的有序列表。
        在实际使用时，单个“元素”可以是向量，也可以是字符串、图片，甚至是tuple或者dict。
    
    GPU的使用
@author: 67443
"""

import tensorflow as tf
import numpy as np
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0,2.0,3.0,4.0,5.0]))

#2.X 写法改变 data_it = tf.compat.v1.data.make_one_shot_iterator(dataset)
iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
one_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(5):
        print(sess.run(one_element))   
        
#tf.data.Dataset.from_tensor_slices真正的作用是切分传入Tensor的第一个维度，生成相应的dataset
dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=(5, 2)))

dataset = tf.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),                                       
        "b": np.random.uniform(size=(5, 2))
    }
)

