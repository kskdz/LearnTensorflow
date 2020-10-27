# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 19:28:29 2020
    第一个tensorflow小程序，介绍tensorflow的常量变量、数据类型
    
@author: kdz-pc
"""
import tensorflow as tf

'''
#常量创建 类型可以省略
hello = tf.constant('Hello,tensorflow!',dtype = tf.string)
#省略创建如下
a = tf.constant(1)
#变量创建
b = tf.Variable(1.0,dtype = tf.float32)
c = tf.Variable(1.0,dtype = tf.float64)
'''
'''
input1 = tf.constant(1)
print(input1)
input2 = tf.Variable(2,tf.int32)
print(input2)

input2 = input1 
sess = tf.Session()
print(sess.run(input2))
'''
 
a=tf.Variable(5)
print(a)

b=tf.Variable(10)
print(b)
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    print(sess.run(b))
