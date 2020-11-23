# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:54:12 2020
    Tensorflow Text库提也提供了一些与文本相关的类与操作集合
    该库可以执行基于文本的模型所需的常规预处理，并包括其他对核心TensorFlow未提供的对序列建模有用的功能。
    在文本预处理中使用这些操作的优势是它们在TensorFlow图中完成。您无需担心训练中的标记化与推理或管理预处理脚本中的标记化不同。

    但是conda和text中都没有下载到tensorflow_text  阿哲，我查查
    这个包只支持支持MacOS和Linux，不支持Windows，windows下无法下载
@author: 67443
"""
import tensorflow as tf
import tensorflow_text as text

'Unicode码问题'
'大多数操作需要输入的字符为UTF-8，如果使用其他编码方法，先用tf核中的方法转化即可'
docs = tf.constant([u'Everything not saved will be lost.'.encode('UTF-16-BE'), u'Sad☹'.encode('UTF-16-BE')])
utf8_docs = tf.strings.unicode_transcode(docs, input_encoding='UTF-16-BE', output_encoding='UTF-8')

'Tokenization 货币化/令牌化'




