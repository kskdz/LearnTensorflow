# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:35:48 2020
        本地MINST数据的读取
            基本的python数据操作
            D:\\BaiduNetdiskDownload\\mygit\\tensorflow2.0\\dataset\\MINST\\
            D:/BaiduNetdiskDownload/mygit/tensorflow2.0/dataset/MINST/
        文件名：(注意windows下和linux下的不同)
            t10k-images.idx3-ubyte
            t10k-labels.idx1-ubyte
            train-images.idx3-ubyte
            train-labels.idx1-ubyte
@author: 67443
"""
import os
import struct
import numpy as np
import matplotlib.pyplot as plt

#读取已经下载好的数据
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
    '''
    os.path.join() 路径拼接函数
    struct:
        python中的struct主要是用来处理C结构数据的，读入时先转换为Python的 字符串 类型，
        然后再转换为Python的结构化类型
        struct.pack()和struct.unpack()
        
        
    读取解释：
        magic number, 它是一个文件协议的描述, 也是在我们调用 fromfile 方法将字节读入 
        NumPy array 之前在文件缓冲中的 item 数(n)
        
        作为参数值传入 struct.unpack 的 >II 有两个部分:
            >: 这是指大端存储(用来定义字节是如何存储的); 
            I: 这是指一个无符号整数.
    '''
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    
    return images, labels

#文件读取
path = "D:/BaiduNetdiskDownload/mygit/tensorflow2.0/dataset/MINST/"
img,labels = load_mnist(path, kind='train') #训练用

#文件效果输出
fig, ax = plt.subplots(
    nrows=2,
    ncols=5,
    sharex=True,
    sharey=True, )

ax = ax.flatten()
for i in range(10):
    imgx = img[i].reshape(28, 28)
    ax[i].imshow(imgx, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


