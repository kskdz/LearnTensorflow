# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 21:04:40 2020
    数据分析：
    数据分析和处理：数据收集中会产生大量缺失，所以常用均值或者和目标相近似的数据替代
    基于统计分析的数据处理：常用的数据特征计算
@author: kdz-pc
"""
import numpy as np
#numpy在存储和处理大型矩阵，比python自身的嵌套表（nested list structure）结构要高效的多

data = np.mat([[1,200,105,3,False],[2,165,80,2,False],[3,184.5,120,2,False],
              [4,116,70.8,1,False],[5,270,150,4,True]])
row = 0
for line in data:
    row += 1
    
print('____')
print(row)
print(data.size)

print('____')
print( print( data[0,3]))
print(print(data[0,4]))

print('____')
print(data)

print('____')
col1 = []
for row in data:
    col1.append(row[0,1])
print( np.sum(col1))
print(np.mean(col1))
print(np.std(col1))
print(np.var(col1))