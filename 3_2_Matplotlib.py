# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 21:41:34 2020
    数据的可视化：直观的反应出数据的偏差和集中程度
    1、差异的可视化
@author: kdz-pc
"""

import numpy as np
import pylab
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plot

'''
data = np.mat([[1,200,105,3,False],[2,165,80,2,False],[3,184.5,120,2,False],
              [4,116,70.8,1,False],[5,270,150,4,True]])
col1 = []
for row in data:
    col1.append(row[0,1])
    
stats.probplot(col1,plot=pylab)
pylab.show()
'''

rocksVMines = pd.DataFrame([[1,200,105,3,False],[2,165,80,2,False],[3,184.5,120,2,False],
              [4,116,70.8,1,False],[5,270,150,4,True]])

dataRow1 = rocksVMines.iloc[1,0:3]
dataRow2 = rocksVMines.iloc[2,0:3]
plot.scatter(dataRow1,dataRow2)
plot.xlabel("特征1")
plot.ylabel("特征2")
plot.show()

