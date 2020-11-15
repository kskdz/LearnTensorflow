# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 19:53:11 2020
    pd数据的可视化：
        实际上同样也是也是plot的使用方法
@author: 67443
"""
import pandas as pd

df = pd.read_excel('D:/BaiduNetdiskDownload/mygit/tensorflow2.0/dataset/team.xlsx')

# 可以不用写参数名，直接按位置传入
df[:5].plot('name', 'Q1')
df[:5].plot.bar('name', ['Q1', 'Q2'])
df[:5].plot.barh(x='name', y='Q4')
df[:5].plot.area('name', ['Q1', 'Q2'])
df[:5].plot.scatter('name', 'Q3')    #只允许有一个Y值

#图表边框相关
df[:5].plot(title='my plot')  #标题
df[:5].plot(fontsize=15)  #大小

'线条样式'
df[:5].plot(style=':') # 虚线
df[:5].plot(style='-.') # 虚实相间
df[:5].plot(style='--') # 长虚线
df[:5].plot(style='-') # 实线（默认）
df[:5].plot(style='.') # 点
df[:5].plot(style='*-') # 实线，数值为星星
df[:5].plot(style='^-') # 实线，数值为三角形
df[:5].plot(style=[':', '--', '.-', '*-'])

df[:5].plot(grid=True)  #背景辅助线

df[:5].plot(legend=False)      #取消图例
df[:5].plot(legend='reverse')  #反向排序图例

df[:5].plot.bar(figsize=(10.5,5)) #图形大小

df[:5].plot.barh(colormap='rainbow') #色系

'其他'
df[:10].plot.line(color='k') # 图的颜色
df[:5].plot.bar(rot=45) # 主轴上文字的方向度数

'组合:多条参数一起使用'
(
 df[:5]
 .set_index('name')
 .plot(style=[':', '--', '.-', '*-'])
)


