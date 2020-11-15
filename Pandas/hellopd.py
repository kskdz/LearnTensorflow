# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 16:24:23 2020
    Pandas Demo
        学习几个简单的Pandas例子
        系列、数据帧、面板
    例子来源：https://www.yiibai.com/pandas/python_pandas_quick_start.html
             https://www.gairuo.com/p/pandas-quick-start
@author: 67443
"""
import pandas as pd
import numpy as np

#Series 数据创建
s = pd.Series([1,3,5,np.nan,6,8])
print(s)

#DataFrame 创建1 一般方式
dates = pd.date_range('20170101', periods=7)
print(dates)

print("--"*16)
''
df = pd.DataFrame(np.random.randn(7,4), index=dates, columns=list('ABCD'))
print(df)

#DataFrame 创建2 类似字典方式
df2 = pd.DataFrame({ 'A' : 1.,
                     'B' : pd.Timestamp('20170102'),
                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                     'D' : np.array([3] * 4,dtype='int32'),
                     'E' : pd.Categorical(["test","train","test","train"]),
                     'F' : 'foo' })

print(df2)
'或者字典直接创建'
dictx = { 'A' : 1.,
         'B' : pd.Timestamp('20170102'),
         'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
         'D' : np.array([3] * 4,dtype='int32'),
         'E' : pd.Categorical(["test","train","test","train"]),
         'F' : 'foo' }
df2 = pd.DataFrame(data = dictx)

#DataFrame 创建3 csv或者xlsx文件创建
df3 = pd.read_excel('D:/BaiduNetdiskDownload/mygit/tensorflow2.0/dataset/team.xlsx')
df4 = pd.read_csv('D:/BaiduNetdiskDownload/mygit/tensorflow2.0/dataset/income.csv')

#查看整体描述
df = df3
df.shape # (100, 6) 查看行数和列数
df.info() # 查看索引、数据类型和内存信息 数据类型、索引情况、行列数、各字段数据类型、内存占用
df.describe() # 查看数值型列的汇总统计 总数、平均数、标准差、最大最小值和四分位数
df.dtypes # 查看各字段类型
df.axes # 显示数据行和列名
df.columns # 列名

#修改索引 现在索引是序号，但是有意义的索引是'name'
df.set_index('name', inplace=True) # 建立索引并生效
df.head() # 查看前 5 条，括号里可以传你想看的条数

#数据筛选
'总体筛选'
n = 5
df.head(n) # 查看 DataFrame 对象的前n行
df.tail(n) # 查看 DataFrame 对象的最后n行
df.sample(n) # 查看 n 个样本，随机
df.index # 查看索引内容
df.columns # 查看行名
'列选择'
df['Q1']
df.Q1 # 同上，如果列名符合 python 变量名要求，可使用
df[['team', 'Q1']] # 只看这两列，注意括号
df.loc[:, ['team', 'Q1']] # 和上边效果一样
#df.loc[x, y] 是一个非常强大的数据选择函数，其中 x 代表行，y 代表列，行和列都支持条件表达式，也支持类似列表那样的切片（如果要用自然索引需要用 df.iloc[]）。
'行选择'
# 用人工索引选取
df[df.index == 'Liver'] # 指定索引
# 用自然索引选择，类似列表的切片
df[0:3] # 取前三行,
df[0:10:2] # 前10个，每两个取一个
df.iloc[:10,:] # 前10个
df.loc['Ben', 'Q1':'Q4'] # 只看 Ben 的四个季度成绩
df.loc['Eorge':'Alexander', 'team':'Q4'] # 指定行区间
'条件选择'
df[df.Q1 > 90] # Q1 列大于90的
df[df.team == 'C'] # team 列为 'C' 的
df[df.index == 'Oscar'] # 指定索引即原数据中的 name
'排序'
df.sort_values(by='Q1') # 按 Q1 列数据升序排列
df.sort_values(by='Q1', ascending=False) # 降序
df.sort_values(['team', 'Q1'], ascending=[True, False]) # team 升，Q1 降序
# 组合条件
df[(df['Q1'] > 90) & (df['team'] == 'C')] # and 关系
df[df['team'] == 'C'].loc[df.Q1>90] # 多重筛选
'分组聚合'
df.groupby('team').sum() # 按团队分组对应列相加
df.groupby('team').mean() # 按团队分组对应列求平均
# 不同列不同的计算方法
df.groupby('team').agg({'Q1': sum,  # 总和
                        'Q2': 'count', # 总数
                        'Q3':'mean', # 平均
                        'Q4': max}) # 最大值
'数据转换（矩阵转换）'
df.groupby('team').sum().T
df.groupby('team').sum().stack()
df.groupby('team').sum().unstack()
'增加列'
df['one'] = 1 # 增加一个固定值的列
df['total'] = df.Q1 + df.Q2 + df.Q3 + df.Q4 # 增加总成绩列
# 指定一些列相加增加一个新列
df['total'] = df.loc[:,'Q1':'Q4'].apply(lambda x:sum(x), axis=1)
df['total'] = df.sum(axis=1) # 可以把所有为数字的列相加 （axis=0 表示行和）
df['avg'] = df.total/4 # 增加平均成绩列

#简单分析
df.mean() # 返回所有列的均值
df.mean(1) # 返回所有行的均值，下同
df.corr() # 返回列与列之间的相关系数
df.count() # 返回每一列中的非空值的个数
df.max() # 返回每一列的最大值
df.min() # 返回每一列的最小值
df.median() # 返回每一列的中位数
df.std() # 返回每一列的标准差
df.var() # 方差
s.mode() # 众数

#画图
'Pandas 利用plot() 调用 matplotlib 快速绘制出数据可视化图形。注意，第一次使用 plot() 时可能需要执行两下才能显示图形。'
df['Q1'].plot() # Q1 成绩的折线分布
df.loc['Ben','Q1':'Q4'].plot() # ben 四个季度的成绩变化
df.loc[ 'Ben','Q1':'Q4'].plot.bar() # 柱状图
df.loc[ 'Ben','Q1':'Q4'].plot.barh() # 横向柱状图
df.groupby('team').sum().T.plot() # 各 Team 四个季度总成绩趋势
df.groupby('team').count().Q1.plot.pie() # 各组人数对比

#输出Excel和CSV
df.to_excel('team-done.xlsx') # 导出 excel
df.to_csv('team-done.csv') # 导出 csv



