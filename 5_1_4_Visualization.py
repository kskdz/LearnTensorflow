# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 16:24:37 2020
    可视化的数据拟合
@author: 67443
"""
import numpy as np
import pandas as pd
from numpy.random import uniform, seed
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import tensorflow as tf
from IPython.display import clear_output

# 生成数据。
seed(0)
npts = 5000
x = uniform(-2, 2, npts)
y = uniform(-2, 2, npts)
z = x*np.exp(-x**2 - y**2)
xy = np.zeros((2,np.size(x)))
xy[0] = x
xy[1] = y
xy = xy.T

# 准备用于训练的数据。
df = pd.DataFrame({'x': x, 'y': y, 'z': z})

xi = np.linspace(-2.0, 2.0, 200),
yi = np.linspace(-2.1, 2.1, 210),
xi,yi = np.meshgrid(xi, yi)

df_predict = pd.DataFrame({
    'x' : xi.flatten(),
    'y' : yi.flatten(),
})
predict_shape = xi.shape

def plot_contour(x, y, z, **kwargs):
  # 准备用于训练的数据。
  plt.figure(figsize=(10, 8))
  # 绘制等值线图，标出非均匀数据点。
  CS = plt.contour(x, y, z, 15, linewidths=0.5, colors='k')
  CS = plt.contourf(x, y, z, 15,
                    vmax=abs(zi).max(), vmin=-abs(zi).max(), cmap='RdBu_r')
  plt.colorbar()  # 绘制颜色图例。
  # 绘制数据点。
  plt.xlim(-2, 2)
  plt.ylim(-2, 2)

#可视化这个方程 红色代表较大数值
zi = griddata(xy, z, (xi, yi), method='linear', fill_value='0')
plot_contour(xi, yi, zi)
plt.scatter(df.x, df.y, marker='.')
plt.title('Contour on training data')
plt.show()

'构建特征和输入函数'
fc = [tf.feature_column.numeric_column('x'),
      tf.feature_column.numeric_column('y')]

'构建输入函数'
# 当数据集小的时候，将整个数据集作为一个 batch。
NUM_EXAMPLES = 627

def make_input_fn(X, y, n_epochs=None, shuffle=True):
  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X.to_dict(orient='list'), y))
    if shuffle:
      dataset = dataset.shuffle(NUM_EXAMPLES)
    # 训练时让数据迭代尽可能多次 （n_epochs=None）。
    dataset = (dataset
      .repeat(n_epochs)
      .batch(NUM_EXAMPLES))
    return dataset
  return input_fn


def predict(est):
  """已有估算器给出的预测"""
  predict_input_fn = lambda: tf.data.Dataset.from_tensors(dict(df_predict))
  preds = np.array([p['predictions'][0] for p in est.predict(predict_input_fn)])
  return preds.reshape(predict_shape)

'首先尝试用线性模型拟合数据'
train_input_fn = make_input_fn(df, df.z)
est = tf.estimator.LinearRegressor(fc)
est.train(train_input_fn, max_steps=500);
  
plot_contour(xi, yi, predict(est))

'下面使用提升树拟合数据'
n_trees = 10
#在n=10的情况下可以运行 20之后内核会崩溃

est = tf.estimator.BoostedTreesRegressor(fc, n_batches_per_layer=1, n_trees=n_trees)
est.train(train_input_fn, max_steps=500)
clear_output()
plot_contour(xi, yi, predict(est))
plt.text(-1.8, 2.1, '# trees: {}'.format(n_trees), color='w', backgroundcolor='black', size=20)
plt.show()

'''
本文介绍了如何使用定向特征贡献（DFCs）及几种特征重要性来解释提升树模型。
这些方法可以帮助您了解特征是如何影响模型的预测。 
最后，您还可以通过观察其他模型的超平面（decision surface）并结合本文内容来学习提升树模型是如何拟合方程的。
'''