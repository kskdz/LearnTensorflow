# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 15:07:59 2020
     tf.estimator API的决策树来训练梯度提升模型的端到端演示。提升树（Boosted Trees）模型是回归和分类问题中最受欢迎并最有效的机器学习方法之一。
     这是一种融合技术，它结合了几个（10 个，100 个或者甚至 1000 个）树模型的预测值。
     
     提升树（Boosted Trees）模型受到许多机器学习从业者的欢迎，因为它们可以通过最小化的超参数调整获得令人印象深刻的性能。
@author: 67443
"""
import numpy as np
import pandas as pd
from IPython.display import clear_output
from matplotlib import pyplot as plt

# 加载数据集。
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

import tensorflow as tf
tf.random.set_seed(123)

#创建特征列与输入函数
'''
梯度提升（Gradient Boosting） Estimator 可以利用数值和分类特征。
特征列适用于所有的 Tensorflow estimator，其目的是定义用于建模的特征。
此外，它们还提供一些特征工程功能，如独热编码（one-hot-encoding）、标准化（normalization）和桶化（bucketization）。
在本教程中，CATEGORICAL_COLUMNS 中的字段从分类列转换为独热编码列(指标列)：
'''
fc = tf.feature_column
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

def one_hot_cat_column(feature_name, vocab):
  return tf.feature_column.indicator_column(
      tf.feature_column.categorical_column_with_vocabulary_list(feature_name,
                                                 vocab))
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  # Need to one-hot encode categorical features.
  vocabulary = dftrain[feature_name].unique()
  feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name,
                                           dtype=tf.float32))

#查看列转换效果：
example = dict(dftrain.head(1))
class_fc = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('class', ('First', 'Second', 'Third')))
print('Feature value: "{}"'.format(example['class'].iloc[0]))
print('One-hot encoded: ', tf.keras.layers.DenseFeatures([class_fc])(example).numpy())
#查看所有特征列的转换
tf.keras.layers.DenseFeatures(feature_columns)(example).numpy()

'创建输入模型'
# 使用大小为全部数据的 batch ，因为数据规模非常小.
NUM_EXAMPLES = len(y_train)

def make_input_fn(X, y, n_epochs=None, shuffle=True):
  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    if shuffle:
      dataset = dataset.shuffle(NUM_EXAMPLES)
    # 对于训练，可以按需多次循环数据集（n_epochs=None）。
    dataset = dataset.repeat(n_epochs)
    # 在内存中训练不使用 batch。
    dataset = dataset.batch(NUM_EXAMPLES)
    return dataset
  return input_fn

# 训练与评估的输入函数。
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)

'训练和评估模型'
'''
初始化模型，指定特征和超参数。
使用 train_input_fn 将训练数据输入模型，使用 train 函数训练模型。
您将使用此示例中的评估集评估模型性能，即 dfeval DataFrame。您将验证预测是否与 y_eval 数组中的标签匹配。
'''
linear_est = tf.estimator.LinearClassifier(feature_columns)

# 训练模型。
linear_est.train(train_input_fn, max_steps=100)

# 评估。
result = linear_est.evaluate(eval_input_fn)
clear_output()
print(pd.Series(result))

'训练提升树模型'
'''
提升树（Boosted Trees）是支持回归（BoostedTreesRegressor）和分类（BoostedTreesClassifier）的。
由于目标是预测一个生存与否的标签，您将使用 BoostedTreesClassifier。
'''
# 由于数据存入内存中，在每层使用全部数据会更快。
# 上面一个 batch 定义为整个数据集。
n_batches = 1
est = tf.estimator.BoostedTreesClassifier(feature_columns,
                                          n_batches_per_layer=n_batches)

# 一旦建立了指定数量的树，模型将停止训练，
# 而不是基于训练步数。
est.train(train_input_fn, max_steps=100)

# 评估。
result = est.evaluate(eval_input_fn)
clear_output()
print(pd.Series(result))


#Tensorflow 模型经过优化，可以同时在一个 batch 或一个集合的样本上进行预测。
pred_dicts = list(est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')
plt.show()

#查看ROC曲线
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_eval, probs)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0,)
plt.ylim(0,)
plt.show()

'''
局部可解释性：
    局部可解释性指模型的预测在单一样例层面上的理解程度，而全局可解释性指模型作为一个整体的理解能力。
    这种技术可以帮助使用机器学习的人在模型开发阶段检测偏差（bias）和bug。
    
    对于局部可解释性，您将了解到如何创造并可视化每个实例（per-instance）的贡献度。
    区别于特征重要性，这种贡献被称为 DFCs（定向特征贡献，directional feature contributions）。

    对于全局可解释性，您将学习并可视化基于增益的特征重要性（gain-based feature importances），
    排列特征重要性（permutation feature importances）和总DFCs。
'''
'''
出于性能方面的原因，当您的数据是内存数据集时，我们推荐您使用 boosted_trees_classifier_train_in_memory 函数。
此外，如果您对训练时间没有要求抑或是您的数据集很大且不愿做分布式训练，请使用上面显示的 tf.estimator.BoostedTrees API。

当您使用此方法时，请不要对数据分批（batch），而是对整个数据集进行操作。
'''
params = {
  'n_trees': 50,
  'max_depth': 3,
  'n_batches_per_layer': 1,
  # 为了得到 DFCs，请设置 center_bias = True。这将强制
  # 模型在使用特征（例如：回归中训练集标签的均值，分类中使
  # 用交叉熵损失函数时的对数几率）前做一个初始预测。
  'center_bias': True
}

in_memory_params = dict(params)
in_memory_params['n_batches_per_layer'] = 1
# 在内存中的输入方程请不要对数据分批。
def make_inmemory_train_input_fn(X, y):
  y = np.expand_dims(y, axis=1)
  def input_fn():
    return dict(X), y
  return input_fn
train_input_fn = make_inmemory_train_input_fn(dftrain, y_train)

# 训练模型。
est = tf.estimator.BoostedTreesClassifier(
    feature_columns, 
    train_in_memory=True, 
    **in_memory_params)

est.train(train_input_fn)
print(est.evaluate(eval_input_fn))

'模型说明与绘制'
import matplotlib.pyplot as plt
import seaborn as sns
sns_colors = sns.color_palette('colorblind')

'局部可解释性（Local interpretability）'
'''
接下来，您将输出定向特征贡献（DFCs）来解释单个预测。输出依据 Palczewska et al 和 Saabas 在 解释随机森林（Interpreting Random Forests） 
中提出的方法产生(scikit-learn 中随机森林相关的包 treeinterpreter 使用原理相同的远离). 使用以下语句输出 DFCs:

pred_dicts = list(est.experimental_predict_with_explanations(pred_input_fn))

（注意：带 “experimental” 前缀为实验版本（开发中），在正式版发布前可能对其修改。）
'''
pred_dicts = list(est.experimental_predict_with_explanations(eval_input_fn))

# 创建 DFCs 的 DataFrame。
labels = y_eval.values
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
df_dfc = pd.DataFrame([pred['dfc'] for pred in pred_dicts])
df_dfc.describe().T

# DFCs的和 + 偏差（bias） == 可能性
bias = pred_dicts[0]['bias']
dfc_prob = df_dfc.sum(axis=1) + bias
np.testing.assert_almost_equal(dfc_prob.values,
                               probs.values)
#为单个乘客绘制 DFCs，绘图时按贡献的方向性对其进行涂色并添加特征的值。
# 绘制模版 :)
def _get_color(value):
    """正的 DFCs 标为绿色，负的为红色。"""
    green, red = sns.color_palette()[2:4]
    if value >= 0: return green
    return red

def _add_feature_values(feature_values, ax):
    """在图的左侧显示特征的值"""
    x_coord = ax.get_xlim()[0]
    OFFSET = 0.15
    for y_coord, (feat_name, feat_val) in enumerate(feature_values.items()):
        t = plt.text(x_coord, y_coord - OFFSET, '{}'.format(feat_val), size=12)
        t.set_bbox(dict(facecolor='white', alpha=0.5))
    from matplotlib.font_manager import FontProperties
    font = FontProperties()
    font.set_weight('bold')
    t = plt.text(x_coord, y_coord + 1 - OFFSET, 'feature\nvalue',
    fontproperties=font, size=12)

def plot_example(example):
  TOP_N = 8 # 显示前8个特征。
  sorted_ix = example.abs().sort_values()[-TOP_N:].index  # 按值排序。
  example = example[sorted_ix]
  colors = example.map(_get_color).tolist()
  ax = example.to_frame().plot(kind='barh',
                          color=[colors],
                          legend=None,
                          alpha=0.75,
                          figsize=(10,6))
  ax.grid(False, axis='y')
  ax.set_yticklabels(ax.get_yticklabels(), size=14)

  # 添加特征的值。
  _add_feature_values(dfeval.iloc[ID][sorted_ix], ax)
  return ax

# 绘制结果。
ID = 182
example = df_dfc.iloc[ID]  # 从评估集中选择第 i 个样例。
TOP_N = 8  # 显示前8个特征。
sorted_ix = example.abs().sort_values()[-TOP_N:].index
ax = plot_example(example)
ax.set_title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(ID, probs[ID], labels[ID]))
ax.set_xlabel('Contribution to predicted probability', size=14)
plt.show()

'''
更大的贡献值意味着对模型的预测有更大的影响。
负的贡献表示此样例该特征的值减小了减小了模型的预测，正贡献值表示增加了模型的预测。

您也可以使用小提琴图（violin plot）来绘制该样例的 DFCs 并与整体分布比较。
'''
# 绘制代码模版。
def dist_violin_plot(df_dfc, ID):
  # 初始化画布。
  fig, ax = plt.subplots(1, 1, figsize=(10, 6))

  # 创建样例 DataFrame。
  TOP_N = 8  # 显示前8个特征。
  example = df_dfc.iloc[ID]
  ix = example.abs().sort_values()[-TOP_N:].index
  example = example[ix]
  example_df = example.to_frame(name='dfc')

  # 添加整个分布的贡献。
  parts=ax.violinplot([df_dfc[w] for w in ix],
                 vert=False,
                 showextrema=False,
                 widths=0.7,
                 positions=np.arange(len(ix)))
  face_color = sns_colors[0]
  alpha = 0.15
  for pc in parts['bodies']:
      pc.set_facecolor(face_color)
      pc.set_alpha(alpha)

  # 添加特征的值。
  _add_feature_values(dfeval.iloc[ID][sorted_ix], ax)

  # 添加局部贡献。
  ax.scatter(example,
              np.arange(example.shape[0]),
              color=sns.color_palette()[2],
              s=100,
              marker="s",
              label='contributions for example')

  # 图例。
  # 生成小提琴图的详细图例。
  ax.plot([0,0], [1,1], label='eval set contributions\ndistributions',
          color=face_color, alpha=alpha, linewidth=10)
  legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large',
                     frameon=True)
  legend.get_frame().set_facecolor('white')

  # 调整格式。
  ax.set_yticks(np.arange(example.shape[0]))
  ax.set_yticklabels(example.index)
  ax.grid(False, axis='y')
  ax.set_xlabel('Contribution to predicted probability', size=14)

#绘制样例
dist_violin_plot(df_dfc, ID)
plt.title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(ID, probs[ID], labels[ID]))
plt.show()

#最后，第三方的工具，如：LIME 和 shap 也可以帮助理解模型的各个预测。

'全局特征重要性（Global feature importances）'
'''
您或许想了解模型这个整体而不是单个预测。接下来，您将计算并使用：

通过 est.experimental_feature_importances 得到基于增益的特征重要性（Gain-based feature importances）
排列特征重要性（Permutation feature importances）
使用 est.experimental_predict_with_explanations 得到总 DFCs。
基于增益的特征重要性在分离特定特征时测量损失的变化。而排列特征重要性是在评估集上通过每次打乱一个特征后观察模型性能的变化计算而出。

一般来说，排列特征重要性要优于基于增益的特征重要性，尽管这两种方法在潜在预测变量的测量范围或类别数量不确定时和特征相关联时不可信（来源）。 对不同种类特征重要性的更透彻概括和更翔实讨论请参考 这篇文章 。
'''
#基于增益的特征重要性（Gain-based feature importances）
#TensorFlow 的提升树估算器（estimator）内置了函数 est.experimental_feature_importances 用于计算基于增益的特征重要性。
importances = est.experimental_feature_importances(normalize=True)
df_imp = pd.Series(importances)

# 可视化重要性。
N = 8
ax = (df_imp.iloc[0:N][::-1]
    .plot(kind='barh',
          color=sns_colors[0],
          title='Gain feature importances',
          figsize=(10, 6)))
ax.grid(False, axis='y')

#平均绝对 DFCs
# 绘图。
dfc_mean = df_dfc.abs().mean()
N = 8
sorted_ix = dfc_mean.abs().sort_values()[-N:].index  # 求平均并按绝对值排序。
ax = dfc_mean[sorted_ix].plot(kind='barh',
                       color=sns_colors[1],
                       title='Mean |directional feature contributions|',
                       figsize=(10, 6))
ax.grid(False, axis='y')

FEATURE = 'fare'
feature = pd.Series(df_dfc[FEATURE].values, index=dfeval[FEATURE].values).sort_index()
ax = sns.regplot(feature.index.values, feature.values, lowess=True)
ax.set_ylabel('contribution')
ax.set_xlabel(FEATURE)
ax.set_xlim(0, 100)
plt.show()

#排列特征重要性（Permutation feature importances）
def permutation_importances(est, X_eval, y_eval, metric, features):
    """
    分别对每列，打散列中的值并观察其对评估集的影响。

    在训练过程中，有一种类似的方法，请参阅文章（来源：http://explained.ai/rf-importance/index.html）
    中有关 “Drop-column importance” 的部分。
    """
    baseline = metric(est, X_eval, y_eval)
    imp = []
    for col in features:
        save = X_eval[col].copy()
        X_eval[col] = np.random.permutation(X_eval[col])
        m = metric(est, X_eval, y_eval)
        X_eval[col] = save
        imp.append(baseline - m)
    return np.array(imp)

def accuracy_metric(est, X, y):
    """TensorFlow 估算器精度"""
    eval_input_fn = make_input_fn(X,
                                  y=y,
                                  shuffle=False,
                                  n_epochs=1)
    return est.evaluate(input_fn=eval_input_fn)['accuracy']
features = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
importances = permutation_importances(est, dfeval, y_eval, accuracy_metric,
                                      features)
df_imp = pd.Series(importances, index=features)

sorted_ix = df_imp.abs().sort_values().index
ax = df_imp[sorted_ix][-5:].plot(kind='barh', color=sns_colors[2], figsize=(10, 6))
ax.grid(False, axis='y')
ax.set_title('Permutation feature importance')
plt.show()


