# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:03:25 2020
    预创建的 Estimators
    Estimator 是 Tensorflow 完整模型的高级表示，它被设计用于轻松扩展和异步训练。更多细节请参阅 Estimators。
    是一种更高级的封装
    
    数据：
    本文档中的示例程序构建并测试了一个模型，该模型根据花萼和花瓣的大小将鸢尾花分成三种物种。
    您将使用鸢尾花数据集训练模型。该数据集包括四个特征和一个标签。这四个特征确定了单个鸢尾花的以下植物学特征
        花萼长度
        花萼宽度
        花瓣长度
        花瓣宽度
@author: 67443
"""
import tensorflow as tf
import pandas as pd
import numpy as np

#常量数据
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

#数据下载
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

#分离出标签数据
train_y = train.pop('Species')
test_y = test.pop('Species')

# 标签列现已从数据中删除
train.head()

'''
Estimator 编程概述:
    现在您已经设定好了数据，您可以使用 Tensorflow Estimator 定义模型。Estimator 是从 tf.estimator.Estimator 中派生的任何类。
    Tensorflow提供了一组tf.estimator(例如，LinearRegressor)来实现常见的机器学习算法。此外，您可以编写您自己的自定义 Estimator。
    入门阶段我们建议使用预创建的 Estimator。
    
    为了编写基于预创建的 Estimator 的 Tensorflow 项目，您必须完成以下工作：
        创建一个或多个输入函数
        定义模型的特征列
        实例化一个 Estimator，指定特征列和各种超参数。
        在 Estimator 对象上调用一个或多个方法，传递合适的输入函数以作为数据源。

'''

'创建输入输出函数'
'''
输入函数是一个返回 tf.data.Dataset 对象的函数，此对象会输出下列含两个元素的元组
    features——Python字典，其中：
        每个键都是特征名称
        每个值都是包含此特征所有值的数组
    label 包含每个样本的标签的值的数组。
'''
#例1：
def input_evaluation_set():
    features = {'SepalLength': np.array([6.4, 5.0]),
                'SepalWidth':  np.array([2.8, 2.3]),
                'PetalLength': np.array([5.6, 3.3]),
                'PetalWidth':  np.array([2.2, 1.0])}
    labels = np.array([2, 1])
    return features, labels

#例2：
'''
    您的输入函数可以以您喜欢的方式生成 features 字典与 label 列表。
    但是，我们建议使用 Tensorflow 的 Dataset API，该 API 可以用来解析各种类型的数据。
    
    Dataset API 可以为您处理很多常见情况。例如，使用 Dataset API，您可以轻松地从大量文件中并行读取记录，
    并将它们合并为单个数据流。
    
    为了简化此示例，我们将使用 pandas 加载数据，并利用此内存数据构建输入管道。
'''
def input_fn(features, labels, training=True, batch_size=256):
    """An input function for training or evaluating"""
    # 将输入转换为数据集。
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # 如果在训练模式下混淆并重复数据。
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

'定义特征列（feature columns）'
'''
特征列（feature columns）是一个对象，用于描述模型应该如何使用特征字典中的原始输入数据。
当您构建一个 Estimator 模型的时候，您会向其传递一个特征列的列表，其中包含您希望模型使用的每个特征。
tf.feature_column 模块提供了许多为模型表示数据的选项。

对于鸢尾花问题，4 个原始特征是数值，因此我们将构建一个特征列的列表，以告知 Estimator 模型将 4 个特征都表示为 32 位浮点值。
故创建特征列的代码如下所示：
'''
# 特征列描述了如何使用输入。
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    
'实例化 Estimator'
'''
鸢尾花为题是一个经典的分类问题。幸运的是，Tensorflow 提供了几个预创建的 Estimator 分类器，其中包括：
    tf.estimator.DNNClassifier 用于多类别分类的深度模型
    tf.estimator.DNNLinearCombinedClassifier 用于广度与深度模型
    tf.estimator.LinearClassifier 用于基于线性模型的分类器
'''
# 构建一个拥有两个隐层，隐藏节点分别为 30 和 10 的深度神经网络。
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # 隐层所含结点数量分别为 30 和 10.
    hidden_units=[30, 10],
    # 模型必须从三个类别中做出选择。
    n_classes=3)


# 训练模型。
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)

#评估模型
eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

#使用训练好的模型
# 由模型生成预测
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}

def input_fn(features, batch_size=256):
    """An input function for prediction."""
    # 将输入转换为无标签数据集。
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

predictions = classifier.predict(
    input_fn = lambda: input_fn(predict_x))

#predict 方法返回一个 Python 可迭代对象，为每个样本生成一个预测结果字典。以下代码输出了一些预测及其概率：
for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        SPECIES[class_id], 100 * probability, expec))






