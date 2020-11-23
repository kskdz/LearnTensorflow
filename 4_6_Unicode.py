# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 16:54:21 2020
    处理自然语言的模型通常使用不同的字符集来处理不同的语言。Unicode 是一种标准的编码系统，用于表示几乎所有语言的字符。每个字符使用 0 和 0x10FFFF 之间的唯一整数码位进行编码。Unicode 字符串是由零个或更多码位组成的序列。
    本教程介绍了如何在 TensorFlow 中表示 Unicode 字符串，以及如何使用标准字符串运算的 Unicode 等效项对其进行操作。它会根据字符体系检测将 Unicode 字符串划分为不同词例。
@author: 67443
"""

import tensorflow as tf

'您可以使用基本的 TensorFlow tf.string dtype 构建字节字符串张量。Unicode 字符串默认使用 UTF-8 编码。'
tf.constant(u"Thanks 😊")
'tf.string 张量可以容纳不同长度的字节字符串，因为字节字符串会被视为原子单元。字符串长度不包括在张量维度中。'
tf.constant([u"You're", u"welcome!"]).shape
'在 v3 中，字符串默认使用 Unicode 编码。'

'Unicode表示'
'三种表示方法'
'''
两种表示方法：
    string 标量 - 使用已知字符编码对码位序列进行编码。
    <tf.Tensor: shape=(), dtype=string, numpy=b'\xe8\xaf\xad\xe8\xa8\x80\xe5\xa4\x84\xe7\x90\x86'>
    int32 向量 - 每个位置包含单个码位。
    <tf.Tensor: shape=(4,), dtype=int32, numpy=array([35821, 35328, 22788, 29702], dtype=int32)>
'''
# Unicode string, represented as a UTF-8 encoded string scalar.
text_utf8 = tf.constant(u"语言处理")  #创建函数
text_utf8

# Unicode string, represented as a UTF-16-BE encoded string scalar.
text_utf16be = tf.constant(u"语言处理".encode("UTF-16-BE"))
text_utf16be

# Unicode string, represented as a vector of Unicode code points.
text_chars = tf.constant([ord(char) for char in u"语言处理"])
text_chars  #ord() 返回对应的 ASCII 数值，或者 Unicode 数值


'不同类型转换'
'向量标量转化；不同编码方式转化'
'tf.strings.unicode_decode：将编码的字符串标量转换为码位的向量。'
tf.strings.unicode_decode(text_utf8,
                          input_encoding='UTF-8')

'tf.strings.unicode_encode：将码位的向量转换为编码的字符串标量。'
tf.strings.unicode_encode(text_chars,
                          output_encoding='UTF-8')

'tf.strings.unicode_transcode：将编码的字符串标量转换为其他编码。'
tf.strings.unicode_transcode(text_utf8,
                             input_encoding='UTF8',
                             output_encoding='UTF-16-BE')


'长度处理问题：长度不同的类型转化'
'批次维度：解码多个字符串时，每个字符串中的字符数可能不相等。返回结果是 tf.RaggedTensor 残差不起的产量'
# A batch of Unicode strings, each represented as a UTF8-encoded string.
batch_utf8 = [s.encode('UTF-8') for s in
              [u'hÃllo',  u'What is the weather tomorrow',  u'Göödnight', u'😊']]
batch_chars_ragged = tf.strings.unicode_decode(batch_utf8,
                                               input_encoding='UTF-8')
for sentence_chars in batch_chars_ragged.to_list():
  print(sentence_chars)
  
'也可以使用 tf.RaggedTensor.to_tensor 和 tf.RaggedTensor.to_sparse 方法将其转换为带有填充的密集 tf.Tensor 或 tf.SparseTensor'
batch_chars_padded = batch_chars_ragged.to_tensor(default_value=-1)
print(batch_chars_padded.numpy())

batch_chars_sparse = batch_chars_ragged.to_sparse()
#稀疏转化为[句子编号 字符号]

#等长字符转化
tf.strings.unicode_encode([[99, 97, 116], [100, 111, 103], [ 99, 111, 119]],
                          output_encoding='UTF-8')
#不等长字符转化
tf.strings.unicode_encode(batch_chars_ragged, output_encoding='UTF-8')

'如果您的张量具有填充或稀疏格式的多个字符串，请在调用 unicode_encode 之前将其转换为 tf.RaggedTensor'
tf.strings.unicode_encode(
    tf.RaggedTensor.from_sparse(batch_chars_sparse),
    output_encoding='UTF-8')

tf.strings.unicode_encode(
    tf.RaggedTensor.from_tensor(batch_chars_padded, padding=-1),
    output_encoding='UTF-8')

'Unicode 运算'
#长度计算
# Note that the final character takes up 4 bytes in UTF8.
thanks = u'Thanks 😊'.encode('UTF-8')
num_bytes = tf.strings.length(thanks).numpy()
num_chars = tf.strings.length(thanks, unit='UTF8_CHAR').numpy()
print('{} bytes; {} UTF-8 characters'.format(num_bytes, num_chars))

#子字符串
#tf.strings.substr 运算会接受 "unit" 参数，并用它来确定 "pos" 和 "len" 参数包含的偏移类型
# default: unit='BYTE'. With len=1, we return a single byte.
tf.strings.substr(thanks, pos=7, len=1).numpy()
# Specifying unit='UTF8_CHAR', we return a single character, which in this case
# is 4 bytes.
print(tf.strings.substr(thanks, pos=7, len=1, unit='UTF8_CHAR').numpy())

#拆分字符串
#tf.strings.unicode_split 运算会将 Unicode 字符串拆分为单个字符的子字符串
tf.strings.unicode_split(thanks, 'UTF-8').numpy()

#字符的字节偏移量
#为了将 tf.strings.unicode_decode 生成的字符张量与原始字符串对齐，了解每个字符开始位置的偏移量很有用。
#方法 tf.strings.unicode_decode_with_offsets 与 unicode_decode 类似，不同的是它会返回包含每个字符起始偏移量的第二张量。
codepoints, offsets = tf.strings.unicode_decode_with_offsets(u"🎈🎉🎊", 'UTF-8')

for (codepoint, offset) in zip(codepoints.numpy(), offsets.numpy()): #zip效果是打包为元组
  print("At byte offset {}: codepoint {}".format(offset, codepoint))


'Unicode 字符体系'
'''
每个 Unicode 码位都属于某个码位集合，这些集合被称作字符体系。某个字符的字符体系有助于确定该字符可能所属的语言。
例如，已知 'Б' 属于西里尔字符体系，表明包含该字符的现代文本很可能来自某个斯拉夫语种（如俄语或乌克兰语）。

TensorFlow 提供了 tf.strings.unicode_script 运算来确定某一给定码位使用的是哪个字符体系。
字符体系代码是对应于国际 Unicode 组件 (ICU) UScriptCode 值的 int32 值。
'''
uscript = tf.strings.unicode_script([33464, 1041])  # ['芸', 'Б']

print(uscript.numpy())  # [17, 8] == [USCRIPT_HAN, USCRIPT_CYRILLIC]

#tf.strings.unicode_script 运算还可以应用于码位的多维 tf.Tensor 或 tf.RaggedTensor
print(tf.strings.unicode_script(batch_chars_ragged))

'示例：简单分词'
'''
分词是将文本拆分为类似单词的单元的任务。当使用空格字符分隔单词时，这通常很容易，
但是某些语言（如中文和日语）不使用空格，而某些语言（如德语）中存在长复合词，必须进行拆分才能分析其含义。
在网页文本中，不同语言和字符体系常常混合在一起，例如“NY株価”（纽约证券交易所）。

我们可以利用字符体系的变化进行粗略分词（不实现任何 ML 模型），从而估算词边界。
这对类似上面“NY株価”示例的字符串都有效。这种方法对大多数使用空格的语言也都有效，
因为各种字符体系中的空格字符都归类为 USCRIPT_COMMON，这是一种特殊的字符体系代码，不同于任何实际文本。
'''
# dtype: string; shape: [num_sentences]
#
# The sentences to process.  Edit this line to try out different inputs!
sentence_texts = [u'Hello, world.', u'世界こんにちは']

#先编码字符：再找每个字符的标识体系
# dtype: int32; shape: [num_sentences, (num_chars_per_sentence)]
#
# sentence_char_codepoint[i, j] is the codepoint for the j'th character in
# the i'th sentence.
sentence_char_codepoint = tf.strings.unicode_decode(sentence_texts, 'UTF-8')
print(sentence_char_codepoint)

# dtype: int32; shape: [num_sentences, (num_chars_per_sentence)]
#
# sentence_char_scripts[i, j] is the unicode script of the j'th character in
# the i'th sentence.
sentence_char_script = tf.strings.unicode_script(sentence_char_codepoint)
print(sentence_char_script)

#添加词边界的位置
# dtype: bool; shape: [num_sentences, (num_chars_per_sentence)]
#
# sentence_char_starts_word[i, j] is True if the j'th character in the i'th
# sentence is the start of a word.
'''
函数说明：
    tf.concat 拼接张量的函数tf.concat()
         t1 = [[1, 2, 3], [4, 5, 6]]
         t2 = [[7, 8, 9], [10, 11, 12]]
         tf.concat([t1, t2], axis=0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
         tf.concat([t1, t2], axis=1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
         按照最少的一列链接
         
    tf.fill(dims, value, name=None)  创建一个维度为dims，值为value的tensor对象。
        a = tf.fill([10], 0)         #[0 0 0 0 0 0 0 0 0 0]
        b = tf.fill([2, 3, 4], 5)    #[[[5 5 5 5]  [5 5 5 5]  [5 5 5 5]] [[5 5 5 5]  [5 5 5 5]  [5 5 5 5]]]

    tf.not_equal(x,y,name = None)    返回 (x! = y) 元素的真值.
    tf.not_equal(sentence_char_script[:, 1:], sentence_char_script[:, :-1])]
    这个写法很厉害啊，列表中相邻两个元素是否相等
'''
sentence_char_starts_word = tf.concat(
    [tf.fill([sentence_char_script.nrows(), 1], True),
     tf.not_equal(sentence_char_script[:, 1:], sentence_char_script[:, :-1])],
    axis=1)

# dtype: int64; shape: [num_words]
#
# word_starts[i] is the index of the character that starts the i'th word (in
# the flattened list of characters from all sentences).
'''
    'tf.squeeze' 将原始input中所有维度为1的那些维都删掉 axis指定维度
    tf.where(sentence_char_starts_word.values) #array([[ 0], [ 5], [ 7], ... ,        [15]], dtype=int64)>
    tf.squeeze(tf.where(sentence_char_starts_word.values), axis=1)
    #<tf.Tensor: shape=(6,), dtype=int64, numpy=array([ 0,  5,  7, 12, 13, 15], dtype=int64)>
'''

word_starts = tf.squeeze(tf.where(sentence_char_starts_word.values), axis=1)
print(word_starts)

#然后构建 RaggedTensor
# dtype: int32; shape: [num_words, (num_chars_per_word)]
#
# word_char_codepoint[i, j] is the codepoint for the j'th character in the
# i'th word.
'tf.RaggedTensor.from_row_starts方法'
word_char_codepoint = tf.RaggedTensor.from_row_starts(
    values=sentence_char_codepoint.values,
    row_starts=word_starts)
print(word_char_codepoint)

#最后划分到句子中 词码位 RaggedTensor 划分回句子
# dtype: int64; shape: [num_sentences]
#
# sentence_num_words[i] is the number of words in the i'th sentence.
'tf.reduce_sum 用于计算张量tensor沿着某一维度的和，可以在求和后降维'
sentence_num_words = tf.reduce_sum(
    tf.cast(sentence_char_starts_word, tf.int64),  #数据类型转换
    axis=1)

# dtype: int32; shape: [num_sentences, (num_words_per_sentence), (num_chars_per_word)]
#
# sentence_word_char_codepoint[i, j, k] is the codepoint for the k'th character
# in the j'th word in the i'th sentence.
'tf.RaggedTensor.from_row_lengths'
sentence_word_char_codepoint = tf.RaggedTensor.from_row_lengths(
    values=word_char_codepoint,
    row_lengths=sentence_num_words)
print(sentence_word_char_codepoint)

#转化为UTF-8
tf.strings.unicode_encode(sentence_word_char_codepoint, 'UTF-8').to_list()

