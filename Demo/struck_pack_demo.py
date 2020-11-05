# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 20:06:24 2020
python只有6中数据结构就可以完成大部分工作：
字符串，整数，浮点数，元组（set），列表（array），字典（key/value）

但是在和其他平台数据交互过程中，需要对其他平台数据类型进行翻译 ，这时候用struct模块进行处理。
Pack/unpack 二进制字节流和python可读类型的互相转换。
@author: 67443
"""

import struct
import binascii
import ctypes

#struct的格式必须用utf-8的格式
values = (1, ('abc').encode('utf-8'), 2.7)
s = struct.Struct('I3sf')
packed_data = s.pack(*values)
unpacked_data = s.unpack(packed_data)
	 
print ('Original values:', values)
print ('Format string :', s.format)
print ('Uses :', s.size, 'bytes')
print ('Packed Value :', binascii.hexlify(packed_data))
print ('Unpacked Type :', type(unpacked_data), ' Value:', unpacked_data)

values1 = (1, ('abc').encode('utf-8'), 2.7)
values2 = (('defg').encode('utf-8'),101)
s1 = struct.Struct('I3sf')
s2 = struct.Struct('4sI')
	 
prebuffer = ctypes.create_string_buffer(s1.size+s2.size) #calcsize(fmt)长度和
print ('Before :',binascii.hexlify(prebuffer))
struct.pack_into('I3sf',prebuffer,0,*values1)
s2.pack_into(prebuffer,s1.size,*values2)
print ('After pack:',binascii.hexlify(prebuffer))
print (s1.unpack_from(prebuffer,0))
print (s2.unpack_from(prebuffer,s1.size))
