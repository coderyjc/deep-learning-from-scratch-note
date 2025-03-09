# coding: utf-8
import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from deep_convnet import DeepConvNet
from dataset.mnist import load_mnist

# 加载MNIST数据集，flatten=False表示不将图像展平为一维数组
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 创建一个DeepConvNet神经网络实例
network = DeepConvNet()

# 加载预训练的网络参数
network.load_params("ch08\\deep_convnet_params.pkl")

# 设置采样数量为10000
sampled = 10000

# 对测试集进行采样，只取前10000个样本
x_test = x_test[:sampled]
t_test = t_test[:sampled]

# 打印信息，表示开始计算float64精度下的准确率
print("caluculate accuracy (float64) ... ")

# 计算并打印float64精度下的准确率
print(network.accuracy(x_test, t_test))

# 将测试集数据转换为float16精度
x_test = x_test.astype(np.float16)

# 将网络中的所有参数转换为float16精度
for param in network.params.values():
    param[...] = param.astype(np.float16)

# 打印信息，表示开始计算float16精度下的准确率
print("caluculate accuracy (float16) ... ")

# 计算并打印float16精度下的准确率
print(network.accuracy(x_test, t_test))
