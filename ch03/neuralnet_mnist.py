# coding: utf-8
import sys, os
sys.path.append('.') 

import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


# 函数 get_data，用于加载 MNIST 数据集并返回测试数据
def get_data():
    # 使用 load_mnist 函数加载 MNIST 数据集，并进行归一化和展平处理
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    # 返回测试集的特征和标签
    return x_test, t_test

# 函数 init_network，用于加载预训练的神经网络权重
def init_network():
    # 打开存储权重的文件，以二进制模式读取
    with open("ch03\\sample_weight.pkl", 'rb') as f:
        # 使用 pickle 模块加载权重数据
        network = pickle.load(f)
    # 返回加载的神经网络权重
    return network

# 函数 predict，用于根据输入数据和网络权重进行预测
def predict(network, x):
    # 从网络权重中提取各层的权重和偏置
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 计算第一层的激活值
    a1 = np.dot(x, W1) + b1
    # 对第一层的激活值应用 sigmoid 函数
    z1 = sigmoid(a1)
    # 计算第二层的激活值
    a2 = np.dot(z1, W2) + b2
    # 对第二层的激活值应用 sigmoid 函数
    z2 = sigmoid(a2)
    # 计算第三层的激活值
    a3 = np.dot(z2, W3) + b3
    # 对第三层的激活值应用 softmax 函数，得到最终的预测概率分布
    y = softmax(a3)

    # 返回预测结果
    return y

# 获取测试数据
x, t = get_data()
# 加载神经网络权重
network = init_network()
# 初始化准确率计数器
accuracy_cnt = 0

# 遍历测试数据中的每个样本
for i in range(len(x)):
    # 使用神经网络对当前样本进行预测
    y = predict(network, x[i])
    # 获取预测结果中概率最大的类别
    p = np.argmax(y)
    # 如果预测结果与真实标签一致，则增加准确率计数器
    if p == t[i]:
        accuracy_cnt += 1

# 计算并输出模型的准确率
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
