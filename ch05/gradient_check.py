# coding: utf-8
import sys, os
sys.path.append(os.pardir) 
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 使用 load_mnist 函数加载 MNIST 数据集，并进行归一化和 one-hot 编码
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 创建一个具有 784 个输入节点、50 个隐藏节点和 10 个输出节点的两层神经网络
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 从训练数据中取出前 3 个样本作为小批量数据
x_batch = x_train[:3]
t_batch = t_train[:3]

# 使用数值梯度计算方法计算小批量数据的梯度
grad_numerical = network.numerical_gradient(x_batch, t_batch)

# 使用反向传播算法计算小批量数据的梯度
grad_backprop = network.gradient(x_batch, t_batch)

# 遍历每个参数的梯度，计算数值梯度和反向传播梯度之间的差异
for key in grad_numerical.keys():
    # 计算两个梯度之间的平均绝对差异
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    # 输出参数名称及其对应的梯度差异
    print(key + ":" + str(diff))
