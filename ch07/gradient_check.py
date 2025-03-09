# coding: utf-8
import numpy as np
from simple_convnet import SimpleConvNet


# 创建一个 SimpleConvNet 网络实例，指定输入维度、卷积参数、隐藏层大小、输出大小和权重初始化标准差
network = SimpleConvNet(input_dim=(1,10, 10), 
                        conv_param = {'filter_num':10, 'filter_size':3, 'pad':0, 'stride':1},
                        hidden_size=10, output_size=10, weight_init_std=0.01)

# 生成一个随机输入数据 X，形状为 (1, 1, 10, 10)
X = np.random.rand(100).reshape((1, 1, 10, 10))

# 生成一个目标标签 T，形状为 (1, 1)
T = np.array([1]).reshape((1,1))

# 使用数值梯度方法计算网络参数的梯度
grad_num = network.numerical_gradient(X, T)

# 使用反向传播方法计算网络参数的梯度
grad = network.gradient(X, T)

# 遍历梯度字典，计算数值梯度和反向传播梯度的平均绝对误差，并输出结果
for key, val in grad_num.items():
    print(key, np.abs(grad_num[key] - grad[key]).mean())
