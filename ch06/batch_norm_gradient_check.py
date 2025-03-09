# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend

# 使用 load_mnist 函数加载 MNIST 数据集，并对其进行归一化和 one-hot 编码
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 创建一个多层神经网络实例，输入大小为 784，隐藏层大小为 [100, 100]，输出大小为 10，并使用批量归一化
network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100], output_size=10,
                              use_batchnorm=True)

# 从训练数据中取第一个样本作为批处理数据
x_batch = x_train[:1]
t_batch = t_train[:1]

# 使用反向传播计算梯度
grad_backprop = network.gradient(x_batch, t_batch)

# 使用数值方法计算梯度
grad_numerical = network.numerical_gradient(x_batch, t_batch)

# 遍历所有梯度键值，计算并打印反向传播梯度和数值梯度的差异
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))
