# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 加载MNIST数据集，并进行归一化和one-hot编码处理
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 初始化一个两层的神经网络，输入层大小为784，隐藏层大小为50，输出层大小为10
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 设置训练的总迭代次数为10000次
iters_num = 10000

# 获取训练数据集的大小
train_size = x_train.shape[0]

# 设置每次训练的批量大小为100
batch_size = 100

# 设置学习率为0.1
learning_rate = 0.1

# 初始化用于存储训练损失、训练准确率和测试准确率的列表
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 计算每个epoch的迭代次数
iter_per_epoch = max(train_size / batch_size, 1)

# 开始训练循环，共进行iters_num次迭代
for i in range(iters_num):
    # 从训练数据中随机选择batch_size个样本
    batch_mask = np.random.choice(train_size, batch_size)
    
    # 获取选中的训练数据和对应的标签
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
   
    # 计算当前批量的梯度
    grad = network.gradient(x_batch, t_batch)
   
    # 更新网络的参数（权重和偏置）
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 计算当前批量的损失，并将其添加到训练损失列表中
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 如果当前迭代次数是每个epoch的整数倍，计算并记录训练和测试准确率
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        # 打印当前的训练准确率和测试准确率
        print(train_acc, test_acc)
