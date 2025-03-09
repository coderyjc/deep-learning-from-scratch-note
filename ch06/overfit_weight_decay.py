# coding: utf-8
import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

# 加载MNIST数据集，并将数据分为训练集和测试集
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 仅使用前300个训练样本和标签
x_train = x_train[:300]
t_train = t_train[:300]

# 设置权重衰减参数
weight_decay_lambda = 0.1

# 初始化一个多层神经网络，输入大小为784，隐藏层大小为6层100个神经元，输出大小为10
network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                        weight_decay_lambda=weight_decay_lambda)

# 使用随机梯度下降（SGD）作为优化器，学习率为0.01
optimizer = SGD(lr=0.01)

# 设置最大训练轮数为201
max_epochs = 201

# 获取训练集的大小
train_size = x_train.shape[0]

# 设置每个批次的样本数为100
batch_size = 100

# 初始化用于存储训练损失、训练准确率和测试准确率的列表
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 计算每个epoch的迭代次数
iter_per_epoch = max(train_size / batch_size, 1)

# 初始化epoch计数器
epoch_cnt = 0

# 开始训练过程
for i in range(1000000000):
    # 从训练集中随机选择一批样本
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grads = network.gradient(x_batch, t_batch)

    # 更新网络参数
    optimizer.update(network.params, grads)

    # 如果当前迭代次数是epoch的整数倍，计算并记录准确率
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        # 打印当前epoch的训练准确率和测试准确率
        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        # 增加epoch计数器
        epoch_cnt += 1

        # 如果达到最大epoch数，结束训练
        if epoch_cnt >= max_epochs:
            break

# 绘制训练和测试准确率的曲线
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
