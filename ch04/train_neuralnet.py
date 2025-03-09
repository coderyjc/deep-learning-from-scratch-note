# coding: utf-8
import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 加载MNIST数据集，并进行归一化和one-hot编码
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 创建一个两层的神经网络，输入层大小为784，隐藏层大小为50，输出层大小为10
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 设置训练的总迭代次数为10000次
iters_num = 10000 

# 获取训练数据的总样本数
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

# 开始训练循环
for i in range(iters_num):
    # 从训练数据中随机选择一批样本
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 计算当前批次的梯度
    grad = network.gradient(x_batch, t_batch)
   
    # 更新网络的参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 计算当前批次的损失并记录
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 每隔一个epoch计算并记录训练和测试的准确率
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 绘制训练和测试准确率的图表
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
