# coding: utf-8
import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

# 加载MNIST数据集，并进行归一化处理
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 仅使用训练数据的前300个样本
x_train = x_train[:300]
t_train = t_train[:300]

# 设置是否使用Dropout以及Dropout的比例
use_dropout = True 
dropout_ratio = 0.2

# 创建一个多层神经网络，输入大小为784，隐藏层大小为6层每层100个神经元，输出大小为10
network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                              output_size=10, use_dropout=use_dropout, dropout_ration=dropout_ratio)

# 创建训练器，使用SGD优化器，学习率为0.01，训练301个epoch，mini_batch大小为100
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=301, mini_batch_size=100,
                  optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=True)

# 开始训练神经网络
trainer.train()

# 获取训练和测试的准确率列表
train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

# 绘制训练和测试准确率的图表
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
