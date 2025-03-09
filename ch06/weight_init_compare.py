# coding: utf-8
import os
import sys

sys.path.append(os.pardir) 
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

# 0:
# 加载MNIST数据集，并进行归一化处理
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 获取训练集的大小
train_size = x_train.shape[0]

# 设置批量大小和最大迭代次数
batch_size = 128
max_iterations = 2000


# 1
# 定义不同权重初始化方法的参数
weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}

# 使用随机梯度下降（SGD）优化器，学习率为0.01
optimizer = SGD(lr=0.01)

# 初始化网络和训练损失字典
networks = {}
train_loss = {}

# 为每种权重初始化方法创建多层神经网络
for key, weight_type in weight_init_types.items():
    networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100],
                                  output_size=10, weight_init_std=weight_type)
    train_loss[key] = []


# 2
# 开始训练，进行最大迭代次数的循环
for i in range(max_iterations):
    # 随机选择一批数据进行训练
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 对每种权重初始化方法的网络进行训练
    for key in weight_init_types.keys():
        # 计算梯度
        grads = networks[key].gradient(x_batch, t_batch)
        # 更新网络参数
        optimizer.update(networks[key].params, grads)
    
        # 计算损失并记录
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
    
    # 每100次迭代打印一次损失
    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in weight_init_types.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


# 3
# 定义不同权重初始化方法的绘图标记
markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}

# 创建x轴数据，表示迭代次数
x = np.arange(max_iterations)

# 绘制每种权重初始化方法的损失曲线
for key in weight_init_types.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)

# 设置x轴和y轴标签
plt.xlabel("iterations")
plt.ylabel("loss")

# 设置y轴范围
plt.ylim(0, 2.5)

# 显示图例
plt.legend()

# 显示图形
plt.show()
