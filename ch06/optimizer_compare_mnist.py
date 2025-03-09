# coding: utf-8
import sys
sys.path.append('.') 

import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *

# 加载MNIST数据集，normalize=True表示对数据进行归一化处理
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 获取训练数据的总样本数
train_size = x_train.shape[0]

# 设置每次训练的批量大小为128
batch_size = 128

# 设置最大迭代次数为2000
max_iterations = 2000

# 创建一个字典来存储不同的优化器
optimizers = {}

# 初始化SGD优化器
optimizers['SGD'] = SGD()

# 初始化Momentum优化器
optimizers['Momentum'] = Momentum()

# 初始化AdaGrad优化器
optimizers['AdaGrad'] = AdaGrad()

# 初始化Adam优化器
optimizers['Adam'] = Adam()

# 创建一个字典来存储不同的神经网络
networks = {}

# 创建一个字典来存储每个优化器对应的训练损失
train_loss = {}

# 为每个优化器初始化一个多层神经网络，并初始化对应的训练损失列表
for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10)
    train_loss[key] = []

# 开始训练过程，迭代max_iterations次
for i in range(max_iterations):
    # 从训练数据中随机选择batch_size个样本
    batch_mask = np.random.choice(train_size, batch_size)
    
    # 获取对应的输入数据和标签
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 对每个优化器进行参数更新
    for key in optimizers.keys():
        # 计算当前网络的梯度
        grads = networks[key].gradient(x_batch, t_batch)
        
        # 使用优化器更新网络参数
        optimizers[key].update(networks[key].params, grads)
        
        # 计算当前网络的损失
        loss = networks[key].loss(x_batch, t_batch)
        
        # 将损失值添加到对应的训练损失列表中
        train_loss[key].append(loss)
    
    # 每100次迭代打印一次损失值
    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))

# 设置不同优化器的绘图标记
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}

# 创建一个x轴数据，表示迭代次数
x = np.arange(max_iterations)

# 绘制每个优化器的损失曲线
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)

# 设置x轴标签为"iterations"
plt.xlabel("iterations")

# 设置y轴标签为"loss"
plt.ylabel("loss")

# 设置y轴范围为0到1
plt.ylim(0, 1)

# 显示图例
plt.legend()

# 显示图形
plt.show()
