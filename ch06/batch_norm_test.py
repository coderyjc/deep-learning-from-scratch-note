# coding: utf-8
import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD

# 加载MNIST数据集，并进行归一化处理
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 只使用前1000个训练样本
x_train = x_train[:1000]
t_train = t_train[:1000]

# 设置最大训练轮数为20
max_epochs = 20
# 获取训练集的大小
train_size = x_train.shape[0]
# 设置批量大小为100
batch_size = 100
# 设置学习率为0.01
learning_rate = 0.01

# 定义训练函数，接受权重初始化标准差作为参数
def __train(weight_init_std):
    # 创建一个使用Batch Normalization的神经网络
    bn_network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10, 
                                    weight_init_std=weight_init_std, use_batchnorm=True)
    # 创建一个不使用Batch Normalization的神经网络
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10,
                                weight_init_std=weight_init_std)
    # 使用随机梯度下降优化器
    optimizer = SGD(lr=learning_rate)
    
    # 初始化训练准确率列表
    train_acc_list = []
    bn_train_acc_list = []
    
    # 计算每个epoch的迭代次数
    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0
    
    # 开始训练循环
    for i in range(1000000000):
        # 随机选择批量数据
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
    
        # 对每个网络进行梯度更新
        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)
    
        # 每个epoch结束后计算并记录准确率
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)
    
            # 打印当前epoch的准确率
            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))
    
            epoch_cnt += 1
            # 如果达到最大epoch数，停止训练
            if epoch_cnt >= max_epochs:
                break
                
    # 返回训练准确率列表
    return train_acc_list, bn_train_acc_list

# 生成一个对数空间的权重初始化标准差列表
weight_scale_list = np.logspace(0, -4, num=16)
# 创建一个表示epoch的数组
x = np.arange(max_epochs)

# 对每个权重初始化标准差进行训练并绘制结果
for i, w in enumerate(weight_scale_list):
    print( "============== " + str(i+1) + "/16" + " ==============")
    train_acc_list, bn_train_acc_list = __train(w)
    
    # 创建子图
    plt.subplot(4,4,i+1)
    plt.title("W:" + str(w))
    if i == 15:
        # 在最后一个子图中绘制带标签的曲线
        plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
        plt.plot(x, train_acc_list, linestyle = "--", label='Normal(without BatchNorm)', markevery=2)
    else:
        # 在其他子图中绘制不带标签的曲线
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", markevery=2)

    # 设置y轴范围为0到1
    plt.ylim(0, 1.0)
    if i % 4:
        # 隐藏y轴刻度
        plt.yticks([])
    else:
        # 显示y轴标签
        plt.ylabel("accuracy")
    if i < 12:
        # 隐藏x轴刻度
        plt.xticks([])
    else:
        # 显示x轴标签
        plt.xlabel("epochs")
    # 添加图例
    plt.legend(loc='lower right')
    
# 显示所有子图
plt.show()
