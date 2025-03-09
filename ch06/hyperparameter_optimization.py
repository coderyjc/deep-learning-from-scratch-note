# coding: utf-8
import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer

# 加载MNIST数据集，并将数据归一化
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 仅使用前500个训练样本
x_train = x_train[:500]
t_train = t_train[:500]

# 设置验证集的比例为20%
validation_rate = 0.20
# 计算验证集的数量
validation_num = int(x_train.shape[0] * validation_rate)
# 打乱训练数据集
x_train, t_train = shuffle_dataset(x_train, t_train)
# 从训练集中分离出验证集
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
# 剩余的样本作为训练集
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]

# 定义训练函数，用于训练多层神经网络
def __train(lr, weight_decay, epocs=50):
    # 创建一个多层神经网络实例
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10, weight_decay_lambda=weight_decay)
    # 创建一个训练器实例
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs, mini_batch_size=100,
                      optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
    # 开始训练
    trainer.train()

    # 返回测试准确率和训练准确率列表
    return trainer.test_acc_list, trainer.train_acc_list

# 设置优化试验的次数为100
optimization_trial = 100
# 用于存储验证集和训练集的准确率结果
results_val = {}
results_train = {}
# 进行优化试验
for _ in range(optimization_trial):
   
    # 随机生成权重衰减参数
    weight_decay = 10 ** np.random.uniform(-8, -4)
    # 随机生成学习率
    lr = 10 ** np.random.uniform(-6, -2)
   
    # 使用随机生成的参数进行训练
    val_acc_list, train_acc_list = __train(lr, weight_decay)
    # 打印验证集准确率、学习率和权重衰减参数
    print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
    # 生成一个唯一的键值
    key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
    # 存储验证集和训练集的准确率结果
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

# 打印超参数优化结果
print("=========== Hyper-Parameter Optimization Result ===========")
# 设置绘制的图形数量为20
graph_draw_num = 20
# 设置每行显示的图形数量为5
col_num = 5
# 计算需要的行数
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

# 按照验证集准确率排序并绘制图形
for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
    # 打印最佳结果
    print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

    # 创建一个子图
    plt.subplot(row_num, col_num, i+1)
    # 设置子图标题
    plt.title("Best-" + str(i+1))
    # 设置y轴范围
    plt.ylim(0.0, 1.0)
    # 每5个子图不显示y轴刻度
    if i % 5: plt.yticks([])
    # 不显示x轴刻度
    plt.xticks([])
    # 生成x轴数据
    x = np.arange(len(val_acc_list))
    # 绘制验证集准确率曲线
    plt.plot(x, val_acc_list)
    # 绘制训练集准确率曲线（虚线）
    plt.plot(x, results_train[key], "--")
    i += 1

    # 如果绘制的图形数量达到20，则停止
    if i >= graph_draw_num:
        break

# 显示所有图形
plt.show()
