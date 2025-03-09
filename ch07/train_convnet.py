# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
from common.trainer import Trainer

# 加载MNIST数据集，并将数据分为训练集和测试集
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 设置最大训练轮数为20
max_epochs = 20

# 创建一个简单的卷积神经网络，指定输入维度、卷积参数、隐藏层大小、输出层大小和权重初始化标准差
network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

# 创建一个训练器对象，用于训练神经网络，指定网络、训练数据、测试数据、最大轮数、小批量大小、优化器及其参数
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)

# 开始训练神经网络
trainer.train()

# 将训练好的网络参数保存到文件 "params.pkl" 中
network.save_params("params.pkl")

# 打印提示信息，表示网络参数已保存
print("Saved Network Parameters!")

# 定义绘图时使用的标记符号，训练集用 'o'，测试集用 's'
markers = {'train': 'o', 'test': 's'}

# 创建一个从0到最大训练轮数的数组，用于绘制x轴
x = np.arange(max_epochs)

# 绘制训练准确率曲线，使用 'o' 标记，每两个点标记一次
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)

# 绘制测试准确率曲线，使用 's' 标记，每两个点标记一次
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)

# 设置x轴标签为 "epochs"
plt.xlabel("epochs")

# 设置y轴标签为 "accuracy"
plt.ylabel("accuracy")

# 设置y轴范围为0到1.0
plt.ylim(0, 1.0)

# 在右下角显示图例
plt.legend(loc='lower right')

# 显示绘制的图形
plt.show()
