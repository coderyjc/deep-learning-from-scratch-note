# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet
from common.trainer import Trainer

# 加载MNIST数据集，并将其分为训练集和测试集，flatten=False表示不将图像展平
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 创建一个深度卷积神经网络（DeepConvNet）的实例
network = DeepConvNet()  

# 创建一个训练器（Trainer）实例，用于训练网络
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)

# 开始训练网络
trainer.train()

# 将训练好的网络参数保存到文件"deep_convnet_params.pkl"中
network.save_params("ch08\\deep_convnet_params.pkl")

# 输出提示信息，表示网络参数已成功保存
print("Saved Network Parameters!")
