# coding: utf-8
import sys
sys.path.append('.')

import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

# 定义一个名为 TwoLayerNet 的类，用于表示一个两层的神经网络
class TwoLayerNet:

    # 初始化方法，设置网络的结构和参数
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
       
        # 初始化参数字典，包含权重和偏置
        self.params = {}
        # 初始化第一层的权重矩阵 W1，使用正态分布随机生成
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 初始化第一层的偏置向量 b1，全零
        self.params['b1'] = np.zeros(hidden_size)
        # 初始化第二层的权重矩阵 W2，使用正态分布随机生成
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        # 初始化第二层的偏置向量 b2，全零
        self.params['b2'] = np.zeros(output_size)

        # 初始化网络层的有序字典，包含各层的计算
        self.layers = OrderedDict()
        # 第一层全连接层（Affine1）
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        # 第一层激活函数（Relu1）
        self.layers['Relu1'] = Relu()
        # 第二层全连接层（Affine2）
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        # 最后一层使用 Softmax 和 Loss 层
        self.lastLayer = SoftmaxWithLoss()
        
    # 预测方法，通过前向传播计算输出
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # 计算损失函数
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    # 计算准确率
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # 使用数值梯度法计算梯度
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    # 使用反向传播计算梯度
    def gradient(self, x, t):
       
        # 计算损失
        self.loss(x, t)

        # 反向传播开始
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        # 反向传播每一层
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 获取梯度
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
