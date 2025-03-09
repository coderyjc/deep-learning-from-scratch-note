# coding: utf-8
import sys
sys.path.append('.')

from common.functions import *
from common.gradient import numerical_gradient
import numpy as np

# 定义一个名为 TwoLayerNet 的类，表示一个两层的神经网络
class TwoLayerNet:

    # 初始化方法，设置网络的输入大小、隐藏层大小、输出大小和权重初始化标准差
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
       
        # 初始化参数字典，包含权重和偏置
        self.params = {}
        # 初始化第一层的权重矩阵 W1
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 初始化第一层的偏置向量 b1
        self.params['b1'] = np.zeros(hidden_size)
        # 初始化第二层的权重矩阵 W2
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # 初始化第二层的偏置向量 b2
        self.params['b2'] = np.zeros(output_size)

    # 定义 predict 方法，用于进行前向传播并返回输出
    def predict(self, x):
        # 获取权重和偏置
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        # 计算第一层的激活值
        a1 = np.dot(x, W1) + b1
        # 通过 sigmoid 函数进行激活
        z1 = sigmoid(a1)
        # 计算第二层的激活值
        a2 = np.dot(z1, W2) + b2
        # 通过 softmax 函数进行激活，得到输出
        y = softmax(a2)
        
        # 返回输出结果
        return y
        
    # 定义 loss 方法，计算网络的损失值
    def loss(self, x, t):
        # 通过 predict 方法获取输出
        y = self.predict(x)
        
        # 计算交叉熵误差并返回
        return cross_entropy_error(y, t)
    
    # 定义 accuracy 方法，计算网络的准确率
    def accuracy(self, x, t):
        # 通过 predict 方法获取输出
        y = self.predict(x)
        # 获取预测结果的类别
        y = np.argmax(y, axis=1)
        # 获取真实标签的类别
        t = np.argmax(t, axis=1)
        
        # 计算准确率并返回
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # 定义 numerical_gradient 方法，计算网络中参数的数值梯度
    def numerical_gradient(self, x, t):
        # 定义一个 lambda 函数，用于计算损失
        loss_W = lambda W: self.loss(x, t)
        
        # 初始化梯度字典
        grads = {}
        # 计算 W1 的梯度
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        # 计算 b1 的梯度
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        # 计算 W2 的梯度
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        # 计算 b2 的梯度
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        # 返回梯度字典
        return grads
        
    # 定义 gradient 方法，计算网络中参数的梯度
    def gradient(self, x, t):
        # 获取权重和偏置
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        # 初始化梯度字典
        grads = {}
        
        # 获取批次大小
        batch_num = x.shape[0]
        
        # 计算第一层的激活值
        a1 = np.dot(x, W1) + b1
        # 通过 sigmoid 函数进行激活
        z1 = sigmoid(a1)
        # 计算第二层的激活值
        a2 = np.dot(z1, W2) + b2
        # 通过 softmax 函数进行激活，得到输出
        y = softmax(a2)
        
        # 计算输出层的误差
        dy = (y - t) / batch_num
        # 计算 W2 的梯度
        grads['W2'] = np.dot(z1.T, dy)
        # 计算 b2 的梯度
        grads['b2'] = np.sum(dy, axis=0)
        
        # 计算第一层的误差
        dz1 = np.dot(dy, W2.T)
        # 计算第一层的激活值梯度
        da1 = sigmoid_grad(a1) * dz1
        # 计算 W1 的梯度
        grads['W1'] = np.dot(x.T, da1)
        # 计算 b1 的梯度
        grads['b1'] = np.sum(da1, axis=0)

        # 返回梯度字典
        return grads
