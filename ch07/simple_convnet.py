# coding: utf-8
import sys, os
sys.path.append(os.pardir) 
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


# 定义一个简单的卷积神经网络类
class SimpleConvNet:

    # 初始化网络结构，设置输入维度、卷积参数、隐藏层大小、输出大小和权重初始化标准差
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        # 从卷积参数中提取滤波器数量、大小、填充和步幅
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        
        # 计算输入图像的尺寸
        input_size = input_dim[1]
        
        # 计算卷积层的输出尺寸
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        
        # 计算池化层的输出尺寸
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        # 初始化网络的参数
        self.params = {}
        # 初始化第一层卷积层的权重和偏置
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        # 初始化第二层全连接层的权重和偏置
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        # 初始化第三层全连接层的权重和偏置
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # 初始化网络层的有序字典
        self.layers = OrderedDict()
        # 添加第一层卷积层
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        # 添加第一层ReLU激活函数
        self.layers['Relu1'] = Relu()
        # 添加第一层池化层
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        # 添加第二层全连接层
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        # 添加第二层ReLU激活函数
        self.layers['Relu2'] = Relu()
        # 添加第三层全连接层
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        # 设置最后一层为SoftmaxWithLoss层
        self.last_layer = SoftmaxWithLoss()

    # 定义前向传播函数，用于预测输出
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # 定义损失函数，计算预测值与真实标签的损失
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    # 定义准确率计算函数，计算模型在给定数据上的准确率
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    # 定义数值梯度计算函数，用于计算参数的梯度
    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads

    # 定义梯度计算函数，通过反向传播计算参数的梯度
    def gradient(self, x, t):
        self.loss(x, t)

        # 反向传播
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 获取各层的梯度
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
        
    # 定义保存参数函数，将网络参数保存到文件中
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    # 定义加载参数函数，从文件中加载网络参数
    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        # 更新各层的权重和偏置
        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
