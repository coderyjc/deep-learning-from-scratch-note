# coding: utf-8
import sys, os
sys.path.append(os.pardir) 
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient

# 定义一个扩展的多层神经网络类，支持多种功能如 Dropout、BatchNorm 等
class MultiLayerNetExtend:

    # 初始化网络结构，设置输入大小、隐藏层大小列表、输出大小等参数
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0, 
                 use_dropout = False, dropout_ration = 0.5, use_batchnorm=False):
        # 设置输入层大小
        self.input_size = input_size
        # 设置输出层大小
        self.output_size = output_size
        # 设置隐藏层大小列表
        self.hidden_size_list = hidden_size_list
        # 计算隐藏层的数量
        self.hidden_layer_num = len(hidden_size_list)
        # 是否使用 Dropout
        self.use_dropout = use_dropout
        # 权重衰减系数
        self.weight_decay_lambda = weight_decay_lambda
        # 是否使用 BatchNorm
        self.use_batchnorm = use_batchnorm
        # 初始化参数字典
        self.params = {}

        # 初始化权重
        self.__init_weight(weight_init_std)

        # 定义激活函数层
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        # 使用有序字典存储网络层
        self.layers = OrderedDict()
        # 遍历隐藏层，构建网络结构
        for idx in range(1, self.hidden_layer_num+1):
            # 添加全连接层
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            # 如果使用 BatchNorm，添加 BatchNorm 层
            if self.use_batchnorm:
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1])
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx-1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)], self.params['beta' + str(idx)])
                
            # 添加激活函数层
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()
            
            # 如果使用 Dropout，添加 Dropout 层
            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)

        # 添加输出层的全连接层
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

        # 设置最后一层为 Softmax 损失层
        self.last_layer = SoftmaxWithLoss()

    # 初始化权重参数
    def __init_weight(self, weight_init_std):
        # 构建所有层的大小列表
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        # 遍历每一层，初始化权重和偏置
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            # 根据激活函数类型设置初始化权重比例
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # ReLU 推荐的初始化值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # Sigmoid 推荐的初始化值
            # 初始化权重矩阵
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            # 初始化偏置向量
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    # 前向传播，用于预测
    def predict(self, x, train_flg=False):
        # 遍历所有层，进行前向传播
        for key, layer in self.layers.items():
            # 如果是 Dropout 或 BatchNorm 层，需要传入 train_flg 参数
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    # 计算损失函数，包括权重衰减
    def loss(self, x, t, train_flg=False):
        # 调用 predict 方法进行前向传播
        y = self.predict(x, train_flg)

        # 计算权重衰减项
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        # 返回损失值，包括 Softmax 损失和权重衰减
        return self.last_layer.forward(y, t) + weight_decay

    # 计算模型准确率
    def accuracy(self, x, t):
        # 调用 predict 方法进行前向传播
        y = self.predict(x, train_flg=False)
        # 获取预测结果的类别
        y = np.argmax(y, axis=1)
        # 如果目标值是多维的，获取其类别
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        # 计算准确率
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 计算数值梯度
    def numerical_gradient(self, x, t):
        # 定义损失函数
        loss_W = lambda W: self.loss(x, t, train_flg=True)

        # 初始化梯度字典
        grads = {}
        # 遍历所有层，计算梯度
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])
            
            # 如果使用 BatchNorm，计算 gamma 和 beta 的梯度
            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = numerical_gradient(loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(loss_W, self.params['beta' + str(idx)])

        return grads
        
    # 计算梯度（反向传播）
    def gradient(self, x, t):
        # 前向传播计算损失
        self.loss(x, t, train_flg=True)

        # 反向传播
        dout = 1
        dout = self.last_layer.backward(dout)

        # 反转层列表，方便反向传播
        layers = list(self.layers.values())
        layers.reverse()
        # 遍历所有层，进行反向传播
        for layer in layers:
            dout = layer.backward(dout)

        # 计算梯度
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.params['W' + str(idx)]
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            # 如果使用 BatchNorm，计算 gamma 和 beta 的梯度
            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta

        return grads