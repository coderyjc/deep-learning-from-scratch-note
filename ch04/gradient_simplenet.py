# coding: utf-8
import sys, os
sys.path.append(os.pardir) 
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

# 定义一个名为 simpleNet 的类，表示一个简单的神经网络
class simpleNet:
    # 定义初始化方法 __init__，在创建 simpleNet 实例时自动调用
    def __init__(self):
        # 初始化权重矩阵 W，使用随机数生成一个 2x3 的矩阵
        self.W = np.random.randn(2,3)

    # 定义 predict 方法，用于根据输入 x 进行预测
    def predict(self, x):
        # 返回输入 x 与权重矩阵 W 的点积结果
        return np.dot(x, self.W)

    # 定义 loss 方法，用于计算损失值
    def loss(self, x, t):
        # 调用 predict 方法得到中间结果 z
        z = self.predict(x)
        # 对 z 进行 softmax 处理，得到概率分布 y
        y = softmax(z)
        # 计算交叉熵误差 loss
        loss = cross_entropy_error(y, t)
        # 返回损失值
        return loss

# 定义输入数据 x
x = np.array([0.6, 0.9])
# 定义目标标签 t
t = np.array([0, 0, 1])

# 创建一个 simpleNet 实例
net = simpleNet()

# 定义一个 lambda 函数 f，用于计算损失值
f = lambda w: net.loss(x, t)
# 使用数值梯度法计算权重矩阵 W 的梯度
dW = numerical_gradient(f, net.W)

# 打印计算得到的梯度 dW
print(dW)
