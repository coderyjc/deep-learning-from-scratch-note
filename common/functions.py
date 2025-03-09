# coding: utf-8
import numpy as np

# 定义一个恒等函数，返回输入值本身
def identity_function(x):
    return x

# 定义一个阶跃函数，输入大于0时返回1，否则返回0
def step_function(x):
    return np.array(x > 0, dtype=int)

# 定义Sigmoid函数，将输入值映射到0到1之间
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义Sigmoid函数的导数
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

# 定义ReLU函数，返回输入值和0之间的较大值
def relu(x):
    return np.maximum(0, x)

# 定义ReLU函数的导数，输入大于0时返回1，否则返回0
def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x>=0] = 1
    return grad

# 定义Softmax函数，将输入值转换为概率分布
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)  # 防止数值溢出
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

# 定义平方和误差函数，计算预测值与真实值之间的误差
def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 定义交叉熵误差函数，用于多分类问题的误差计算
def cross_entropy_error(y, t):
    if y.ndim == 1:  # 如果输入是一维数组，将其转换为二维数组
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:  # 如果t是one-hot编码，转换为类别标签
        t = t.argmax(axis=1)

    batch_size = y.shape[0]  # 获取批量大小
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size  # 计算交叉熵误差

# 定义Softmax损失函数，结合Softmax和交叉熵误差
def softmax_loss(X, t):
    y = softmax(X)  # 先计算Softmax
    return cross_entropy_error(y, t)  # 再计算交叉熵误差
