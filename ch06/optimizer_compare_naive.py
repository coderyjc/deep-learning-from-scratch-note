# coding: utf-8
import sys, os
sys.path.append(os.pardir) 
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.optimizer import *


# 函数 f，计算 x 和 y 的平方和的加权值
def f(x, y):
    return x**2 / 20.0 + y**2

# 函数 df，计算 f 函数对 x 和 y 的偏导数
def df(x, y):
    return x / 10.0, 2.0*y

# 初始化位置参数
init_pos = (-7.0, 2.0)

# 创建一个空字典 params，用于存储 x 和 y 的值
params = {}
# 将初始位置赋值给 params 中的 x 和 y
params['x'], params['y'] = init_pos[0], init_pos[1]

# 创建一个空字典 grads，用于存储 x 和 y 的梯度值
grads = {}
# 初始化 grads 中的 x 和 y 为 0
grads['x'], grads['y'] = 0, 0

# 创建一个有序字典 optimizers，用于存储不同的优化器
optimizers = OrderedDict()
# 添加 SGD 优化器到 optimizers 字典中
optimizers["SGD"] = SGD(lr=0.95)
# 添加 Momentum 优化器到 optimizers 字典中
optimizers["Momentum"] = Momentum(lr=0.1)
# 添加 AdaGrad 优化器到 optimizers 字典中
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
# 添加 Adam 优化器到 optimizers 字典中
optimizers["Adam"] = Adam(lr=0.3)

# 初始化索引 idx 为 1，用于绘制子图
idx = 1

# 遍历 optimizers 字典中的每个优化器
for key in optimizers:
    # 获取当前优化器
    optimizer = optimizers[key]
    # 初始化 x 和 y 的历史记录列表
    x_history = []
    y_history = []
    # 重置 params 中的 x 和 y 为初始位置
    params['x'], params['y'] = init_pos[0], init_pos[1]
    
    # 进行 30 次迭代
    for i in range(30):
        # 将当前的 x 和 y 值添加到历史记录中
        x_history.append(params['x'])
        y_history.append(params['y'])
        
        # 计算当前的梯度
        grads['x'], grads['y'] = df(params['x'], params['y'])
        # 使用优化器更新参数
        optimizer.update(params, grads)
    
    # 生成 x 和 y 的网格数据
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    
    # 创建网格矩阵 X 和 Y
    X, Y = np.meshgrid(x, y) 
    # 计算每个网格点的函数值 Z
    Z = f(X, Y)
    
    # 将 Z 中大于 7 的值置为 0
    mask = Z > 7
    Z[mask] = 0
    
    # 创建一个 2x2 的子图，当前绘制第 idx 个子图
    plt.subplot(2, 2, idx)
    # 增加索引 idx
    idx += 1
    # 绘制 x 和 y 的历史路径
    plt.plot(x_history, y_history, 'o-', color="red")
    # 绘制等高线图
    plt.contour(X, Y, Z)
    # 设置 y 轴的范围
    plt.ylim(-10, 10)
    # 设置 x 轴的范围
    plt.xlim(-10, 10)
    # 在图中标记原点
    plt.plot(0, 0, '+')
   
    # 设置子图的标题
    plt.title(key)
    # 设置 x 轴标签
    plt.xlabel("x")
    # 设置 y 轴标签
    plt.ylabel("y")
    
# 显示所有绘制的图形
plt.show()
