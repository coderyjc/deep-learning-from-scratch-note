# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# 定义 sigmoid 激活函数，将输入值映射到 (0, 1) 区间
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义 ReLU 激活函数，返回输入值和 0 之间的较大值
def ReLU(x):
    return np.maximum(0, x)

# 定义 tanh 激活函数，将输入值映射到 (-1, 1) 区间
def tanh(x):
    return np.tanh(x)

# 生成一个 1000x100 的随机输入数据矩阵
input_data = np.random.randn(1000, 100)

# 定义每层的节点数为 100
node_num = 100

# 定义隐藏层的层数为 5
hidden_layer_size = 5

# 初始化一个空字典，用于存储每一层的激活值
activations = {}

# 将输入数据赋值给变量 x
x = input_data

# 遍历每一层隐藏层
for i in range(hidden_layer_size):
    # 如果不是第一层，将上一层的激活值作为当前层的输入
    if i != 0:
        x = activations[i-1]

    # 生成一个 100x100 的随机权重矩阵，并乘以 1
    w = np.random.randn(node_num, node_num) * 1
    # 其他权重初始化方式（注释掉）
    # w = np.random.randn(node_num, node_num) * 0.01
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

    # 计算当前层的加权输入
    a = np.dot(x, w)

    # 使用 sigmoid 激活函数计算当前层的激活值
    z = sigmoid(a)
    # 其他激活函数（注释掉）
    # z = ReLU(a)
    # z = tanh(a)

    # 将当前层的激活值存储到字典中
    activations[i] = z

# 绘制每一层激活值的直方图
for i, a in activations.items():
    # 创建子图，用于显示当前层的激活值分布
    plt.subplot(1, len(activations), i+1)
    # 设置子图标题
    plt.title(str(i+1) + "-layer")
    # 如果不是第一层，隐藏 y 轴刻度
    if i != 0: plt.yticks([], [])
    # 设置 x 轴和 y 轴的范围（注释掉）
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)
    # 绘制当前层激活值的直方图，分为 30 个区间，范围在 (0, 1) 之间
    plt.hist(a.flatten(), 30, range=(0,1))
# 显示所有子图
plt.show()
