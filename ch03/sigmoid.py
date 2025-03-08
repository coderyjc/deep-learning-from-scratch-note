# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


# sigmoid 函数，用于计算输入 x 的 sigmoid 值
def sigmoid(x):
    # 返回 sigmoid 函数的计算结果：1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))    

# 创建一个从 -5.0 到 5.0 的数组，步长为 0.1
X = np.arange(-5.0, 5.0, 0.1)

# 对数组 X 中的每个元素应用 sigmoid 函数，得到对应的 Y 值
Y = sigmoid(X)

# 使用 matplotlib 绘制 X 和 Y 的曲线图
plt.plot(X, Y)

# 设置 y 轴的范围为 -0.1 到 1.1
plt.ylim(-0.1, 1.1)

# 显示绘制的图形
plt.show()
