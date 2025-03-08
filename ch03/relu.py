# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


# ReLU 激活函数，返回输入 x 和 0 之间的最大值
def relu(x):
    return np.maximum(0, x)

# 创建一个从 -5.0 到 5.0 的数组，步长为 0.1
x = np.arange(-5.0, 5.0, 0.1)

# 对数组 x 应用 ReLU 函数，得到输出 y
y = relu(x)

# 绘制 x 和 y 的关系图
plt.plot(x, y)

# 设置 y 轴的范围为 -1.0 到 5.5
plt.ylim(-1.0, 5.5)

# 显示绘制的图形
plt.show()
