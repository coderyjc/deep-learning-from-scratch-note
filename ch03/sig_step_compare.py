# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


# sigmoid 函数，用于计算输入 x 的 sigmoid 值
def sigmoid(x):
    # 返回 1 / (1 + e^(-x))，即 sigmoid 函数的值
    return 1 / (1 + np.exp(-x))

# step_function 函数，用于计算输入 x 的阶跃函数值
def step_function(x):
    # 返回一个布尔数组，表示 x 是否大于 0，并将其转换为整数类型
    return np.array(x > 0, dtype=int)

# 生成一个从 -5.0 到 5.0 的数组，步长为 0.1
x = np.arange(-5.0, 5.0, 0.1)

# 计算 sigmoid 函数在 x 上的值
y1 = sigmoid(x)

# 计算阶跃函数在 x 上的值
y2 = step_function(x)

# 绘制 sigmoid 函数的图像
plt.plot(x, y1)

# 绘制阶跃函数的图像，使用黑色虚线表示
plt.plot(x, y2, 'k--')

# 设置 y 轴的范围为 -0.1 到 1.1
plt.ylim(-0.1, 1.1)

# 显示绘制的图像
plt.show()

