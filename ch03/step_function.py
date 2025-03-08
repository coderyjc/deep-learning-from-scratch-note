# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


# 阶跃函数 step_function，输入 x 返回一个布尔值数组，并将其转换为整数
def step_function(x):
    return np.array(x > 0, dtype=int)

# 使用 np.arange 生成一个从 -5.0 到 5.0 的数组，步长为 0.1
X = np.arange(-5.0, 5.0, 0.1)

# 将 X 作为输入传递给 step_function，得到对应的 Y 值
Y = step_function(X)

# 使用 matplotlib 绘制 X 和 Y 的图形
plt.plot(X, Y)

# 设置 y 轴的范围为 -0.1 到 1.1
plt.ylim(-0.1, 1.1)

# 显示绘制的图形
plt.show()

