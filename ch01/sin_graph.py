# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# 使用 numpy 的 arange 函数生成从 0 到 6，步长为 0.1 的数组
x = np.arange(0, 6, 0.1)

# 计算数组 x 中每个元素的正弦值，并存储在数组 y 中
y = np.sin(x)

# 使用 matplotlib 的 plot 函数绘制 x 和 y 的图形
plt.plot(x, y)

# 显示绘制的图形
plt.show()
