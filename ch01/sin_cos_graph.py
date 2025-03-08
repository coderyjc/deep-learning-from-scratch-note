# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# 使用 numpy 的 arange 函数生成从 0 到 6，步长为 0.1 的数组
x = np.arange(0, 6, 0.1)

# 计算数组 x 中每个元素的正弦值，并存储在数组 y1 中
y1 = np.sin(x)

# 计算数组 x 中每个元素的余弦值，并存储在数组 y2 中
y2 = np.cos(x)

# 绘制 y1（sin(x)）的图形，并设置标签为 "sin"
plt.plot(x, y1, label="sin")

# 绘制 y2（cos(x)）的图形，设置线型为虚线，并设置标签为 "cos"
plt.plot(x, y2, linestyle="--", label="cos")

# 设置 x 轴的标签为 "x"
plt.xlabel("x")

# 设置 y 轴的标签为 "y"
plt.ylabel("y")

# 设置图表的标题为 'sin & cos'
plt.title('sin & cos')

# 显示图例，用于区分 sin 和 cos 的曲线
plt.legend()

# 显示绘制的图形
plt.show()