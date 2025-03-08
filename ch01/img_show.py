# coding: utf-8

import matplotlib.pyplot as plt
from matplotlib.image import imread

# 使用 imread 函数从指定路径读取图片文件，并将其存储在变量 img 中
img = imread('./dataset/cover.jpg')

# 使用 plt.imshow 函数将图片数据显示为图像
plt.imshow(img)

# 使用 plt.show 函数显示图像窗口
plt.show()