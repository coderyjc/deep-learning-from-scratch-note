# coding: utf-8
import sys, os

import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


# 将当前目录的父目录添加到系统路径中
sys.path.append(os.pardir)

# 定义一个函数 img_show，用于显示图像
def img_show(img):
    # 将 numpy 数组转换为 PIL 图像对象
    pil_img = Image.fromarray(np.uint8(img))
    # 显示图像
    pil_img.show()

# 加载 MNIST 数据集，flatten=True 表示将图像展平为一维数组，normalize=False 表示不进行归一化
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# 获取训练集中的第一张图像
img = x_train[0]
# 获取对应的标签
label = t_train[0]
# 打印标签
print(label) 

# 打印图像的形状（展平后的形状）
print(img.shape) 
# 将图像从一维数组重塑为 28x28 的二维数组
img = img.reshape(28, 28) 
# 打印重塑后的图像形状
print(img.shape) 

# 调用 img_show 函数显示图像
img_show(img)

