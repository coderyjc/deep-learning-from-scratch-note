# coding: utf-8
import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from simple_convnet import SimpleConvNet
from matplotlib.image import imread
from common.layers import Convolution

# 可视化卷积核的函数，用于展示卷积核的权重
def filter_show(filters, nx=4, show_num=16):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    # 获取卷积核的形状，FN表示卷积核的数量，C表示通道数，FH和FW表示卷积核的高度和宽度
    FN, C, FH, FW = filters.shape
    # 计算需要展示的行数
    ny = int(np.ceil(show_num / nx))

    # 创建一个新的图形
    fig = plt.figure()
    # 调整子图的间距
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # 遍历需要展示的卷积核数量
    for i in range(show_num):
        # 添加子图，并去除刻度
        ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
        # 显示卷积核的第一个通道的图像
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')


# 创建一个简单的卷积神经网络实例
network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

# 加载训练好的网络参数
network.load_params("ch07\\params.pkl")

# 调用filter_show函数，展示卷积核的权重
filter_show(network.params['W1'], 16)

# 读取灰度图像
img = imread('img/lena_gray.png')
# 调整图像的形状以匹配网络输入
img = img.reshape(1, 1, *img.shape)

# 创建一个新的图形
fig = plt.figure()

# 初始化权重索引
w_idx = 1

# 遍历16个卷积核
for i in range(16):
    # 获取卷积核的权重和偏置
    w = network.params['W1'][i]
    b = 0 

    # 调整权重的形状以匹配卷积层的要求
    w = w.reshape(1, *w.shape)
    # 创建卷积层
    conv_layer = Convolution(w, b) 
    # 对图像进行卷积操作
    out = conv_layer.forward(img)
    # 调整输出的形状以方便显示
    out = out.reshape(out.shape[2], out.shape[3])
    
    # 添加子图，并去除刻度
    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    # 显示卷积后的图像
    ax.imshow(out, cmap=plt.cm.gray_r, interpolation='nearest')

# 显示所有子图
plt.show()
