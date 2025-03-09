# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from simple_convnet import SimpleConvNet


# 定义一个函数，用于可视化卷积核的权重
def filter_show(filters, nx=8, margin=3, scale=10):
    """
    参考：https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    # 获取卷积核的形状：FN（卷积核数量）、C（通道数）、FH（高度）、FW（宽度）
    FN, C, FH, FW = filters.shape
    
    # 计算子图的行数 ny，基于卷积核数量和每行显示的数量 nx
    ny = int(np.ceil(FN / nx))

    # 创建一个新的图形
    fig = plt.figure()
    
    # 调整子图的间距，使其紧密排列
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # 遍历每个卷积核，绘制其权重
    for i in range(FN):
        # 添加子图，并去除坐标轴
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        
        # 显示卷积核的权重，使用灰度图并插值为最近邻
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    
    # 显示图形
    plt.show()

# 创建一个简单的卷积神经网络实例
network = SimpleConvNet()

# 可视化随机初始化后的卷积核权重
filter_show(network.params['W1'])

# 加载训练后的权重参数
network.load_params("params.pkl")

# 可视化训练后的卷积核权重
filter_show(network.params['W1'])
