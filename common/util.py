# coding: utf-8
import numpy as np

# 平滑曲线函数，使用卷积操作对输入数据进行平滑处理
def smooth_curve(x):
    """
    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    # 定义卷积窗口的长度
    window_len = 11
    # 扩展输入数据，以便在边界处进行卷积
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    # 生成Kaiser窗口函数
    w = np.kaiser(window_len, 2)
    # 对扩展后的数据进行卷积操作
    y = np.convolve(w/w.sum(), s, mode='valid')
    # 返回平滑后的数据，去除边界效应
    return y[5:len(y)-5]

# 打乱数据集中的样本顺序
def shuffle_dataset(x, t):
    # 生成随机排列的索引
    permutation = np.random.permutation(x.shape[0])
    # 根据随机索引重新排列输入数据和标签
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]
    # 返回打乱后的输入数据和标签
    return x, t

# 计算卷积层输出的大小
def conv_output_size(input_size, filter_size, stride=1, pad=0):
    # 根据输入大小、滤波器大小、步长和填充计算输出大小
    return (input_size + 2*pad - filter_size) / stride + 1

# 将输入数据转换为列形式，以便进行卷积操作
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    # 获取输入数据的形状
    N, C, H, W = input_data.shape
    # 计算卷积输出的高度和宽度
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    # 对输入数据进行填充
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    # 初始化用于存储转换后数据的数组
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    # 遍历滤波器的高度和宽度，将输入数据转换为列形式
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    # 调整数据的形状并返回
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

# 将列形式的数据转换回图像形式
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    # 获取输入数据的形状
    N, C, H, W = input_shape
    # 计算卷积输出的高度和宽度
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    # 调整数据的形状
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    # 初始化用于存储转换后数据的数组
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    # 遍历滤波器的高度和宽度，将列形式的数据转换回图像形式
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    # 返回去除填充后的图像数据
    return img[:, :, pad:H + pad, pad:W + pad]
