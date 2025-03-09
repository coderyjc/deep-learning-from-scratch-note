# coding: utf-8
import numpy as np
from common.functions import *
from common.util import im2col, col2im


# 定义一个名为 Relu 的类，用于实现 ReLU 激活函数
class Relu:
    def __init__(self):
        # 初始化 mask 属性，用于记录输入中哪些元素小于等于 0
        self.mask = None

    def forward(self, x):
        # 计算 mask，标记输入中小于等于 0 的元素
        self.mask = (x <= 0)
        # 复制输入 x 到 out
        out = x.copy()
        # 将 out 中标记为 True 的元素设置为 0
        out[self.mask] = 0
        # 返回处理后的输出
        return out

    def backward(self, dout):
        # 在反向传播中，将 dout 中标记为 True 的元素设置为 0
        dout[self.mask] = 0
        # 返回处理后的梯度
        dx = dout
        return dx


# 定义一个名为 Sigmoid 的类，用于实现 Sigmoid 激活函数
class Sigmoid:
    def __init__(self):
        # 初始化 out 属性，用于保存前向传播的输出
        self.out = None

    def forward(self, x):
        # 计算 Sigmoid 激活函数的输出
        out = sigmoid(x)
        # 保存输出到 self.out
        self.out = out
        # 返回输出
        return out

    def backward(self, dout):
        # 计算 Sigmoid 函数的反向传播梯度
        dx = dout * (1.0 - self.out) * self.out
        # 返回梯度
        return dx


# 定义一个名为 Affine 的类，用于实现仿射变换
class Affine:
    def __init__(self, W, b):
        # 初始化权重 W 和偏置 b
        self.W = W
        self.b = b
        # 初始化 x 和 original_x_shape 属性，用于保存输入和其原始形状
        self.x = None
        self.original_x_shape = None
        # 初始化 dW 和 db 属性，用于保存反向传播中的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        # 保存输入的原始形状
        self.original_x_shape = x.shape
        # 将输入 x 重塑为二维矩阵
        x = x.reshape(x.shape[0], -1)
        # 保存重塑后的输入
        self.x = x
        # 计算仿射变换的输出
        out = np.dot(self.x, self.W) + self.b
        # 返回输出
        return out

    def backward(self, dout):
        # 计算输入 x 的梯度
        dx = np.dot(dout, self.W.T)
        # 计算权重 W 的梯度
        self.dW = np.dot(self.x.T, dout)
        # 计算偏置 b 的梯度
        self.db = np.sum(dout, axis=0)
        # 将梯度 dx 重塑为原始输入的形状
        dx = dx.reshape(*self.original_x_shape)
        # 返回梯度
        return dx


# 定义一个名为 SoftmaxWithLoss 的类，用于实现 Softmax 和交叉熵损失
class SoftmaxWithLoss:
    def __init__(self):
        # 初始化 loss、y 和 t 属性，用于保存损失、预测值和真实标签
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        # 保存真实标签
        self.t = t
        # 计算 Softmax 函数的输出
        self.y = softmax(x)
        # 计算交叉熵损失
        self.loss = cross_entropy_error(self.y, self.t)
        # 返回损失
        return self.loss

    def backward(self, dout=1):
        # 获取批量大小
        batch_size = self.t.shape[0]
        # 如果真实标签和预测值的形状相同，直接计算梯度
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            # 否则，复制预测值并调整梯度
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        # 返回梯度
        return dx


# 定义一个名为 Dropout 的类，用于实现 Dropout 正则化
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        # 初始化 dropout_ratio 和 mask 属性
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        # 如果处于训练模式，随机生成 mask 并应用 Dropout
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            # 否则，直接缩放输入
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        # 在反向传播中，将梯度 dout 乘以 mask
        return dout * self.mask


# 定义一个名为 BatchNormalization 的类，用于实现批量归一化
class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        # 初始化 gamma、beta、momentum 等属性
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None
        # 初始化 running_mean 和 running_var 属性，用于保存移动平均值
        self.running_mean = running_mean
        self.running_var = running_var
        # 初始化其他属性，用于保存中间结果
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        # 保存输入的形状
        self.input_shape = x.shape
        # 如果输入不是二维的，重塑为二维
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)
        # 调用 __forward 方法进行前向传播
        out = self.__forward(x, train_flg)
        # 将输出重塑为原始输入的形状
        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        # 如果 running_mean 和 running_var 未初始化，进行初始化
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
        # 如果处于训练模式，计算均值和方差，并更新 running_mean 和 running_var
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            # 否则，使用 running_mean 和 running_var 进行归一化
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
        # 计算输出
        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        # 如果 dout 不是二维的，重塑为二维
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)
        # 调用 __backward 方法进行反向传播
        dx = self.__backward(dout)
        # 将梯度 dx 重塑为原始输入的形状
        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        # 计算 beta 和 gamma 的梯度
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        # 计算 xn 的梯度
        dxn = self.gamma * dout
        # 计算 xc 的梯度
        dxc = dxn / self.std
        # 计算 std 和 var 的梯度
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        # 更新 xc 的梯度
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        # 计算 mu 的梯度
        dmu = np.sum(dxc, axis=0)
        # 计算最终梯度 dx
        dx = dxc - dmu / self.batch_size
        # 保存 gamma 和 beta 的梯度
        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx


# 定义一个名为 Convolution 的类，用于实现卷积操作
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        # 初始化卷积核 W、偏置 b、步幅 stride 和填充 pad
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        # 初始化 x、col 和 col_W 属性，用于保存输入和中间结果
        self.x = None
        self.col = None
        self.col_W = None
        # 初始化 dW 和 db 属性，用于保存反向传播中的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        # 获取卷积核的形状
        FN, C, FH, FW = self.W.shape
        # 获取输入的形状
        N, C, H, W = x.shape
        # 计算输出的高度和宽度
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
        # 将输入转换为列矩阵
        col = im2col(x, FH, FW, self.stride, self.pad)
        # 将卷积核转换为列矩阵
        col_W = self.W.reshape(FN, -1).T
        # 计算卷积输出
        out = np.dot(col, col_W) + self.b
        # 将输出重塑为正确的形状
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        # 保存输入和中间结果
        self.x = x
        self.col = col
        self.col_W = col_W
        return out

    def backward(self, dout):
        # 获取卷积核的形状
        FN, C, FH, FW = self.W.shape
        # 将 dout 转换为列矩阵
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
        # 计算偏置 b 的梯度
        self.db = np.sum(dout, axis=0)
        # 计算卷积核 W 的梯度
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        # 计算输入 x 的梯度
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx


# 定义一个名为 Pooling 的类，用于实现池化操作
class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        # 初始化池化窗口的高度、宽度、步幅和填充
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        # 初始化 x 和 arg_max 属性，用于保存输入和最大值的位置
        self.x = None
        self.arg_max = None

    def forward(self, x):
        # 获取输入的形状
        N, C, H, W = x.shape
        # 计算输出的高度和宽度
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        # 将输入转换为列矩阵
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)
        # 找到每个池化窗口中的最大值及其位置
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        # 将输出重塑为正确的形状
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        # 保存输入和最大值的位置
        self.x = x
        self.arg_max = arg_max
        return out

    def backward(self, dout):
        # 将 dout 转换为列矩阵
        dout = dout.transpose(0, 2, 3, 1)
        # 创建一个与 dout 大小相同的零矩阵，用于保存梯度
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        # 将梯度 dout 分配到最大值的位置
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))
        # 将梯度转换为列矩阵
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        # 将梯度转换为输入的形状
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx
