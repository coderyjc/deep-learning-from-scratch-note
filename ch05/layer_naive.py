# coding: utf-8

# 定义一个名为 MulLayer 的类，用于实现乘法层的正向和反向传播
class MulLayer:
    # 初始化方法，设置两个属性 x 和 y 为 None
    def __init__(self):
        self.x = None
        self.y = None

    # 正向传播方法，计算输入 x 和 y 的乘积
    def forward(self, x, y):
        # 保存输入值 x 和 y
        self.x = x
        self.y = y                
        # 计算并返回 x 和 y 的乘积
        out = x * y
        return out

    # 反向传播方法，计算梯度
    def backward(self, dout):
        # 计算 x 的梯度，即 dout 乘以 y
        dx = dout * self.y
        # 计算 y 的梯度，即 dout 乘以 x
        dy = dout * self.x
        # 返回 x 和 y 的梯度
        return dx, dy


# 定义一个名为 AddLayer 的类，用于实现加法层的正向和反向传播
class AddLayer:
    # 初始化方法，无需额外操作
    def __init__(self):
        pass

    # 正向传播方法，计算输入 x 和 y 的和
    def forward(self, x, y):
        # 计算并返回 x 和 y 的和
        out = x + y
        return out

    # 反向传播方法，计算梯度
    def backward(self, dout):
        # 计算 x 的梯度，即 dout 乘以 1
        dx = dout * 1
        # 计算 y 的梯度，即 dout 乘以 1
        dy = dout * 1
        # 返回 x 和 y 的梯度
        return dx, dy
