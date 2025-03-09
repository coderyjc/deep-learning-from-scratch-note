# coding: utf-8
import numpy as np


# 定义一个函数 _numerical_gradient_1d，用于计算一维数组的数值梯度
def _numerical_gradient_1d(f, x):
    # 设置一个很小的值 h，用于计算梯度
    h = 1e-4
    # 创建一个与 x 形状相同的零数组，用于存储梯度
    grad = np.zeros_like(x)
    
    # 遍历 x 的每个元素
    for idx in range(x.size):
        # 保存当前元素的值
        tmp_val = x[idx]
        # 将当前元素增加 h，并计算函数值 fxh1
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        
        # 将当前元素减少 h，并计算函数值 fxh2
        x[idx] = tmp_val - h 
        fxh2 = f(x)
        # 计算当前元素的梯度，并存储在 grad 中
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        # 恢复当前元素的原始值
        x[idx] = tmp_val
        
    # 返回计算得到的梯度
    return grad


# 定义一个函数 numerical_gradient_2d，用于计算二维数组的数值梯度
def numerical_gradient_2d(f, X):
    # 如果 X 是一维数组，直接调用 _numerical_gradient_1d 计算梯度
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        # 创建一个与 X 形状相同的零数组，用于存储梯度
        grad = np.zeros_like(X)
        
        # 遍历 X 的每一行，并计算每行的梯度
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)
        
        # 返回计算得到的梯度
        return grad


# 定义一个函数 numerical_gradient，用于计算任意维度数组的数值梯度
def numerical_gradient(f, x):
    # 设置一个很小的值 h，用于计算梯度
    h = 1e-4
    # 创建一个与 x 形状相同的零数组，用于存储梯度
    grad = np.zeros_like(x)
    
    # 使用 np.nditer 遍历 x 的所有元素
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # 获取当前元素的索引
        idx = it.multi_index
        # 保存当前元素的值
        tmp_val = x[idx]
        # 将当前元素增加 h，并计算函数值 fxh1
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        # 将当前元素减少 h，并计算函数值 fxh2
        x[idx] = tmp_val - h 
        fxh2 = f(x)
        # 计算当前元素的梯度，并存储在 grad 中
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        # 恢复当前元素的原始值
        x[idx] = tmp_val
        # 移动到下一个元素
        it.iternext()   
        
    # 返回计算得到的梯度
    return grad
