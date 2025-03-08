# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

# 计算函数 f 在点 x 处的数值梯度（非批量处理）
def _numerical_gradient_no_batch(f, x):
    # 定义一个很小的值 h，用于计算梯度
    h = 1e-4  
    # 初始化梯度数组，形状与 x 相同
    grad = np.zeros_like(x)  
    
    # 遍历 x 的每个元素，计算梯度
    for idx in range(x.size):
        # 保存当前元素的值
        tmp_val = x[idx]  
        # 将当前元素增加 h
        x[idx] = float(tmp_val) + h  
        # 计算 f(x + h)
        fxh1 = f(x)  
        
        # 将当前元素减少 h
        x[idx] = tmp_val - h  
        # 计算 f(x - h)
        fxh2 = f(x)  
        # 计算中心差分梯度
        grad[idx] = (fxh1 - fxh2) / (2*h)  
        
        # 恢复 x 的原始值
        x[idx] = tmp_val  
        
    # 返回计算得到的梯度
    return grad  

# 计算函数 f 在点 X 处的数值梯度（支持批量处理）
def numerical_gradient(f, X):
    # 如果 X 是一维数组
    if X.ndim == 1:  
        # 调用非批量处理函数
        return _numerical_gradient_no_batch(f, X)  
    else:  # 如果 X 是多维数组
        # 初始化梯度数组，形状与 X 相同
        grad = np.zeros_like(X)  
        
        # 遍历 X 的每一行，计算梯度
        for idx, x in enumerate(X):
            # 调用非批量处理函数
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        # 返回计算得到的梯度
        return grad  

# 定义一个简单的二次函数 f(x) = x^2
def function_2(x):
    # 如果 x 是一维数组
    if x.ndim == 1:  
        # 返回 x 的平方和
        return np.sum(x**2)  
    else:  # 如果 x 是多维数组
        # 返回每行的平方和
        return np.sum(x**2, axis=1)  

# 计算函数 f 在点 x 处的切线
def tangent_line(f, x):
    # 计算梯度
    d = numerical_gradient(f, x)  
    # 打印梯度
    print(d)  
    # 计算切线的截距
    y = f(x) - d*x  
    # 返回切线函数
    return lambda t: d*t + y  

# 主程序入口
if __name__ == '__main__':
    # 生成 x0 的坐标范围
    x0 = np.arange(-2, 2.5, 0.25)  
    # 生成 x1 的坐标范围
    x1 = np.arange(-2, 2.5, 0.25)  
    # 生成网格点
    X, Y = np.meshgrid(x0, x1)  
    
    # 将 X 展平为一维数组
    X = X.flatten()  
    # 将 Y 展平为一维数组
    Y = Y.flatten()  

    # 计算 function_2 在网格点上的梯度
    grad = numerical_gradient(function_2, np.array([X, Y]).T).T

    # 绘制梯度场
    plt.figure()
    # 绘制箭头图
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")
    # 设置 x 轴范围
    plt.xlim([-2, 2])  
    # 设置 y 轴范围
    plt.ylim([-2, 2])  
    # 设置 x 轴标签
    plt.xlabel('x0')  
    # 设置 y 轴标签
    plt.ylabel('x1')  
    # 显示网格
    plt.grid()  
    # 绘制图形
    plt.draw()  
    # 显示图形
    plt.show()  
