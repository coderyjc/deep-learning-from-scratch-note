# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

#  numerical_diff，用于计算函数 f 在点 x 处的数值导数
def numerical_diff(f, x):
    # 定义一个很小的值 h，用于计算导数的近似值
    h = 1e-4
    # 使用中心差分法计算导数并返回结果
    return (f(x+h) - f(x-h)) / (2*h)

#  function_1，表示一个简单的二次函数
def function_1(x):
    return 0.01*x**2 + 0.1*x 

#  tangent_line，用于计算函数 f 在点 x 处的切线
def tangent_line(f, x):
    # 调用 numerical_diff 函数计算函数 f 在点 x 处的导数 d
    d = numerical_diff(f, x)
    # 打印导数值
    print(d)
    # 计算切线的截距 y
    y = f(x) - d*x
    # 返回一个 lambda 函数，表示切线方程
    return lambda t: d*t + y

# 生成一个从 0.0 到 20.0 的数组 x，步长为 0.1
x = np.arange(0.0, 20.0, 0.1)
# 计算函数 function_1 在数组 x 上的值
y = function_1(x)
# 设置 x 轴标签
plt.xlabel("x")
# 设置 y 轴标签
plt.ylabel("f(x)")

# 计算函数 function_1 在 x=5 处的切线
tf = tangent_line(function_1, 5)
# 计算切线在数组 x 上的值
y2 = tf(x)

# 绘制函数 function_1 的图像
plt.plot(x, y)
# 绘制切线的图像
plt.plot(x, y2)
# 显示图像
plt.show()
