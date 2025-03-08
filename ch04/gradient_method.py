# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient

# 定义梯度下降函数，用于最小化函数 f
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    # 初始化变量 x 为初始值
    x = init_x
    # 用于存储 x 的历史值，以便可视化轨迹
    x_history = []

    # 进行 step_num 次迭代
    for i in range(step_num):
        # 将当前 x 的值保存到历史记录中
        x_history.append( x.copy() )

        # 计算函数 f 在 x 处的梯度
        grad = numerical_gradient(f, x)
        # 更新 x 的值：x = x - 学习率 * 梯度
        x -= lr * grad

    # 返回最终的 x 值和历史记录
    return x, np.array(x_history)

# 定义一个简单的二次函数 f(x) = x[0]^2 + x[1]^2
def function_2(x):
    return x[0]**2 + x[1]**2

# 初始化 x 的值为 [-3.0, 4.0]
init_x = np.array([-3.0, 4.0])    

# 设置学习率（步长）
lr = 0.1
# 设置迭代次数
step_num = 20
# 调用梯度下降函数，获取最终的 x 值和历史记录
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

# 绘制 x0 轴（水平虚线）
plt.plot( [-5, 5], [0,0], '--b')
# 绘制 x1 轴（垂直虚线）
plt.plot( [0,0], [-5, 5], '--b')
# 绘制 x 的历史轨迹，用圆圈表示
plt.plot(x_history[:,0], x_history[:,1], 'o')

# 设置 x 轴的范围
plt.xlim(-3.5, 3.5)
# 设置 y 轴的范围
plt.ylim(-4.5, 4.5)
# 设置 x 轴标签
plt.xlabel("X0")
# 设置 y 轴标签
plt.ylabel("X1")
# 显示图形
plt.show()
