# coding: utf-8
import numpy as np


# 定义一个名为 OR 的函数，模拟逻辑或操作
def OR(x1, x2):
    # 将输入 x1 和 x2 转换为 numpy 数组
    x = np.array([x1, x2])
    
    # 定义权重数组 w
    w = np.array([0.5, 0.5])
    
    # 定义偏置值 b
    b = -0.2
    
    # 计算加权和并加上偏置值
    tmp = np.sum(w*x) + b
    
    # 如果加权和小于等于 0，返回 0
    if tmp <= 0:
        return 0
    # 否则返回 1
    else:
        return 1

# 如果该脚本作为主程序运行
if __name__ == '__main__':
    # 遍历所有可能的输入组合
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        # 调用 OR 函数计算结果
        y = OR(xs[0], xs[1])
        # 打印输入和输出
        print(str(xs) + " -> " + str(y))
