# coding: utf-8
import numpy as np


# 定义一个名为 NAND 的函数，实现 NAND 逻辑门
def NAND(x1, x2):
    # 将输入 x1 和 x2 转换为 numpy 数组
    x = np.array([x1, x2])
    
    # 定义权重 w 和偏置 b
    w = np.array([-0.5, -0.5])
    b = 0.7
    
    # 计算加权和并加上偏置
    tmp = np.sum(w*x) + b
    
    # 如果加权和小于等于 0，返回 0；否则返回 1
    if tmp <= 0:
        return 0
    else:
        return 1

# 如果该脚本作为主程序运行，则执行以下代码
if __name__ == '__main__':
    # 遍历所有可能的输入组合
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        # 调用 NAND 函数并输出结果
        y = NAND(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
