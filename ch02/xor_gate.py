# coding: utf-8
from and_gate import AND
from or_gate import OR
from nand_gate import NAND


# 定义一个 XOR 函数，用于实现逻辑异或操作
def XOR(x1, x2):
    # 使用 NAND 函数计算 x1 和 x2 的 NAND 结果，赋值给 s1
    s1 = NAND(x1, x2)
    # 使用 OR 函数计算 x1 和 x2 的 OR 结果，赋值给 s2
    s2 = OR(x1, x2)
    # 使用 AND 函数计算 s1 和 s2 的 AND 结果，赋值给 y
    y = AND(s1, s2)
    # 返回最终的异或结果 y
    return y

# 判断当前模块是否为主程序，如果是则执行以下代码
if __name__ == '__main__':
    # 遍历所有可能的输入组合 (0, 0), (1, 0), (0, 1), (1, 1)
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        # 调用 XOR 函数计算当前输入组合的异或结果
        y = XOR(xs[0], xs[1])
        # 打印输入组合及其对应的异或结果
        print(str(xs) + " -> " + str(y))
