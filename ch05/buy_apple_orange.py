# coding: utf-8
from layer_naive import *

# 苹果的单价
apple = 100
# 苹果的数量
apple_num = 2
# 橙子的单价
orange = 150
# 橙子的数量
orange_num = 3
# 税率
tax = 1.1

# 创建乘法层对象，用于计算苹果和橙子的总价
mul_apple_layer = MulLayer()
# 创建乘法层对象，用于计算橙子的总价
mul_orange_layer = MulLayer()
# 创建加法层对象，用于计算苹果和橙子的总价之和
add_apple_orange_layer = AddLayer()
# 创建乘法层对象，用于计算含税的总价
mul_tax_layer = MulLayer()

# 计算苹果的总价
apple_price = mul_apple_layer.forward(apple, apple_num)  # (1)
# 计算橙子的总价
orange_price = mul_orange_layer.forward(orange, orange_num)  # (2)
# 计算苹果和橙子的总价之和
all_price = add_apple_orange_layer.forward(apple_price, orange_price)  # (3)
# 计算含税的总价
price = mul_tax_layer.forward(all_price, tax)  # (4)

# 初始化反向传播的梯度为1
dprice = 1
# 计算含税总价的梯度
dall_price, dtax = mul_tax_layer.backward(dprice)  # (4)
# 计算苹果和橙子总价之和的梯度
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)  # (3)
# 计算橙子总价的梯度
dorange, dorange_num = mul_orange_layer.backward(dorange_price)  # (2)
# 计算苹果总价的梯度
dapple, dapple_num = mul_apple_layer.backward(dapple_price)  # (1)

# 输出最终含税的总价
print("price:", int(price))
# 输出苹果单价的梯度
print("dApple:", dapple)
# 输出苹果数量的梯度
print("dApple_num:", int(dapple_num))
# 输出橙子单价的梯度
print("dOrange:", dorange)
# 输出橙子数量的梯度
print("dOrange_num:", int(dorange_num))
# 输出税率的梯度
print("dTax:", dtax)
