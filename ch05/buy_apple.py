# coding: utf-8
from layer_naive import *


# 定义苹果的单价
apple = 100
# 定义苹果的数量
apple_num = 2
# 定义税率
tax = 1.1

# 创建一个乘法层的实例，用于计算苹果总价
mul_apple_layer = MulLayer()
# 创建另一个乘法层的实例，用于计算含税总价
mul_tax_layer = MulLayer()
# 前向传播：计算苹果的总价
apple_price = mul_apple_layer.forward(apple, apple_num)
# 前向传播：计算含税总价
price = mul_tax_layer.forward(apple_price, tax)
# 反向传播：初始化梯度为1
dprice = 1

# 反向传播：计算苹果总价和税率的梯度
dapple_price, dtax = mul_tax_layer.backward(dprice)
# 反向传播：计算苹果单价和数量的梯度
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

# 输出含税总价
print("price:", int(price))
# 输出苹果单价的梯度
print("dApple:", dapple)
# 输出苹果数量的梯度
print("dApple_num:", int(dapple_num))
# 输出税率的梯度
print("dTax:", dtax)
