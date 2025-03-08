# coding: utf-8

# 定义一个名为 Man 的类
class Man:

    # 定义初始化方法 __init__，在创建 Man 实例时自动调用
    def __init__(self, name):
        # 将传入的 name 参数赋值给实例属性 self.name
        self.name = name
        # 打印初始化完成的消息
        print("Initialized!")

    # 定义 hello 方法，用于打印问候语
    def hello(self):
        # 打印包含实例属性 self.name 的问候语
        print("Hello " + self.name + "!")

    # 定义 goodbye 方法，用于打印告别语
    def goodbye(self):
        # 打印包含实例属性 self.name 的告别语
        print("Good-bye " + self.name + "!")

# 创建 Man 类的一个实例，传入参数 "David"，并将其赋值给变量 m
m = Man("David")

# 调用实例 m 的 hello 方法，打印问候语
m.hello()

# 调用实例 m 的 goodbye 方法，打印告别语
m.goodbye()