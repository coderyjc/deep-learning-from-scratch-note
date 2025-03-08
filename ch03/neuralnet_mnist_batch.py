# coding: utf-8
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


# 函数 get_data，用于加载 MNIST 数据集
def get_data():
    # 使用 load_mnist 函数加载 MNIST 数据集，并进行归一化和展平处理
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    # 返回测试集的输入数据和标签
    return x_test, t_test


# 函数 init_network，用于初始化神经网络
def init_network():
    # 打开存储网络权重的文件，以二进制读取模式
    with open("ch03\\sample_weight.pkl", 'rb') as f:
        # 使用 pickle 加载文件中的网络权重
        network = pickle.load(f)
    # 返回加载的网络权重
    return network


# 函数 predict，用于进行神经网络的预测
def predict(network, x):
    # 从网络权重中提取各层的权重和偏置
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 计算第一层的激活值
    a1 = np.dot(x, w1) + b1
    # 对第一层的激活值应用 sigmoid 函数
    z1 = sigmoid(a1)
    # 计算第二层的激活值
    a2 = np.dot(z1, w2) + b2
    # 对第二层的激活值应用 sigmoid 函数
    z2 = sigmoid(a2)
    # 计算第三层的激活值
    a3 = np.dot(z2, w3) + b3
    # 对第三层的激活值应用 softmax 函数，得到最终的输出
    y = softmax(a3)

    # 返回预测结果
    return y


# 获取测试数据
x, t = get_data()
# 初始化神经网络
network = init_network()

# 设置批处理大小
batch_size = 100
# 初始化准确率计数器
accuracy_cnt = 0

# 对测试数据进行批处理预测
for i in range(0, len(x), batch_size):
    # 获取当前批次的输入数据
    x_batch = x[i: i + batch_size]
    # 对当前批次进行预测
    y_batch = predict(network, x_batch)
    # 获取预测结果中概率最大的类别
    p = np.argmax(y_batch, axis=1)
    # 计算当前批次的准确率，并累加到准确率计数器中
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

# 计算并打印整体准确率
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
