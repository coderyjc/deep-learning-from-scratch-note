# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from deep_convnet import DeepConvNet
from dataset.mnist import load_mnist


# 加载MNIST数据集，并将其分为训练集和测试集
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 初始化一个深度卷积网络
network = DeepConvNet()

# 加载预训练的网络参数
network.load_params("ch08\\deep_convnet_params.pkl")

# 打印提示信息，表示正在计算测试准确率
print("calculating test accuracy ... ")

# 初始化一个空列表，用于存储分类结果
classified_ids = []

# 初始化准确率变量
acc = 0.0

# 设置批量大小
batch_size = 100

# 分批处理测试数据
for i in range(int(x_test.shape[0] / batch_size)):
    # 获取当前批次的测试数据
    tx = x_test[i*batch_size:(i+1)*batch_size]
    tt = t_test[i*batch_size:(i+1)*batch_size]
    
    # 使用网络进行预测
    y = network.predict(tx, train_flg=False)
    
    # 获取预测结果中概率最大的类别
    y = np.argmax(y, axis=1)
    
    # 将预测结果添加到分类结果列表中
    classified_ids.append(y)
    
    # 累加正确分类的数量
    acc += np.sum(y == tt)
    
# 计算并打印测试准确率
acc = acc / x_test.shape[0]
print("test accuracy:" + str(acc))

# 将分类结果列表转换为NumPy数组并展平
classified_ids = np.array(classified_ids)
classified_ids = classified_ids.flatten()

# 设置最大可视化数量
max_view = 20

# 初始化当前可视化数量
current_view = 1

# 创建一个新的图形窗口
fig = plt.figure()

# 调整子图的布局
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2)

# 初始化一个字典，用于存储错误分类的结果
mis_pairs = {}

# 遍历分类结果，找出错误分类的样本
for i, val in enumerate(classified_ids == t_test):
    if not val:
        # 添加一个子图，用于显示错误分类的图像
        ax = fig.add_subplot(4, 5, current_view, xticks=[], yticks=[])
        ax.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
        
        # 将错误分类的标签和预测结果存储到字典中
        mis_pairs[current_view] = (t_test[i], classified_ids[i])
        
        # 更新当前可视化数量
        current_view += 1
        if current_view > max_view:
            break

# 打印错误分类的结果
print("======= misclassified result =======")
print("{view index: (label, inference), ...}")
print(mis_pairs)

# 显示图形窗口
plt.show()
