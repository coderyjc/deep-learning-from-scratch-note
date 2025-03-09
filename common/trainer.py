# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.optimizer import *

# 定义一个名为 Trainer 的类，用于训练神经网络
class Trainer:
    # 初始化方法，设置训练所需的参数
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, 
                 evaluate_sample_num_per_epoch=None, verbose=True):
        # 设置神经网络模型
        self.network = network
        # 设置是否打印训练过程的详细信息
        self.verbose = verbose
        # 设置训练数据和标签
        self.x_train = x_train
        self.t_train = t_train
        # 设置测试数据和标签
        self.x_test = x_test
        self.t_test = t_test
        # 设置训练的轮数
        self.epochs = epochs
        # 设置每个小批次的样本数量
        self.batch_size = mini_batch_size
        # 设置每轮评估时使用的样本数量
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # 定义优化器类字典，根据输入的优化器名称选择相应的优化器
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprop':RMSprop, 'adam':Adam}
        # 初始化优化器
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        # 计算训练集的大小
        self.train_size = x_train.shape[0]
        # 计算每轮的迭代次数
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        # 计算最大迭代次数
        self.max_iter = int(epochs * self.iter_per_epoch)
        # 初始化当前迭代次数和当前轮数
        self.current_iter = 0
        self.current_epoch = 0
        
        # 初始化训练损失、训练准确率和测试准确率的列表
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    # 定义单步训练方法
    def train_step(self):
        # 随机选择一个小批次的样本
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        # 获取小批次的训练数据和标签
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        # 计算梯度
        grads = self.network.gradient(x_batch, t_batch)
        # 使用优化器更新网络参数
        self.optimizer.update(self.network.params, grads)
        
        # 计算当前小批次的损失
        loss = self.network.loss(x_batch, t_batch)
        # 将损失添加到训练损失列表中
        self.train_loss_list.append(loss)
        # 如果 verbose 为 True，打印当前损失
        if self.verbose: print("train loss:" + str(loss))
        
        # 如果当前迭代次数是每轮迭代次数的整数倍，进行一轮评估
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            # 获取用于评估的训练和测试样本
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            # 如果设置了每轮评估的样本数量，则截取相应数量的样本
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
                
            # 计算训练和测试的准确率
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            # 将准确率添加到相应的列表中
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            # 如果 verbose 为 True，打印当前轮数、训练准确率和测试准确率
            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
        # 更新当前迭代次数
        self.current_iter += 1

    # 定义训练方法，执行完整的训练过程
    def train(self):
        # 进行最大迭代次数的训练
        for i in range(self.max_iter):
            self.train_step()

        # 计算最终的测试准确率
        test_acc = self.network.accuracy(self.x_test, self.t_test)

        # 如果 verbose 为 True，打印最终的测试准确率
        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))
