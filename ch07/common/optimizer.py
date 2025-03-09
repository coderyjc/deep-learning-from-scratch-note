# coding: utf-8
import numpy as np
# 定义一个简单的随机梯度下降（SGD）优化器
class SGD:

    # 初始化方法，设置学习率 lr，默认值为 0.01
    def __init__(self, lr=0.01):
        self.lr = lr
        
    # 更新参数的方法，根据梯度 grads 更新参数 params
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key] 


# 定义一个动量（Momentum）优化器
class Momentum:

    """Momentum SGD"""

    # 初始化方法，设置学习率 lr 和动量 momentum，默认值分别为 0.01 和 0.9
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    # 更新参数的方法，根据梯度 grads 更新参数 params，并考虑动量
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():                                
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] 
            params[key] += self.v[key]


# 定义一个Nesterov加速梯度（Nesterov's Accelerated Gradient）优化器
class Nesterov:

    """Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)"""

    # 初始化方法，设置学习率 lr 和动量 momentum，默认值分别为 0.01 和 0.9
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    # 更新参数的方法，根据梯度 grads 更新参数 params，并考虑Nesterov加速
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
            
        for key in params.keys():
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]


# 定义一个AdaGrad优化器
class AdaGrad:

    """AdaGrad"""

    # 初始化方法，设置学习率 lr，默认值为 0.01
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    # 更新参数的方法，根据梯度 grads 更新参数 params，并考虑自适应学习率
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


# 定义一个RMSprop优化器
class RMSprop:

    """RMSprop"""

    # 初始化方法，设置学习率 lr 和衰减率 decay_rate，默认值分别为 0.01 和 0.99
    def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        
    # 更新参数的方法，根据梯度 grads 更新参数 params，并考虑RMSprop算法
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


# 定义一个Adam优化器
class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    # 初始化方法，设置学习率 lr、beta1 和 beta2，默认值分别为 0.001、0.9 和 0.999
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    # 更新参数的方法，根据梯度 grads 更新参数 params，并考虑Adam算法
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
