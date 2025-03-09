# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np

# 定义MNIST数据集的下载地址，使用镜像站点
url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'

# 定义MNIST数据集中各个文件的名称
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

# 获取当前脚本所在的目录，并定义保存MNIST数据集的pickle文件路径
dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

# 定义训练集和测试集的数量，以及图像的维度和大小
train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

# 下载指定的文件
def _download(file_name):
    file_path = dataset_dir + "/" + file_name

    # 如果文件已经存在，则直接返回
    if os.path.exists(file_path):
        return

    # 下载文件并保存到指定路径
    print("Downloading " + file_name + " ... ")
    headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0"}
    request = urllib.request.Request(url_base+file_name, headers=headers)
    response = urllib.request.urlopen(request).read()
    with open(file_path, mode='wb') as f:
        f.write(response)
    print("Done")

# 下载所有MNIST数据集文件
def download_mnist():
    for v in key_file.values():
       _download(v)

# 加载标签数据并将其转换为NumPy数组
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels

# 加载图像数据并将其转换为NumPy数组
def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")

    return data

# 将MNIST数据集转换为NumPy数组并存储在字典中
def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset

# 初始化MNIST数据集，包括下载和转换数据
def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

# 将标签数据转换为one-hot编码格式
def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

# 加载MNIST数据集，并可选地进行归一化、扁平化和one-hot编码
def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    
    # 如果pickle文件不存在，则初始化MNIST数据集
    if not os.path.exists(save_file):
        init_mnist()

    # 从pickle文件中加载数据集
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    # 如果normalize为True，则将图像数据归一化到0-1之间
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    # 如果one_hot_label为True，则将标签数据转换为one-hot编码格式
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    # 如果flatten为False，则将图像数据重塑为4D张量
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    # 返回训练集和测试集
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

# 如果脚本作为主程序运行，则初始化MNIST数据集
if __name__ == '__main__':
    init_mnist()
