> 原著GitHub仓库：https://github.com/oreilly-japan/deep-learning-from-scratch

# 深度学习入门：基于Python的理论和实现

<div align=center>
<img src="./img/cover.jpg" width="200px">
</div>

本仓库从原著的github仓库fork而来，编写本书代码的中文注释。

**本仓库的所有代码都进行了详细的注释，并使用VsCode测试全部代码，可运行。帮助读者更好的入门深度学习。**

**注意：本项目基于VsCode修改，运行时请务必以`deep-learning-from-scratch-note`为项目的根目录，否则会造成路径相关的错误。**

## 目录

|目录 |说明                    |
|:--        |:--              |
|ch01       |第一章相关代码    |
|ch02       |第二章相关代码    |
|...        |...              |
|ch08       |第八章相关代码    |
|common     |共用的代码   |
|dataset    |数据集的代码 |
|notebooks    |ipynb格式的代码（jupyter notebook 使用） |

## 环境

### IDE

VsCode

**关于VsCode中项目结构和导包路径的说明**

VsCode中打开`deep-learning-from-scratch-note`作为项目根目录，则在任意`.py`文件中使用路径的时候，`./`指的是项目根目录，而不是相对于当前`.py`文件的目录。

因此，想要在`./ch*/*.py`中导入`common`和`dataset`中的包的时候，需要将项目的根目录加载到`sys.path`中：

```path
import sys
sys.path.append('.') 
```

### ENV

- Python 3.x
- NumPy
- Matplotlib

pip安装方式

```bash
conda install numpy matplotlib
```

conda安装方式

```bash
pip install numpy matplotlib
```

## 相较于原版的修改

【Update】
- 删除所有日文注释，为所有代码添加详细的**中文**注释。
- 将lena.png图片替换为本书的封面，以避免侵权。
- 本项目使用vscode编写，所有的路径都是以当前打开的路径为基准的，也就是说'./'目录为'项目根目录'

【Delete】
- 删除notebooks中的注释

## 致谢

- [deep-learning-from-scratch](https://github.com/oreilly-japan/deep-learning-from-scratch)
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)