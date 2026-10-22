# 深度学习学习记录

这是一个按主题整理的深度学习学习仓库，记录从基础回归、多层感知机到 CNN、RNN、Transformer，以及一个房价预测实战项目的代码和实践过程。

所有脚本都保留中文注释，便于复习算法细节；目录和文件名使用统一的英文命名，方便检索、引用和后续扩展。

## 目录结构

```text
.
├── 01_foundations/       # 线性回归、Softmax、MLP、Dropout、正则化
├── 02_cnn/               # LeNet、ResNet、Fashion-MNIST 数据探索
├── 03_rnn/               # RNN、LSTM、GRU 从零实现
├── 04_transformer/       # PyTorch Transformer 源码学习材料
├── 05_projects/
│   └── house_price_prediction/
├── data/                 # 运行时自动下载的数据集（不提交）
├── requirements.txt
└── README.md
```

## 内容导航

| 模块 | 文件 | 内容 |
| --- | --- | --- |
| 基础 | `01_foundations/linear_regression_scratch.py` | 从零实现线性回归和 SGD |
| 基础 | `01_foundations/linear_regression_concise.py` | 使用 PyTorch API 实现线性回归 |
| 基础 | `01_foundations/polynomial_regression.py` | 多项式回归与欠拟合/过拟合 |
| 基础 | `01_foundations/softmax_regression_scratch.py` | 从零实现 Softmax 回归 |
| 基础 | `01_foundations/softmax_regression_concise.py` | Softmax 回归的简洁实现 |
| 基础 | `01_foundations/mlp_scratch.py` | 从零实现多层感知机 |
| 基础 | `01_foundations/mlp_concise.py` | 多层感知机的简洁实现 |
| 基础 | `01_foundations/dropout_scratch.py` | 从零实现 Dropout |
| 基础 | `01_foundations/dropout_concise.py` | Dropout 的简洁实现 |
| 基础 | `01_foundations/weight_decay.py` | L2 正则化与权重衰减 |
| CNN | `02_cnn/lenet5.py` | LeNet-5 图像分类 |
| CNN | `02_cnn/resnet18.py` | ResNet 残差块与网络 |
| CNN | `02_cnn/fashion_mnist_exploration.py` | Fashion-MNIST 数据加载和可视化 |
| RNN | `03_rnn/rnn_scratch.py` | 循环神经网络从零实现 |
| RNN | `03_rnn/lstm_scratch.py` | LSTM 从零实现 |
| RNN | `03_rnn/gru_scratch.py` | GRU 从零实现 |
| Transformer | `04_transformer/pytorch_transformer_reference.py` | PyTorch Transformer 源码学习材料 |
| 实战 | `05_projects/house_price_prediction/` | 房价预测特征工程、训练和提交 |

## 快速开始

建议使用 Python 3.10 或更高版本。

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

安装 GPU 版 PyTorch 时，请根据自己的 CUDA 环境选择官方对应的安装命令，再安装其余依赖。

### 基础与 CNN

部分 D2L 教学脚本会按相对路径缓存数据集。进入对应章节目录后再运行：

```powershell
Set-Location 01_foundations
python linear_regression_scratch.py

Set-Location ..\02_cnn
python lenet5.py
```

### RNN、LSTM 与 GRU

三个脚本共享相同的命令行参数。以 LSTM 为例：

```powershell
Set-Location 03_rnn
python lstm_scratch.py --num-epochs 100
```

常用参数包括 `--batch-size`、`--num-steps`、`--num-hiddens`、`--num-epochs`、`--lr` 和 `--device`。训练曲线默认保存为对应的 `*.png`。

### 房价预测项目

```powershell
python 05_projects/house_price_prediction/train_hist_gradient_boosting.py
```

脚本会读取项目内的训练/测试数据，并把新生成的提交文件写入该项目的 `outputs/` 目录。项目说明和数据字段见 `05_projects/house_price_prediction/README.md`。

## 数据与版本控制

- `data/` 中的数据缓存、模型权重、虚拟环境和 Python 缓存不会提交。
- 教学脚本中的图表和训练日志属于可再生成文件，建议保留在本地。
- 房价预测项目的 `submission.csv` 是已提交的历史结果；重新运行脚本时不会覆盖它。

## 关于 Transformer 源码

`04_transformer/pytorch_transformer_reference.py` 是用于阅读 PyTorch Transformer 实现的教学材料，内部使用了 PyTorch 包的相对导入，不能像普通示例脚本一样单独执行。对应的 Transformer 原论文是 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。
