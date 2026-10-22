# 房价预测项目

该项目使用房屋特征预测 `SalePrice`，包含数据、特征工程、两种回归模型和提交文件生成脚本。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `data/train.csv` | 带目标列的清洗后训练集 |
| `data/test.csv` | 待预测测试集 |
| `features.py` | 共用的特征工程函数 |
| `train_random_forest.py` | 随机森林模型 |
| `train_hist_gradient_boosting.py` | 直方图梯度提升模型 |
| `submission.csv` | 已提交的历史预测结果 |
| `outputs/` | 重新运行后生成的提交文件，不提交到 Git |

## 运行

在仓库根目录执行：

```powershell
python 05_projects/house_price_prediction/train_random_forest.py
python 05_projects/house_price_prediction/train_hist_gradient_boosting.py
```

两个脚本共用 `features.py` 中的特征工程逻辑，并分别写入：

- `outputs/submission_random_forest.csv`
- `outputs/submission_hist_gradient_boosting.csv`

## 特征工程

当前流程包括：

1. 合并训练集与测试集，统一数值列和类别列处理。
2. 中位数填充数值缺失值，用 `None` 填充类别缺失值。
3. 构造总面积、卫生间数量、房龄、翻新时间、门廊面积和设施标记等特征。
4. 对类别特征进行独热编码。
5. 对 `SalePrice` 使用 `log1p` 变换，预测后再用 `expm1` 还原。
