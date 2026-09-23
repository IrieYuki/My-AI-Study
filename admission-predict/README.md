# 研究生录取概率预测（PyTorch 线性回归）

根据申请者的 GRE、TOEFL、本科 GPA、研究经历等 7 项信息，预测其被美国研究生院录取的概率。

数据：Kaggle [Graduate Admissions](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions)（400 条样本）。

## 结果

| 数据集 | MAE | RMSE | R² |
|---|---|---|---|
| 训练集 | 0.040 | 0.057 | 0.827 |
| 验证集 | 0.055 | 0.076 | 0.692 |
| 测试集 | 0.048 | 0.069 | 0.815 |

测试集 R² = 0.815，表示模型解释了约 81.5% 的方差；MAE 0.048 表示平均预测误差约 4.8 个百分点。

## 特征权重

标准化后的权重由大到小：`CGPA`（约 0.069）、`GRE Score`、`TOEFL Score`、`Research`、`LOR`、`SOP`、`University Rating`。

即：本科 GPA 对录取影响最大，其次是标准化考试成绩，研究经历有正向作用。

## 流水线

1. 读数据 → 2. 探索性分析 → 3. 特征 / 目标拆分
4. 划分训练 / 验证 / 测试 = 60 / 20 / 20 → 5. 标准化（只在训练集 fit）
6. 转 PyTorch 张量 → 7. `nn.Linear(7, 1)` + MSE + SGD
8. 训练 500 轮并跟踪最佳验证集 → 评估 MAE / MSE / RMSE / R² + 特征权重

## 运行

```bash
pip install -r requirements.txt
python admission_lr.py
```

## 文件结构

```
admission-predict/
├── Admission_Predict.csv
├── LR_jupyter.ipynb
├── admission_lr.py
├── requirements.txt
├── README.md
├── linear_regression_model.pt
└── output/          # 6 张结果图
```
