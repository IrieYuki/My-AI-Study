# 研究生录取概率预测（PyTorch 线性回归）

根据申请者的 GRE、TOEFL、本科 GPA、研究经历等 7 项信息，预测其被美国研究生院录取的概率。

数据：Kaggle [Graduate Admissions](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions)（400 条 × 7 特征）。

## 结果

| 数据集 | MAE | RMSE | R² |
|---|---|---|---|
| 训练集 | 0.040 | 0.057 | 0.827 |
| 验证集 | 0.055 | 0.076 | 0.692 |
| **测试集** | **0.048** | **0.069** | **0.815** |

> R² = 0.815 表示模型解释了测试集中约 81.5% 的方差；MAE 0.048 表示平均预测误差约 4.8 个百分点。

## 特征权重（标准化后）

`CGPA` 权重最高（约 0.069），其次 `GRE`、`TOEFL`、`Research`、`LOR`、`SOP`、`University Rating`。

结论与常识一致：**本科 GPA 对录取影响最大，其次是标准化考试成绩，研究经历有正向作用。**

## 流水线（八步）

1. 读数据 → 2. 探索性分析（特征分布）→ 3. 特征/目标拆分
4. 划分训练/验证/测试 = 60/20/20 → 5. 标准化（只在训练集 fit）
6. 转 PyTorch 张量 → 7. 定义 `nn.Linear(7,1)` + MSE + SGD
8. 训练 500 轮并跟踪最佳验证集 → 评估 MAE/MSE/RMSE/R² + 特征权重

## 怎么跑

```bash
pip install -r requirements.txt
python admission_lr.py
```

输出：
- 终端打印各数据集 MAE / RMSE / R²
- `output/` 下 6 张图（分布、Loss 曲线、预测 vs 真实、残差、指标对比、特征权重）
- `linear_regression_model.pt` 模型权重

## 文件结构

```
admission-predict/
├── Admission_Predict.csv        # 数据（400 条 × 9 列）
├── LR_jupyter.ipynb             # 原始 notebook（分段讲解版）
├── admission_lr.py              # 干净可复现脚本
├── requirements.txt
├── README.md
├── linear_regression_model.pt   # 训练好的权重
└── output/                      # 6 张结果图
```

## 面试 30 秒话术

> “这是我做的研究生录取概率预测：用 PyTorch 实现线性回归，输入 GRE、GPA、研究经历等 7 个特征，
> 60/20/20 划分训练/验证/测试集并做标准化，测试集 R² 达到 0.815，平均误差约 4.8 个百分点。
> 特征权重显示 GPA 影响最大，其次是 GRE/TOEFL，研究经历有正向作用——这和申请常识一致。”

可能被追问的回答要点：
- **为什么用线性回归**：目标是 0~1 的连续概率，线性模型可解释，适合入门验证流程。
- **为什么 60/20/20**：验证集用于每一轮观察过拟合、选最佳模型；测试集只在最后用一次。
- **为什么标准化**：特征量纲差别大（GRE 300 上下、CGPA 8 上下），标准化让梯度下降更稳定。
- **R² 是什么**：决定系数，越接近 1 表示模型解释的方差越多。
