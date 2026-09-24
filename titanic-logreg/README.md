# Titanic 存活预测（逻辑回归）

一个最小可复现的机器学习**二分类**项目：根据乘客的舱位、性别、年龄、票价等信息，用**逻辑回归（Logistic Regression）**预测其在泰坦尼克号上是否幸存。

> 数据（pandas）+ 建模（sklearn）+ 可视化（matplotlib/seaborn）三合一。

## 这是什么任务：分类，不是回归

逻辑回归名字里带“回归”，但它解决的是**分类**问题（本项目的 `Survived` 只有 0 / 1 两个取值）。

| | 线性回归 Linear Regression | 逻辑回归 Logistic Regression |
|---|---|---|
| 任务类型 | 回归：预测连续数值 | 分类：预测类别（0/1） |
| 输出 | 任意实数 | 0~1 之间的概率 |
| 核心公式 | y = wx + b | p = sigmoid(wx + b) |
| 损失函数 | MSE（均方误差） | 交叉熵 Cross-Entropy |
| 评估指标 | MAE / RMSE / R² | 准确率 / 混淆矩阵 / 精确率 / 召回率 |
| 对应项目 | admission-predict、walmart-sales | **本项目** |

## 为什么分类不能用线性回归

线性回归的输出是任意实数——可能预测出 `1.3` 或 `-0.2` 这种无意义的“存活概率”。逻辑回归在线性输出外面再套一层 **sigmoid**：

```
sigmoid(z) = 1 / (1 + e^-z)
```

它把任意实数压到 `(0, 1)` 区间，得到一个真正的概率；再取阈值 `0.5` 完成分类。

```mermaid
flowchart LR
    A["特征 x<br/>Pclass, Sex, Age, Fare…"] --> B["线性组合<br/>z = w·x + b"]
    B --> C["Sigmoid<br/>p = 1 / (1 + e^-z)"]
    C --> D["概率 p ∈ (0,1)"]
    D --> E{"p ≥ 0.5 ?"}
    E -- 是 --> F["预测：存活 (1)"]
    E -- 否 --> G["预测：死亡 (0)"]
```

## 损失函数：为什么分类用交叉熵而不是 MSE

- MSE 在概率预测上容易出现**梯度趋近 0**，导致学习变慢；
- 交叉熵对概率输出梯度更陡、收敛更快，而且天然适合 0/1 标签。

## 结果

| 指标 | 数值 |
|---|---|
| 测试集准确率 | **0.815**（178 个样本） |
| 随机基线（全猜“死亡”） | 0.618 |
| 相比基线提升 | **约 +20 个百分点** |

混淆矩阵（真实 \ 预测）：

| | 预测 Died | 预测 Survived |
|---|---|---|
| **实际 Died** | 98 | 12 |
| **实际 Survived** | 21 | 47 |

特征重要性（逻辑回归系数）：
- `sex_male = -2.55`：男性显著降低存活概率（绝对值最大）
- `pclass = -1.02`：舱位等级越低，存活概率越低
- 结论与史实一致：**女性和高舱位乘客优先获救**

## 流水线（五步）

1. **读数据**：`pd.read_csv("titanic.csv")`
2. **预处理**：年龄缺失用中位数填补；`sex`/`embarked` 用 `get_dummies` 转成 0/1
3. **划分**：`train_test_split(test_size=0.2, random_state=42, stratify=y)`
4. **训练**：`LogisticRegression(max_iter=1000).fit(...)`
5. **评估**：准确率 + 混淆矩阵热力图 + 特征重要性条形图

## 运行

```bash
pip install -r requirements.txt
python titanic_logreg.py
```

输出：
- 终端打印准确率
- `output/confusion_matrix.png` —— 混淆矩阵热力图
- `output/feature_importance.png` —— 特征重要性条形图

## 文件结构

```
titanic-logreg/
├── titanic.csv                # 数据（891 行 × 8 列）
├── titanic_logreg.py          # 主脚本
├── requirements.txt           # 依赖
├── README.md
└── output/
    ├── confusion_matrix.png
    └── feature_importance.png
```
