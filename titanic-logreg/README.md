# Titanic 存活预测（逻辑回归）

一个最小可复现的机器学习入门项目：根据乘客的舱位、性别、年龄、票价等信息，
用 **逻辑回归（Logistic Regression）** 预测其在泰坦尼克号上是否幸存。

> 一次同时展示三件事：**Python（数据处理）+ 数据（pandas/matplotlib）+ AI（sklearn 建模）**。

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
- 结论与史实一致：**女性和高舱位乘客优先获救**。

## 流水线（五步）

1. **读数据**：`pd.read_csv("titanic.csv")`
2. **预处理**：年龄缺失用中位数填补；`sex`/`embarked` 用 `get_dummies` 转成 0/1
3. **划分**：`train_test_split(test_size=0.2, random_state=42, stratify=y)`
4. **训练**：`LogisticRegression(max_iter=1000).fit(...)`
5. **评估**：准确率 + 混淆矩阵热力图 + 特征重要性条形图

## 怎么跑

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

## 面试 30 秒话术

> “这是我做的一个 Titanic 存活预测：891 条乘客数据，先做缺失值处理和特征编码，
> 再用逻辑回归建模，测试集准确率 81.5%，比随机基线高约 20 个百分点。
> 模型还告诉我们，性别和舱位等级是影响存活最重要的两个因素，这和历史上
> ‘女士与高舱位优先获救’是一致的。”

可能被追问的回答要点：
- **为什么用逻辑回归**：二分类、可解释（系数有实际含义），适合入门验证流程。
- **`random_state` 是干嘛的**：固定随机种子，让划分和结果可复现。
- **为什么填中位数而不是删**：缺 177 行，删了损失约 20% 样本；中位数对异常票价更稳健。
