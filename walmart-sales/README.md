# Walmart 周销售额预测（PyTorch 线性回归）

> **源课题（Kaggle）**：<https://www.kaggle.com/datasets/mikhail1681/walmart-sales>

根据门店、节假日、温度、油价、CPI、失业率等信息，用**线性回归**预测 Walmart 某门店当周的销售额（`Weekly_Sales`，美元）。

这是一个「**回归**」任务，和 `admission-predict` 同类型；与 `titanic-logreg` 的「分类」任务相对。

## 结果

| 数据集 | MAE（美元） | RMSE（美元） | R² |
|---|---|---|---|
| 训练集 | 91,293 | 157,299 | 0.921 |
| 验证集 | 93,314 | 167,356 | 0.920 |
| 测试集 | 91,276 | 159,610 | 0.921 |

- R² = 0.921：模型解释了测试集中约 92.1% 的方差；
- MAE ≈ 9.1 万美元：平均预测误差（周销售额量级为 20 万～380 万美元）。

## 每一步做了什么

### 1. 读数据

```python
df = pd.read_csv("Walmart.csv", parse_dates=["Date"], dayfirst=True)
```

数据共 6435 行、8 列（45 家门店、2010–2012 的周记录）。`dayfirst=True` 是因为日期是 `dd-mm-yyyy` 格式。

### 2. 探索性分析（EDA）

- 周销售额分布直方图；
- 节假日 vs 非节假日箱线图（节假日销售更高）；
- 月度平均销售额趋势（有明显季节性波动）。

### 3. 特征工程

- **目标**：`Weekly_Sales`；
- **数值特征**：`Temperature`、`Fuel_Price`、`CPI`、`Unemployment`；
- **二值特征**：`Holiday_Flag`（0/1）；
- **类别特征**：`Store`（45 家门店）→ 用 `pd.get_dummies(..., drop_first=True)` 展开成 44 个 0/1 虚拟变量。

最终特征维度 = 44（门店）+ 1（节假日）+ 4（数值）= 49 维。

### 4. 划分 训练 / 验证 / 测试 = 60 / 20 / 20

与 `admission-predict` 相同：先切 20% 测试集，再从剩余 80% 切 25% 作验证集。

### 5. 标准化（数值特征 + 目标）

- 只对 **4 个数值特征**做 `StandardScaler`，且只在训练集 `fit`；
- 门店虚拟变量和 `Holiday_Flag` 本身是 0/1，**不需要标准化**；
- 目标 `Weekly_Sales` 数值很大（百万级），也做标准化，评估时再 `inverse_transform` 还原成美元。

### 6. 转 PyTorch 张量

`torch.tensor(np.asarray(a, dtype=np.float32))`；标签 `view(-1, 1)` 对齐形状。

### 7. 模型 / 损失 / 优化器

```python
model = nn.Linear(49, 1)              # 49 维特征 → 1 个输出
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
```

### 8. 训练

600 个 epoch：前向 → 算 loss → `backward()` → `optimizer.step()`；每轮用验证集算一次 loss，**保留验证集最佳模型**。

### 9. 评估

把预测值 `inverse_transform` 回美元后，计算 **MAE / RMSE / R²**。

### 10. 图表与保存

损失曲线、预测 vs 真实、残差分布、指标对比、特征权重，并 `torch.save` 保存模型权重。

## 关键发现

- **门店是最大的决定因素**：只用 5 个外部因素时 R² 只有 0.02；加入门店 one-hot 后升到 0.921。
- 经济/天气因素（标准化后权重）：
  - **CPI +**（物价高 → 名义销售额高）；
  - **节假日 +**（约 +7.5 万美元/周）；
  - **失业率 −**、**油价 −**、**温度 −**（均为负向）。

## 线性回归知识点（进阶）

### 模型

与 `admission-predict` 一致：`y = w1·x1 + ... + w49·x49 + b`。

### 类别特征与 one-hot 编码

- 不能直接把 `Store` 当数值用：`45` 并不代表“比 `1` 多 44”，门店之间没有大小关系；
- **one-hot**：每个类别一列，取值为 0/1；
- `drop_first=True`：丢掉第一列，避免“虚拟变量陷阱”（所有虚拟变量之和恒为 1 导致的共线性）。

### 门店固定效应（fixed effects）

每个 `Store` 虚拟变量的系数，可以理解为「该门店相对基准门店的销售额水平」。这也是本模型 R² 很高的原因——门店差异解释了大部分销售额波动。

### 目标也做标准化

当目标量级很大（百万级）时，直接训练会让梯度数值巨大、不稳定。把 `y` 也标准化，训练完再还原，是常见做法。

### Adam 与 SGD

- SGD：固定学习率，简单直接；
- **Adam**：自适应地为每个参数调整学习率，通常收敛更快、对学习率不那么敏感，适合特征多/量级差异大的场景。

### 其余知识点

损失函数 MSE、梯度下降、Epoch/Batch、过拟合与验证集、MAE/MSE/RMSE/R²、标准化的原理，见 `admission-predict/README.md` 的「线性回归知识点」一节。

## 运行

```bash
pip install -r requirements.txt
python walmart_lr.py
```

## 文件结构

```
walmart-sales/
├── Walmart.csv
├── walmart_lr.py
├── requirements.txt
├── README.md
├── walmart_lr_model.pt
└── output/          # 7 张结果图
```
