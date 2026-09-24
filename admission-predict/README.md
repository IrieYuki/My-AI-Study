# 研究生录取概率预测（PyTorch 线性回归）

> **源课题（Kaggle）**：<https://www.kaggle.com/datasets/mohansacharya/graduate-admissions>

根据申请者的 GRE、TOEFL、本科 GPA、研究经历等 7 项信息，用**线性回归**预测其被美国研究生院录取的概率（`Chance of Admit`，0~1 的连续值）。

这是一个「**回归**」任务（预测连续数值），和 `titanic-logreg` 的「分类」任务（逻辑回归）相对。

## 结果

| 数据集 | MAE | RMSE | R² |
|---|---|---|---|
| 训练集 | 0.040 | 0.057 | 0.827 |
| 验证集 | 0.055 | 0.076 | 0.692 |
| 测试集 | 0.048 | 0.069 | 0.815 |

- R² = 0.815：模型解释了测试集中约 81.5% 的方差。
- MAE = 0.048：平均预测误差约 4.8 个百分点。

## 每一步做了什么

### 1. 读数据

```python
df = pd.read_csv("Admission_Predict.csv")
```

数据共 400 行、9 列。

### 2. 探索性分析（EDA）

- 清理列名：`df.columns = [c.strip() for c in df.columns]`（去掉列名里的空格，例如 `"LOR "` → `"LOR"`）。
- 删除无意义列：`Serial No.` 只是学生编号，对预测没有作用。
- 对每一列画分布直方图（`sns.histplot(..., kde=True)`），直观了解取值范围和大致分布。

### 3. 特征与目标

- **特征 X（7 个）**：`GRE Score`、`TOEFL Score`、`University Rating`、`SOP`、`LOR`、`CGPA`、`Research`
- **目标 y**：`Chance of Admit`（录取概率，连续值）

### 4. 划分 训练 / 验证 / 测试 = 60 / 20 / 20

分两步切：

1. 先从全部数据切出 20% 作为**测试集**；
2. 再从剩下的 80% 里切 25% 作为**验证集**（80% × 25% = 20%）。

| 数据集 | 作用 |
|---|---|
| 训练集 | 真正参与训练，更新模型参数 |
| 验证集 | 不更新参数，每轮观察是否过拟合、用来选最佳模型 |
| 测试集 | 训练结束后只用一次，做最终评估 |

### 5. 标准化（StandardScaler）

- **为什么**：各特征量纲差别大（GRE 约 300 上下、CGPA 约 8 上下），会让梯度下降不稳定、收敛变慢。
- **怎么做**：`z = (x − mean) / std`，把每列变成均值 0、标准差 1。
- **关键**：只在训练集上 `fit`（计算均值/方差），验证集和测试集只 `transform`。如果先对全量数据 `fit` 再划分，会把测试集信息泄露给模型。

### 6. 转 PyTorch 张量

```python
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
```

标签用 `view(-1, 1)` 变成 `(样本数, 1)` 的列向量，和模型输出的形状对齐。

### 7. 模型 / 损失 / 优化器

```python
model = nn.Linear(7, 1)          # 一个 7 → 1 的全连接层，即 y = Wx + b
criterion = nn.MSELoss()         # 均方误差
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### 8. 训练

- `DataLoader(batch_size=32, shuffle=True)` 分批喂数据；
- 共 500 个 epoch，每个 epoch：前向计算 → 算 loss → `loss.backward()` 求梯度 → `optimizer.step()` 更新参数；
- 每轮用验证集算一次 loss，**保留验证集表现最好的那一次模型**（本课题出现在第 149 轮）。

### 9. 评估

用 `mean_absolute_error`、`mean_squared_error`、`r2_score` 在三个数据集上分别计算 **MAE / MSE / RMSE / R²**。

### 10. 特征权重与保存

- 线性回归的权重代表每个特征对结果的「方向（正负）+ 相对大小」（标准化后的尺度）。
- `torch.save(model.state_dict(), "linear_regression_model.pt")` 保存训练好的权重。

## 线性回归知识点

### 模型

```
y = w1·x1 + w2·x2 + ... + w7·x7 + b
```

即一条（高维的）直线 / 超平面。

### 损失函数：MSE

```
MSE = (1/n) Σ (y_pred − y_true)²
```

- 平方让误差正负不会互相抵消；
- 误差越大，平方后被放大得越厉害，惩罚越重。

### 梯度下降与 SGD

- 目标：找到一组参数，让 MSE 尽可能小。
- 梯度下降：沿着「负梯度方向」更新参数，`w := w − lr · ∂L/∂w`。
- **SGD（随机梯度下降）**：每次用一个 batch 的数据近似全量梯度来更新参数，更快、且带有随机性。
- **学习率 lr**：步子大小。太大会震荡甚至发散，太小收敛很慢。

### Epoch 与 Batch

- `batch_size = 32`：一次喂 32 条样本；
- 一个 epoch = 把全部 batch 都过一遍；
- 500 个 epoch = 完整数据被反复训练 500 遍。

### 过拟合与验证集

- 训练 loss 低、验证 loss 高 → **过拟合**（模型在背答案而不是学规律）；
- 验证集不参与参数更新，只用来观察表现、选择最佳模型；
- 测试集只在最后用一次，避免“看着测试集调参”。

### 评估指标

- **MAE**：平均绝对误差，单位与 y 相同；
- **MSE**：均方误差，放大了大误差的影响；
- **RMSE**：MSE 开根号，回到 y 的原单位；
- **R²**：决定系数，越接近 1 越好（1 = 完美预测，0 = 不如直接取均值）。

### 标准化为什么重要

量纲不同会让梯度下降收敛慢，还会让某些量纲大的特征获得虚假的高权重；标准化后各特征尺度一致，权重才可公平比较、可解释。

### 可解释性（线性回归最大的优点）

权重可以直接解读：本例中 **CGPA 权重最大** → 本科 GPA 对录取影响最大；其次是 GRE、TOEFL；研究经历有正向作用。

## 运行

```bash
pip install -r requirements.txt
python admission_lr.py
```

## 文件结构

```
admission-predict/
├── Admission_Predict.csv
├── LR_jupyter.ipynb       # 教学讲解版 notebook
├── admission_lr.py        # 干净可复现脚本
├── requirements.txt
├── README.md
├── linear_regression_model.pt
└── output/                # 6 张结果图
```
