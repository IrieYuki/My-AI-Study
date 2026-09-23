# -*- coding: utf-8 -*-
"""
研究生录取概率预测 —— PyTorch 线性回归
========================================
根据 GRE / TOEFL / GPA / 研究经历等 7 项信息，预测被美国研究生院录取的概率。

流水线（八步）：
  1 读数据  2 探索性分析  3 划分训练/验证/测试(60/20/20)
  4 标准化  5 转张量   6 定义模型   7 训练(跟踪最佳验证集)
  8 评估(MAE/MSE/RMSE/R²) + 特征权重

跑法：python admission_lr.py
依赖：pandas, torch, matplotlib, seaborn, scikit-learn
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "PingFang HK", "Heiti SC", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False
torch.manual_seed(42)
os.makedirs("output", exist_ok=True)

# ---------- 1. 读数据 ----------
df = pd.read_csv("Admission_Predict.csv")
df.columns = [c.strip() for c in df.columns]
df.drop(columns=["Serial No."], inplace=True)

# ---------- 2. 探索性分析（每个特征的分布） ----------
fig, axes = plt.subplots(2, 4, figsize=(13, 6))
for ax, col in zip(axes.flatten(), df.columns):
    sns.histplot(data=df, x=col, kde=True, bins=30, ax=ax)
    ax.set_title(col, fontsize=12)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/01_feature_distributions.png", dpi=150)
plt.close(fig)

# ---------- 3. 特征 / 目标 ----------
X = df.drop(columns=["Chance of Admit"])
y = df["Chance of Admit"]

# ---------- 4. 划分 训练/验证/测试 = 60/20/20 ----------
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# ---------- 5. 标准化（只在训练集上 fit） ----------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# ---------- 6. 转张量 ----------
def to_tensor(a): return torch.tensor(a, dtype=torch.float32)
X_train_t, y_train_t = to_tensor(X_train_s), to_tensor(y_train.values).view(-1, 1)
X_val_t, y_val_t = to_tensor(X_val_s), to_tensor(y_val.values).view(-1, 1)
X_test_t, y_test_t = to_tensor(X_test_s), to_tensor(y_test.values).view(-1, 1)

# ---------- 7. 模型 / 损失 / 优化器 ----------
model = nn.Linear(X.shape[1], 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

# ---------- 8. 训练（跟踪验证集最佳模型） ----------
epochs, best_val, best_state, best_epoch = 500, float("inf"), None, 0
train_losses, val_losses = [], []
for epoch in range(epochs):
    model.train(); epoch_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward(); optimizer.step()
        epoch_loss += loss.item() * len(xb)
    epoch_loss /= len(X_train_t)
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_t), y_val_t).item()
    train_losses.append(epoch_loss); val_losses.append(val_loss)
    if val_loss < best_val:
        best_val, best_state, best_epoch = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, epoch + 1

model.load_state_dict(best_state)
print(f"最佳验证集出现在第 {best_epoch} 轮，Loss={best_val:.4f}")

# ---------- 9. 评估 ----------
def evaluate(Xt, yt):
    model.eval()
    with torch.no_grad():
        pred = model(Xt).flatten().numpy()
    true = yt.flatten().numpy()
    return pred, true, mean_absolute_error(true, pred), mean_squared_error(true, pred), r2_score(true, pred)

p_tr, t_tr, mae_tr, mse_tr, r2_tr = evaluate(X_train_t, y_train_t)
p_va, t_va, mae_va, mse_va, r2_va = evaluate(X_val_t, y_val_t)
p_te, t_te, mae_te, mse_te, r2_te = evaluate(X_test_t, y_test_t)

metric_df = pd.DataFrame({
    "数据集": ["训练集", "验证集", "测试集"],
    "MAE": [mae_tr, mae_va, mae_te],
    "RMSE": [mse_tr ** 0.5, mse_va ** 0.5, mse_te ** 0.5],
    "R²": [r2_tr, r2_va, r2_te],
})
print(metric_df.round(4).to_string(index=False))
print(f"\n测试集：MAE={mae_te:.4f}  RMSE={mse_te**0.5:.4f}  R²={r2_te:.4f}")

# ---------- 10. 图表 ----------
plt.figure(figsize=(7, 4.5))
plt.plot(train_losses, label="训练集 Loss")
plt.plot(val_losses, label="验证集 Loss")
plt.axvline(best_epoch - 1, color="r", ls="--", lw=1, label=f"最佳轮次 {best_epoch}")
plt.title("训练集与验证集 Loss 变化"); plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("output/02_loss_curve.png", dpi=150); plt.close()

plt.figure(figsize=(7, 5.5))
plt.scatter(t_te, p_te, alpha=0.6)
plt.plot([0, 1], [0, 1], "r--", label="理想预测")
plt.title("测试集：预测值 vs 真实值"); plt.xlabel("真实录取概率"); plt.ylabel("预测录取概率")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("output/03_pred_vs_actual.png", dpi=150); plt.close()

residuals = t_te - p_te
plt.figure(figsize=(7, 4.5))
sns.histplot(residuals, bins=30, kde=True)
plt.title("测试集残差分布：真实值 - 预测值"); plt.xlabel("残差"); plt.ylabel("频数")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("output/04_residual_distribution.png", dpi=150); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.barplot(data=metric_df, x="数据集", y="MAE", ax=axes[0]); axes[0].set_title("MAE 对比"); axes[0].grid(True, alpha=0.3)
sns.barplot(data=metric_df, x="数据集", y="R²", ax=axes[1]); axes[1].set_title("R² 对比"); axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/05_metrics_comparison.png", dpi=150); plt.close(fig)

weights = model.weight.detach().numpy().flatten()
fi = pd.Series(weights, index=X.columns).sort_values()
plt.figure(figsize=(8, 5))
fi.plot(kind="barh", color="#2F6FED")
plt.title("标准化后的特征权重"); plt.xlabel("权重"); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("output/06_feature_weights.png", dpi=150); plt.close()

torch.save(model.state_dict(), "linear_regression_model.pt")
print("\n模型权重已保存到 linear_regression_model.pt")
print("图表已保存到 output/ 目录。")
