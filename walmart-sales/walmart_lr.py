# -*- coding: utf-8 -*-
"""
Walmart 周销售额预测 —— PyTorch 线性回归
==========================================
根据门店、节假日、温度、油价、CPI、失业率，预测当周销售额。

流水线（与 admission-predict 同套路）：
  1 读数据  2 探索性分析  3 特征工程(门店 one-hot)
  4 划分训练/验证/测试(60/20/20)  5 标准化(数值特征 + 目标)
  6 转张量  7 线性模型 + MSE + Adam  8 训练(跟踪最佳验证集)
  9 评估(MAE/RMSE/R²) + 特征权重

跑法：python walmart_lr.py
依赖：pandas, torch, matplotlib, seaborn, scikit-learn
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "PingFang HK", "Heiti SC", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False
torch.manual_seed(42)
os.makedirs("output", exist_ok=True)

# ---------- 1. 读数据 ----------
df = pd.read_csv("Walmart.csv", parse_dates=["Date"], dayfirst=True)

# ---------- 2. 探索性分析 ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["Weekly_Sales"] / 1e6, bins=40, ax=axes[0])
axes[0].set_title("周销售额分布（百万美元）"); axes[0].set_xlabel("Weekly Sales (M$)"); axes[0].grid(True, alpha=0.3)
sns.boxplot(data=df, x="Holiday_Flag", y=df["Weekly_Sales"] / 1e6, ax=axes[1])
axes[1].set_title("节假日 vs 非节假日"); axes[1].set_xlabel("Holiday_Flag"); axes[1].set_ylabel("Weekly Sales (M$)"); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("output/01_sales_overview.png", dpi=150); plt.close(fig)

monthly = df.groupby(df["Date"].dt.to_period("M"))["Weekly_Sales"].mean()
plt.figure(figsize=(9, 4))
plt.plot([str(m) for m in monthly.index], monthly.values, marker="o", color="#2F6FED")
plt.title("月度平均周销售额趋势"); plt.ylabel("Average Weekly Sales (M$)")
plt.xticks(rotation=45); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("output/02_monthly_sales.png", dpi=150); plt.close()

# ---------- 3. 特征 / 目标 ----------
num_cols = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
cat_cols = ["Store", "Holiday_Flag"]
X = pd.get_dummies(df[cat_cols + num_cols], columns=["Store"], drop_first=True).astype(float)
y = df["Weekly_Sales"].values.reshape(-1, 1)

# ---------- 4. 划分 60/20/20 ----------
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# ---------- 5. 标准化（数值特征 + 目标，都只在训练集 fit） ----------
x_scaler = StandardScaler()
X_train[num_cols] = x_scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = x_scaler.transform(X_val[num_cols])
X_test[num_cols] = x_scaler.transform(X_test[num_cols])

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_val = y_scaler.transform(y_val)
y_test = y_scaler.transform(y_test)

# ---------- 6. 转张量 ----------
def to_tensor(a): return torch.tensor(np.asarray(a, dtype=np.float32))
X_train_t, y_train_t = to_tensor(X_train), to_tensor(y_train).view(-1, 1)
X_val_t, y_val_t = to_tensor(X_val), to_tensor(y_val).view(-1, 1)
X_test_t, y_test_t = to_tensor(X_test), to_tensor(y_test).view(-1, 1)

# ---------- 7. 模型 / 损失 / 优化器 ----------
model = nn.Linear(X.shape[1], 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# ---------- 8. 训练（跟踪最佳验证集） ----------
epochs, best_val, best_state, best_epoch = 600, float("inf"), None, 0
train_losses, val_losses = [], []
for epoch in range(epochs):
    model.train(); optimizer.zero_grad()
    loss = criterion(model(X_train_t), y_train_t)
    loss.backward(); optimizer.step()
    train_losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_t), y_val_t).item()
    val_losses.append(val_loss)
    if val_loss < best_val:
        best_val, best_state, best_epoch = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, epoch + 1
model.load_state_dict(best_state)
print(f"最佳验证集出现在第 {best_epoch} 轮，Loss={best_val:.4f}")

# ---------- 9. 评估（换算回美元） ----------
def evaluate(Xt, yt):
    model.eval()
    with torch.no_grad():
        pred = model(Xt).numpy().reshape(-1, 1)
    pred_d = y_scaler.inverse_transform(pred)
    true_d = y_scaler.inverse_transform(yt.numpy())
    return pred_d, true_d, mean_absolute_error(true_d, pred_d), mean_squared_error(true_d, pred_d) ** 0.5, r2_score(true_d, pred_d)

p_tr, t_tr, mae_tr, rmse_tr, r2_tr = evaluate(X_train_t, y_train_t)
p_va, t_va, mae_va, rmse_va, r2_va = evaluate(X_val_t, y_val_t)
p_te, t_te, mae_te, rmse_te, r2_te = evaluate(X_test_t, y_test_t)

metric_df = pd.DataFrame({
    "数据集": ["训练集", "验证集", "测试集"],
    "MAE(美元)": [mae_tr, mae_va, mae_te],
    "RMSE(美元)": [rmse_tr, rmse_va, rmse_te],
    "R²": [r2_tr, r2_va, r2_te],
})
print(metric_df.round(1).to_string(index=False))
print(f"\n测试集：MAE=${mae_te:,.0f}  RMSE=${rmse_te:,.0f}  R²={r2_te:.4f}")

# ---------- 10. 图表 ----------
plt.figure(figsize=(7, 4.5))
plt.plot(train_losses, label="训练集 Loss"); plt.plot(val_losses, label="验证集 Loss")
plt.axvline(best_epoch - 1, color="r", ls="--", lw=1, label=f"最佳轮次 {best_epoch}")
plt.title("训练集与验证集 Loss 变化"); plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("output/03_loss_curve.png", dpi=150); plt.close()

plt.figure(figsize=(7, 5.5))
plt.scatter(t_te / 1e6, p_te / 1e6, alpha=0.5, s=12)
lo, hi = min(t_te.min(), p_te.min()) / 1e6, max(t_te.max(), p_te.max()) / 1e6
plt.plot([lo, hi], [lo, hi], "r--", label="理想预测")
plt.title("测试集：预测值 vs 真实值"); plt.xlabel("真实周销售额 (M$)"); plt.ylabel("预测周销售额 (M$)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("output/04_pred_vs_actual.png", dpi=150); plt.close()

resid = (t_te - p_te).flatten()
plt.figure(figsize=(7, 4.5))
sns.histplot(resid / 1e3, bins=40, kde=True)
plt.title("测试集残差分布（真实值 - 预测值，千美元）"); plt.xlabel("残差 (K$)"); plt.ylabel("频数"); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("output/05_residual_distribution.png", dpi=150); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.barplot(data=metric_df, x="数据集", y="MAE(美元)", ax=axes[0]); axes[0].set_title("MAE 对比"); axes[0].grid(True, alpha=0.3)
sns.barplot(data=metric_df, x="数据集", y="R²", ax=axes[1]); axes[1].set_title("R² 对比"); axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("output/06_metrics_comparison.png", dpi=150); plt.close(fig)

# 经济/天气因素的特征权重（门店 one-hot 权重另计，单独说明）
w = model.weight.detach().numpy().flatten()
factor_idx = [X.columns.get_loc(c) for c in num_cols + ["Holiday_Flag"]]
fi = pd.Series(w[factor_idx], index=num_cols + ["Holiday_Flag"]).sort_values()
store_w = pd.Series(w, index=X.columns).filter(regex="Store_")
plt.figure(figsize=(8, 4.6))
fi.plot(kind="barh", color=["#E07B39" if v < 0 else "#2F6FED" for v in fi.values])
plt.title("经济与天气因素的特征权重（标准化尺度）"); plt.xlabel("权重"); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("output/07_factor_weights.png", dpi=150); plt.close()

print("\n门店固定效应（44 个 Store 虚拟变量的权重范围）：")
print(f"  min={store_w.min():.3f}  max={store_w.max():.3f}  -> 门店是销售最大的决定因素")

torch.save(model.state_dict(), "walmart_lr_model.pt")
print("\n模型权重已保存到 walmart_lr_model.pt，图表已保存到 output/。")
