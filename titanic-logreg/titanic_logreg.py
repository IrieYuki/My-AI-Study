# -*- coding: utf-8 -*-
"""
Titanic 存活预测 —— 逻辑回归小 demo
========================================
目标：用一条机器学习流水线，根据乘客信息预测 TA 是否幸存。

流水线五步（面试可一句话讲清）：
  1. 读数据  2. 数据预处理  3. 划分训练/测试集
  4. 训练逻辑回归  5. 评估（准确率 + 混淆矩阵 + 特征重要性）

跑法：python titanic_logreg.py
依赖：pandas, scikit-learn, matplotlib, seaborn
"""
import matplotlib
matplotlib.use("Agg")                     # 无界面环境也能画图（服务器/CI 常用）
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------- 1. 读数据 ----------
df = pd.read_csv("titanic.csv")

# ---------- 2. 数据预处理 ----------
df["age"] = df["age"].fillna(df["age"].median())   # 年龄缺失 -> 用中位数填补
df = df.dropna(subset=["embarked"])                 # 登船港口只缺 2 行 -> 直接删

features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
X = pd.get_dummies(df[features], drop_first=True)   # 类别特征转 0/1（sex -> sex_male 等）
y = df["survived"]

# ---------- 3. 划分训练 / 测试集 ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)   # 固定随机种子 -> 结果可复现

# ---------- 4. 训练逻辑回归 ----------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------- 5. 评估 ----------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"测试集准确率：{acc:.3f}  （测试样本数 {len(y_test)}）")

# 图 1：混淆矩阵热力图（横轴=预测，纵轴=真实）
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("output/confusion_matrix.png", dpi=150)

# 图 2：特征重要性（逻辑回归的系数 = 每个特征对结果的“正/负影响”）
coef = pd.Series(model.coef_[0], index=X.columns).sort_values()
plt.figure(figsize=(6, 4))
coef.plot(kind="barh", color=coef.map(lambda v: "#d9534f" if v < 0 else "#5cb85c"))
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.xlabel("Coefficient")
plt.tight_layout()
plt.savefig("output/feature_importance.png", dpi=150)
print("图表已保存到 output/ 目录。")
