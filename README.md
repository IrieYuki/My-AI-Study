# My AI Study

我的 AI 学习笔记与项目合集：数据、机器学习、深度学习。每个项目都包含**可复现代码 + 结果图 + 知识点笔记**，既能作为作品展示，也能作为日后复习的笔记。

## 项目一览（机器学习）

| 项目 | 任务类型 | 模型 | 关键结果 | 知识点 |
|---|---|---|---|---|
| [titanic-logreg](titanic-logreg/) | 二分类 | 逻辑回归 | 准确率 0.815 | sigmoid、交叉熵、混淆矩阵、精确率/召回率 |
| [admission-predict](admission-predict/) | 回归 | 线性回归（PyTorch） | R² 0.815 | 线性回归、梯度下降、标准化、过拟合 |
| [walmart-sales](walmart-sales/) | 回归 | 线性回归（PyTorch） | R² 0.921 | one-hot、门店固定效应、Adam、目标标准化 |

## 深度学习案例

- [softmax_mnist_demo.ipynb](softmax_mnist_demo.ipynb)：Softmax 手写数字分类
- [fashion_mnist_cnn_teaching_case.ipynb](fashion_mnist_cnn_teaching_case.ipynb)：CNN 服饰分类
- [cifar10_cnn_teaching_case.ipynb](cifar10_cnn_teaching_case.ipynb)：CNN 图像分类（含数据增强）
- [media_pipe/](media_pipe/)：MediaPipe 手势 / 人脸 / 姿态检测
- [opencv/](opencv/)：OpenCV 教学案例
- [yolo/](yolo/)：YOLO 目标检测

## 学习地图：分类 vs 回归

- **回归**：预测连续数值 → 线性回归 → 损失用 MSE → 评估看 MAE / RMSE / R² → 见 `admission-predict`、`walmart-sales`
- **分类**：预测类别（0/1）→ 逻辑回归 → 损失用交叉熵 → 评估看准确率 / 混淆矩阵 → 见 `titanic-logreg`

## 环境

依赖见各目录下的 `requirements.txt`。
