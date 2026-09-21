# OpenCV 图像处理实验台

本文件夹提供一个面向初学者的 OpenCV 综合教学案例：

- `opencv_rich_teaching_case.py`：一个窗口同时展示多个 OpenCV 图像处理效果。
- `requirements.txt`：运行需要安装的 Python 库。

## 1. 这个案例演示什么

运行后会看到一个“图像处理实验台”，里面包含：

- 原图显示
- BGR 三通道拆分
- 彩色图转灰度图
- 高斯模糊
- Canny 边缘检测
- 二值阈值分割
- 形态学处理
- 轮廓检测和目标计数
- HSV 颜色提取
- 直方图均衡化

默认不需要摄像头，也不需要准备图片。程序会自动生成一张教学测试图，保证学生第一次运行就能看到效果。

## 2. 安装依赖

可以直接安装到当前 Python 环境，也可以安装到虚拟环境。虚拟环境不是强制要求。

Windows：

```bash
cd 你的课程文件夹\opencv
py -m pip install -r requirements.txt
```

macOS：

```bash
cd 你的课程文件夹/opencv
python3 -m pip install -r requirements.txt
```

## 3. 运行程序

Windows：

```bash
py opencv_rich_teaching_case.py
```

macOS：

```bash
python3 opencv_rich_teaching_case.py
```

窗口打开后：

- 按 `q` 退出。
- 按 `s` 保存当前实验台截图。
- 图片模式下，程序会自动保存一张截图到 `outputs/opencv_teaching_gallery.jpg`。

## 4. 修改配置

打开 `opencv_rich_teaching_case.py`，在文件顶部找到“课堂可修改配置”。

使用摄像头：

```python
USE_CAMERA = True
```

换摄像头编号：

```python
CAMERA_INDEX = 1
```

降低分辨率，让程序更流畅：

```python
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
```

使用自己的图片：

```python
IMAGE_PATH = r"C:\Users\student\Desktop\test.jpg"  # Windows 示例
```

或：

```python
IMAGE_PATH = "/Users/student/Desktop/test.jpg"  # macOS 示例
```

## 5. 常见问题

### No module named 'cv2'

说明 OpenCV 没有安装到当前正在运行代码的 Python 环境。

Windows：

```bash
py -m pip install -r requirements.txt
```

macOS：

```bash
python3 -m pip install -r requirements.txt
```

### 摄像头打不开

检查三件事：

- 摄像头是否被微信、腾讯会议、浏览器等其它软件占用。
- Windows 是否允许当前应用访问相机：设置 -> 隐私和安全性 -> 相机。
- macOS 是否允许当前终端、VS Code、PyCharm 等工具访问相机：系统设置 -> 隐私与安全性 -> 相机。

### 为什么 OpenCV 是 BGR，不是 RGB

OpenCV 默认使用 BGR 顺序：

- B：Blue，蓝色
- G：Green，绿色
- R：Red，红色

所以在 OpenCV 里：

```python
(255, 0, 0)
```

表示蓝色，不是红色。

### 为什么要先模糊再做边缘检测

真实图片里经常有噪声。噪声也可能被 Canny 当成边缘。
先用高斯模糊去掉一部分噪声，边缘检测结果会更稳定。

### 阈值、形态学、轮廓之间是什么关系

可以按这个顺序理解：

1. 阈值分割：把图像变成黑白图。
2. 形态学处理：让白色区域更完整，减少小断裂。
3. 轮廓检测：沿着白色区域边界找到目标外形。

这就是很多传统视觉任务的基础流程。
