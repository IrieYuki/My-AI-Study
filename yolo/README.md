# YOLO 目标检测教程：图片、视频、实时摄像头

这个目录包含 3 个 YOLO 目标检测案例。你可以通过它们理解：卷积神经网络不仅能做手写数字、服装分类，也可以在真实图片、视频和摄像头画面中找出物体。

## 1. 安装依赖

建议使用 Python 3.10、3.11 或 3.12。先进入本目录：

```bash
cd /Users/eric/Desktop/c/第八课/yolo
python3 -m pip install -r requirements.txt
```

这会安装两个主要库：

- `ultralytics`：提供 YOLO 模型和推理接口。
- `opencv-python`：负责读取图片、视频、摄像头，并显示或保存结果。

如果你希望“视频识别结果保留原视频声音”，还需要安装 `ffmpeg`。`ffmpeg` 是一个常用的视频/音频处理工具，这里用它把“检测后的视频画面”和“原视频声音”合并起来。

Mac 安装方式：

如果你已经安装了 Homebrew，可以运行：

```bash
brew install ffmpeg
```

Homebrew 是 Mac 上常用的软件包管理工具，可以理解为“用命令安装软件的工具”。如果你的电脑还没有 `brew` 命令，可以先搜索并安装 Homebrew，或者请有经验的人协助安装。

Windows 安装方式：

如果你的 Windows 支持 `winget`，可以在 PowerShell 中运行：

```powershell
winget install Gyan.FFmpeg
```

也可以下载 `ffmpeg` 的 Windows 版本，解压后把里面的 `bin` 目录添加到系统 `PATH`。

安装完成后，重新打开终端，运行：

```bash
ffmpeg -version
```

如果能看到版本号，说明安装成功。

首次运行脚本时，`ultralytics` 会自动下载默认模型 `yolov8n.pt`。这个模型比较小，速度快，适合初学练习。

## 2. 单体照片识别

把一张测试照片放到本目录，例如 `test.jpg`。

打开 [01_yolo_image_detect.py](/Users/eric/Desktop/c/第八课/yolo/01_yolo_image_detect.py)，在文件开头找到配置区：

```python
IMAGE_PATH = "./test.jpg"
MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.35
OUTPUT_PATH = "./outputs/image_result.jpg"
SHOW_WINDOW = True
```

确认 `IMAGE_PATH` 是你的图片路径，然后运行：

```bash
python3 01_yolo_image_detect.py
```

输出结果默认保存到：

```text
./outputs/image_result.jpg
```

你需要观察：

- 图片进入模型后，YOLO 会预测物体类别和位置。
- 终端会打印每个目标的类别、置信度和坐标。
- 输出图片会把检测框直接画出来。

## 3. 动态视频识别

把一个测试视频放到本目录，例如 `street.mp4`。

打开 [02_yolo_video_detect.py](/Users/eric/Desktop/c/第八课/yolo/02_yolo_video_detect.py)，在文件开头找到配置区：

```python
VIDEO_PATH = "./street.mp4"
MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.35
OUTPUT_PATH = "./outputs/video_result.mp4"
SHOW_WINDOW = True
PRESERVE_AUDIO = True
```

确认 `VIDEO_PATH` 是你的视频路径，然后运行：

```bash
python3 02_yolo_video_detect.py
```

输出结果默认保存到：

```text
./outputs/video_result.mp4
```

如果 `PRESERVE_AUDIO = True`，程序会先生成一个无声的检测视频，再用 `ffmpeg` 把原视频声音合并回来。

你需要观察：

- 视频其实就是很多张连续图片。
- 程序逐帧读取视频，把每一帧送入 YOLO。
- 每一帧检测完成后写入新视频，所以得到一个带识别框的视频。

## 4. 实时摄像头识别

运行：

```bash
python3 03_yolo_webcam_realtime.py
```

窗口打开后：

- 按 `q` 退出。
- 按 `s` 保存当前检测画面截图。

如果摄像头打不开，可以尝试：

```python
CAMERA_INDEX = 1
```

也就是打开 [03_yolo_webcam_realtime.py](/Users/eric/Desktop/c/第八课/yolo/03_yolo_webcam_realtime.py)，把配置区里的 `CAMERA_INDEX = 0` 改成 `CAMERA_INDEX = 1`。

你需要观察：

- 摄像头不断产生新的画面帧。
- 程序边读取、边识别、边显示。
- 画面左上角的 FPS 表示每秒处理多少帧。

## 5. 常用配置

提高置信度，让检测结果更严格：

```python
CONFIDENCE_THRESHOLD = 0.6
```

换模型：

```python
MODEL_PATH = "yolov8s.pt"
```

模型大小通常是：

```text
yolov8n.pt 速度最快，精度较低
yolov8s.pt 速度较快，精度更好
yolov8m.pt 精度更高，但更慢
```

学习阶段优先推荐 `yolov8n.pt`，因为等待时间短，你更容易先跑通完整流程。
