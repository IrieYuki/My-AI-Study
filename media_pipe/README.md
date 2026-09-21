# MediaPipe 视觉识别课堂示例

这个文件夹包含 4 个教程版 Python 示例：

- `mp_hand.py`：手部关键点识别，最多识别 2 只手。
- `mp_face.py`：人脸关键点识别，并用 face blendshapes 做简单表情演示。
- `mp_body.py`：身体骨骼 / 姿态关键点识别。
- `mp_holistic.py`：全身综合识别 Holistic Landmarker，同时识别身体、人脸、左右手。

## 1. 环境准备

建议使用 Python 3.12。Python 3.13 / 3.14 可能因为 MediaPipe 没有对应安装包而安装失败。

**MediaPipe 的版本比 Python 版本更关键。** `requirements.txt` 里把 mediapipe 锁死在 `0.10.35`：
2026 年发布的 `1.0.0` / `1.0.1` 是 alpha 版本，在 macOS 上使用 CPU delegate 时，会在
“创建识别器对象”的那一刻直接让 Python 崩溃退出，终端不会给出 Python 报错（详见第 5 节）。
所以不要把这一行改成 `mediapipe>=...`，否则 pip 会装成 1.0.1。

macOS 上强烈建议使用本文件夹内的专用虚拟环境 `mp_venv`：既不影响电脑上其它 Python 项目，
也不会因为装错版本把整个解释器环境弄乱。

> **为什么这里叫 `mp_venv`，不叫 `.venv`？**
> 家目录下可能已经有一个 `~/.venv`（本机就有，里面装的是 mediapipe 1.0.1）。
> `source .venv/bin/activate` 是**相对当前所在目录**解析的，不是一个固定位置：
> 在家目录里执行会激活 `~/.venv`，只有在课程目录里执行才会激活本项目的环境。
> 两个目录同名极易搞混，所以本项目改用带 `mp_` 前缀的 `mp_venv`，一眼可辨、不会误激活。


### Windows 安装方式

```bash
cd 你的课程文件夹\media_pipe
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

运行示例：

```bash
py mp_hand.py
```

### macOS 安装方式

```bash
cd 你的课程文件夹/media_pipe
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

运行示例：

```bash
python3 mp_hand.py
```

### 如果老师要求使用虚拟环境

Windows：

```bash
py -m venv mp_venv
mp_venv\Scripts\activate
py -m pip install -r requirements.txt
```

macOS：

```bash
python3 -m venv mp_venv
source mp_venv/bin/activate
python3 -m pip install -r requirements.txt
```

### macOS 专用虚拟环境（本机已配好）

本文件夹里已经创建好 `mp_venv`（Python 3.12 + mediapipe 0.10.35）。
以后每次运行前先激活它，激活后终端提示符前面会出现 `(mp_venv)`：

> **注意：这套课程目录在本机有两份，而 `mp_venv` 只在 `~/AI/media_pipe` 这一份里。**
> 另一份是 `~/Documents/Obsidian Vault/AI/media_pipe`（课程资料副本，没有虚拟环境）。
> 因为 `source mp_venv/bin/activate` 是相对当前目录解析的，在那份副本里执行会直接报
> `no such file or directory: mp_venv/bin/activate`。
> 激活前后都建议核对一下自己到底在哪个目录、用的哪个 mediapipe：
>
> ```bash
> pwd                                                          # 应为 /Users/zhangzixin/AI/media_pipe
> python -c "import mediapipe; print(mediapipe.__version__)"   # 应为 0.10.35
> ```

```bash
cd "/Users/zhangzixin/AI/media_pipe"
source mp_venv/bin/activate

python mp_face.py
```

关掉终端再打开时需要重新 `source mp_venv/bin/activate`。

如果 `mp_venv` 损坏，或者换到别的电脑需要重建：

```bash
cd "/Users/zhangzixin/AI/media_pipe"

# 用 Python 3.12 创建，不要用 3.13 / 3.14
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv mp_venv

source mp_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. 模型文件

本文件夹已经包含这些模型：

- `hand_landmarker.task`
- `face_landmarker.task`
- `pose_landmarker.task`
- `holistic_landmarker.task`

代码里使用 `Path(__file__).resolve().parent` 自动定位模型文件，所以只要 `.py` 和 `.task` 在同一个文件夹，从哪里运行都不容易报模型路径错误。

如果以后需要重新下载 Holistic 模型：

```bash
curl -L -o holistic_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
```

## 3. 运行示例

手部识别：

```bash
py mp_hand.py        # Windows
python3 mp_hand.py   # macOS
```

人脸识别：

```bash
py mp_face.py        # Windows
python3 mp_face.py   # macOS
```

身体骨骼识别：

```bash
py mp_body.py        # Windows
python3 mp_body.py   # macOS
```

全身综合识别：

```bash
py mp_holistic.py        # Windows
python3 mp_holistic.py   # macOS
```

运行后按 `q` 退出。

## 4. 修改课堂配置

为了让初学者更容易理解，本课程示例不使用命令行参数。
如果要调整程序，请打开对应 `.py` 文件，修改文件顶部的“课堂可修改配置”常量。

例如 `mp_hand.py` 顶部有这些配置：

```python
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
NUM_HANDS = 2
CONFIDENCE = 0.55
```

如果电脑有多个摄像头，可以把摄像头编号从 0 改成 1：

```python
CAMERA_INDEX = 1
```

如果画面卡顿，可以降低分辨率：

```python
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
```

如果识别不到人、手或脸，可以降低置信度阈值：

```python
CONFIDENCE = 0.4
```

Holistic 示例默认不开启人体分割遮罩。如果想观察“模型认为哪里是人体”，可以开启：

```python
ENABLE_SEGMENTATION_MASK = True
```

## 5. 常见问题

### No module named 'mediapipe'

通常说明依赖没有安装到正在运行代码的那个 Python 里。

Windows 可以检查：

```bash
py --version
py -m pip show mediapipe
py -m pip install -r requirements.txt
```

macOS 可以检查：

```bash
python3 --version
python3 -m pip show mediapipe
python3 -m pip install -r requirements.txt
```

### 运行后 Python 直接“意外退出”（macOS 弹崩溃提示 / 终端显示 Abort trap: 6）

**症状**：程序刚启动、甚至还没出现摄像头窗口，Python 进程就直接消失。
终端里可能只有 `Abort trap: 6` 或 `zsh: abort`，看不到 Python 的 Traceback。

**原因**：这是 MediaPipe 的 C++ 层调用了 `abort()`，属于进程级崩溃，不是 Python 异常，
所以 `try / except` 拦不住，程序也不会走到 `finally` 里清理摄像头。

**定位方法**：macOS 会把崩溃报告写在这里，用崩溃报告能看到真正出错的 C++ 调用栈。

```bash
ls -lt ~/Library/Logs/DiagnosticReports/ | grep Python | head
```

本例最常见的两条报错：

```text
# delegate=CPU 时，崩在“创建识别器”这一步（本次就是这个）
F0000 graph_service.h:139] Check failed: service_ Service is unavailable.
    @ -[DrishtiMetalHelper initWithCalculatorContext:]
    @ mediapipe::api2::TensorsToDetectionsCalculator::Open()

# delegate=GPU 时，崩在“处理第一帧”这一步
F0000 gpu_buffer_storage_cv_pixel_buffer.cc:154] Check failed: status_or_buffer is OK
    (UNKNOWN: unsupported ImageFrame format: 1 ...)
```

**结论**：这是 mediapipe 1.0.x（alpha 版本）在 macOS 上的 bug，CPU 和 GPU 两条路都会崩，
所以**不能靠改 `delegate` 绕过**，正确的解法是把 mediapipe 降回 0.10.35。

先确认当前版本：

```bash
python -c "import mediapipe; print(mediapipe.__version__)"
```

如果输出 `1.0.0` 或 `1.0.1`，就降级（必须装在运行代码的那个虚拟环境里）：

```bash
source mp_venv/bin/activate
python -m pip install "mediapipe==0.10.35"
python -c "import mediapipe; print(mediapipe.__version__)"   # 期望输出 0.10.35
```

最省事的做法是直接用本文件夹里已配好的 `mp_venv`，它已经锁定 0.10.35。

### 无法打开摄像头

检查三件事：

- 摄像头是否被微信、腾讯会议、浏览器等其它软件占用。
- Windows 是否允许当前应用访问相机：设置 -> 隐私和安全性 -> 相机。
- macOS 是否允许当前终端、VS Code、PyCharm 等工具访问相机：系统设置 -> 隐私与安全性 -> 相机。
- 摄像头编号是否正确，可以把脚本顶部的 `CAMERA_INDEX = 0` 改成 `CAMERA_INDEX = 1`。

### 画面能打开但识别很差

优先检查这些条件：

- 光线是否太暗。
- 手或脸是否离摄像头太近，导致关键部位出画。
- 身体是否被桌子、椅子、屏幕边缘遮挡。
- 背景是否特别杂乱。

### 为什么要 BGR 转 RGB

OpenCV 读取摄像头画面时颜色顺序是 BGR，也就是蓝、绿、红。
MediaPipe 模型需要 RGB，也就是红、绿、蓝。
如果不转换，模型看到的颜色和真实画面不一致，识别效果可能变差。

```python
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

### 为什么 VIDEO 模式需要 timestamp

MediaPipe 的视频模式会利用上一帧的信息做跟踪。
为了知道“哪一帧在前、哪一帧在后”，每次调用 `detect_for_video()` 都要传入递增的毫秒时间戳。

```python
detection_result = detector.detect_for_video(mp_image, timestamp_ms)
```

## 6. 课堂讲解顺序建议

1. 先讲 `mp_hand.py`：关键点、连接线、左右手、置信度。
2. 再讲 `mp_face.py`：468 点太多，所以只画重点区域；blendshape 是面部动作分数。
3. 再讲 `mp_body.py`：33 个身体点和 visibility，可见度低时线条变暗。
4. 最后讲 `mp_holistic.py`：一个综合模型同时输出身体、人脸、左右手，适合做交互应用。
