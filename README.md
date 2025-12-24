# EmbodiedCar - 多模态智能小车控制系统

一个基于树莓派的智能小车控制系统，支持**姿态识别**、**目标检测**和**语音控制**三种交互方式。

## 📋 项目简介

EmbodiedCar 是一个集成了多种 AI 技术的智能小车控制平台，可以通过以下方式控制小车：

- 🤲 **姿态识别控制**：使用 MediaPipe 实时检测人体姿态，通过手势控制小车
- 👁️ **目标检测控制**：使用 YOLO-FastestV2 检测目标，实现自动跟随
- 🎤 **语音控制**：使用 Sherpa-ONNX 进行中文语音识别，通过语音命令控制小车

## 🏗️ 项目结构

```
embodiedcar/
├── mediapipe_app/          # MediaPipe 姿态识别控制模块
│   ├── mediapipe_detect.py # 主程序：实时姿态检测与手势识别
│   ├── pose_landmarker.task # MediaPipe 姿态检测模型
│   └── requirements.txt    # Python 依赖
│
├── Yolo-FastestV2/         # YOLO 目标检测控制模块
│   ├── rasp_yolo.py        # 主程序：目标检测与控制
│   ├── model/              # YOLO 模型文件
│   └── requirements.txt    # Python 依赖
│
├── small-car/              # 语音控制模块
│   ├── Step1-TestYourMic.py              # 测试麦克风
│   ├── Step2-TryASR.py                   # 测试语音识别
│   ├── Step3-VAD+ASR.py                  # VAD + ASR
│   ├── Step4-ASR+Control.py             # 语音控制（关键词匹配）
│   ├── Step4_1-SimpleVoiceControl.py     # 简单语音控制
│   ├── Step4_2c-SmarterVoiceControl.py  # 智能语音控制（语言模型）
│   ├── model/                            # ASR、VAD、MLP 模型
│   └── README.md                         # 语音模块说明
│
├── GroundingDINO/          # GroundingDINO 目标检测（高级功能）
│   └── ...
│
├── CppVISUAL/              # C++ 视觉处理模块
│   └── ...
│
└── command_repository.txt  # 命令参考文档
```

## 🚀 快速开始

### 环境要求

- **树莓派**：运行 ROS2 控制节点和视频流服务
- **PC/笔记本电脑**：运行 AI 推理和控制程序
- **Python 3.8+**（推荐使用 Anaconda 管理环境）

### 1. 树莓派端设置

#### 1.1 安装 ROS2 控制节点

```bash
# 创建工作空间
mkdir -p ~/VISUAL/src
cd ~/VISUAL/src
# 放置 ROS2 节点代码...

# 编译
cd ~/VISUAL
colcon build --symlink-install

# 激活环境
source install/setup.bash
```

#### 1.2 启动视频流服务

```bash
cd mjpg-streamer/mjpg-streamer-experimental

# 启动视频流（640x480, 10fps）
./mjpg_streamer -i "./input_uvc.so -d /dev/video0 -r 640x480 -f 10" \
                -o "./output_http.so -w ./www"

# 提高帧率（20fps）
./mjpg_streamer -i "./input_uvc.so -d /dev/video0 -r 640x480 -f 20" \
                -o "./output_http.so -w ./www"

# 降低分辨率以提高性能（320x240, 20fps）
./mjpg_streamer -i "./input_uvc.so -d /dev/video0 -r 320x240 -f 20" \
                -o "./output_http.so -w ./www"
```

**参数说明**：
- `-d /dev/video0`：摄像头设备
- `-r 640x480`：分辨率（可调整）
- `-f 10`：帧率（可调整，CPU 慢时可降低）

#### 1.3 启动控制服务

```bash
cd ~/VISUAL
source install/setup.bash
ros2 run yolo yolo_control
```

### 2. PC 端设置

#### 2.1 创建 Python 环境

```bash
# 创建基础环境
conda create -n embodiedcar python=3.8
conda activate embodiedcar

# 或创建 MediaPipe 专用环境（需要 Python 3.9+）
conda create -n embodiedmediapipe python=3.9
conda activate embodiedmediapipe
```

#### 2.2 安装依赖

**MediaPipe 姿态识别模块**：
```bash
cd mediapipe_app
pip install -r requirements.txt
# 或手动安装
pip install opencv-python>=4.5.0 mediapipe>=0.10.0 numpy>=1.19.0 requests>=2.25.0
```

**YOLO 目标检测模块**：
```bash
cd Yolo-FastestV2
pip install -r requirements.txt
# 或手动安装
pip install torch torchvision opencv-python numpy tqdm torchsummary
```

**语音控制模块**：
```bash
cd small-car
pip install sherpa-onnx numpy librosa sounddevice sentence-transformers

# 下载 ASR 模型
# 从 https://github.com/k2-fsa/sherpa-onnx/releases 下载
# sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2
# 解压到 model/ASR/ 文件夹
```

#### 2.3 配置 IP 地址

在运行程序前，请修改代码中的树莓派 IP 地址：

**MediaPipe 模块** (`mediapipe_app/mediapipe_detect.py`)：
```python
stream_url = "http://YOUR_RASPBERRY_PI_IP:8080/?action=stream"
control_url = "http://YOUR_RASPBERRY_PI_IP:5000/control"
```

**YOLO 模块** (`Yolo-FastestV2/rasp_yolo.py`)：
```python
stream_url = "http://YOUR_RASPBERRY_PI_IP:8080/?action=stream"
control_url = "http://YOUR_RASPBERRY_PI_IP:5000/control"
```

## 🎮 使用方法

### 方式一：姿态识别控制

使用 MediaPipe 实时检测人体姿态，通过手势控制小车。

**支持的手势**：
- 🤲 **T 字型（双臂水平张开）** → 前进 (FORWARD)
- 👈 **左手向左平举** → 左转 (LEFT)
- 👉 **右手向右平举** → 右转 (RIGHT)
- 🙌 **双手上举** → 后退 (BACKWARD)
- 🛑 **其他姿势** → 停止 (STOP)

**运行**：
```bash
cd mediapipe_app
conda activate embodiedmediapipe  # 或你的环境名
python mediapipe_detect.py
```

**特性**：
- 实时姿态检测（33 个关键点）
- 自动绘制人体骨架
- 每个命令执行 30 帧后自动停止
- 支持 GPU 加速（如果可用）

### 方式二：目标检测控制

使用 YOLO-FastestV2 检测目标（如人物），实现自动跟随。

**运行**：
```bash
cd Yolo-FastestV2
conda activate embodiedcar
python rasp_yolo.py
```

**功能**：
- 实时目标检测
- 自动跟踪指定目标
- 根据目标位置控制小车移动

### 方式三：语音控制

使用中文语音识别，通过语音命令控制小车。

**支持的语音命令**：
- "前进" / "直行" → FORWARD
- "后退" → BACKWARD
- "左转" → LEFT
- "右转" → RIGHT
- "停止" / "停车" → STOP
- "加速" / "快点" → 提高速度
- "减速" / "慢一点" → 降低速度

**运行**：

1. **简单关键词匹配**：
```bash
cd small-car
python Step4-ASR+Control.py
```

2. **智能语音控制（使用语言模型）**：
```bash
cd small-car
python Step4_2c-SmarterVoiceControl.py
```

**测试步骤**：
```bash
# 1. 测试麦克风
python Step1-TestYourMic.py

# 2. 测试语音识别
python Step2-TryASR.py

# 3. 测试 VAD + ASR
python Step3-VAD+ASR.py
```

## 🔧 硬件配置

### 树莓派电机引脚配置

```python
电机引脚映射：
- left_back:   Motor(forward=27, backward=18)
- left_front:  Motor(forward=6,  backward=5)
- right_front: Motor(forward=19, backward=13)
- right_back:  Motor(forward=16, backward=12)
```

### 摄像头配置

- 设备：`/dev/video0`
- 默认分辨率：640x480
- 默认帧率：10fps

## 📊 技术栈

| 模块 | 技术 | 用途 |
|------|------|------|
| 姿态识别 | MediaPipe 0.10+ | 实时人体姿态检测与手势识别 |
| 目标检测 | YOLO-FastestV2 | 实时目标检测与跟踪 |
| 语音识别 | Sherpa-ONNX | 中文语音识别 |
| 视频流 | mjpg-streamer | HTTP MJPEG 视频流 |
| 控制框架 | ROS2 | 机器人控制节点 |
| 图像处理 | OpenCV | 图像处理与可视化 |

## 🐛 常见问题

### 1. 无法连接到树莓派

- 检查树莓派 IP 地址是否正确
- 确认树莓派和 PC 在同一网络
- 检查防火墙设置
- 验证视频流服务是否运行：浏览器访问 `http://树莓派IP:8080/?action=stream`

### 2. MediaPipe 导入错误

- 确保使用 Python 3.9+
- 重新安装 MediaPipe：`pip install --upgrade mediapipe`
- 检查环境是否正确激活

### 3. 视频流延迟或卡顿

- 降低视频流帧率：`-f 5` 或 `-f 10`
- 降低分辨率：`-r 320x240`
- 检查网络连接质量

### 4. 语音识别不准确

- 确保环境安静
- 检查麦克风是否正常工作
- 尝试调整 VAD 阈值

### 5. 时间戳错误（MediaPipe）

代码已自动处理时间戳递增问题。如果仍出现错误，检查系统时间是否正常。

## 📝 开发说明

### 添加新手势

在 `mediapipe_app/mediapipe_detect.py` 的 `recognize_gesture()` 函数中添加新的手势识别逻辑：

```python
def recognize_gesture(pose_landmarks):
    # ... 现有代码 ...
    
    # 添加新手势识别
    if your_condition:
        return "YOUR_GESTURE"
```

### 修改命令执行时间

在 `mediapipe_app/mediapipe_detect.py` 中修改：

```python
max_command_frames = 30  # 修改为你想要的帧数
```

### 调试视频流

在浏览器中访问：`http://树莓派IP:8080/?action=stream` 查看实时视频流。

## 📚 相关文档

- [MediaPipe 官方文档](https://developers.google.com/mediapipe)
- [YOLO-FastestV2 文档](Yolo-FastestV2/README.md)
- [语音控制模块文档](small-car/README.md)
- [命令参考](command_repository.txt)

## 🔄 更新日志

### v1.0.0
- ✅ 支持 MediaPipe 姿态识别控制
- ✅ 支持 YOLO 目标检测控制
- ✅ 支持语音控制
- ✅ 自动命令停止机制
- ✅ 错误处理与重连机制

## 📄 许可证

本项目仅供学习和研究使用。

## 👥 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题，请查看 `command_repository.txt` 或提交 Issue。

---

**注意**：使用前请确保树莓派和 PC 在同一网络，并正确配置 IP 地址。

