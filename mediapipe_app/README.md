## mediapipe_app

基于 MediaPipe Pose 的三/四类姿态控制小车示例（前进、左转、右转，新增：双手上举=后退）。

### 功能概述
- 从树莓派摄像头流读取视频（MJPEG/HTTP）。
- 使用 MediaPipe `pose_landmarker.task` 进行人体关键点检测。
- 手势识别并下发控制命令到树莓派控制服务：
  - FORWARD：T 字型（双臂水平展开）。
  - LEFT：左手在身体左侧平举（手腕明显在左肩左侧，接近肩高）。
  - RIGHT：右手在身体右侧平举（手腕明显在右肩右侧，接近肩高）。
  - BACKWARD：双手竖直上举超过肩（手腕高于肩，且水平位置靠近各自肩）。
- 命令持续时间：30 帧（约 1 秒@30FPS），期间检测到新手势会立即切换。

### 目录结构
- `mediapipe_detect.py`：主程序。
- `pose_landmarker.task`：MediaPipe 模型文件（放在 `embodiedcar/Mediapipe/`，代码里有绝对路径，可按需修改）。

### 环境要求
- Python 3.8+（建议与项目其余部分一致）。
- 依赖安装：

```bash
pip install opencv-python requests mediapipe numpy
```

说明：Windows 下 pip 版 MediaPipe 的 GPU delegate 常不可用，程序会自动回退到 CPU。即使机器有 NVIDIA GPU（如 RTX 5000），若 GPU delegate 初始化失败也会打印并使用 CPU 推理。

### 配置
在 `mediapipe_detect.py` 中根据你的环境调整：
- 摄像头流地址：

```python
stream_url = "http://<pi-ip>:8080/?action=stream"
```

- 控制服务地址：

```python
control_url = "http://<pi-ip>:5000/control"
```

- 模型路径：

```python
model_path = r"D:\embodiedcar\embodiedcar\Mediapipe\pose_landmarker.task"
```

若目录不同，请改为你的绝对/相对路径。

### 运行

```bash
python embodiedcar/mediapipe_app/mediapipe_detect.py
```

按 `q` 退出程序。

### 手势规则（关键阈值）
- 可见度阈值：`visibility > 0.3`（更宽容，抗抖动）。
- FORWARD（T 字型）：
  - 双腕高度接近且不低太多；左右相对各自肩部水平明显外展。
- LEFT（左转）：
  - `left_wrist.x < left_shoulder.x - 0.06`
  - `left_wrist.y <= left_shoulder.y + 0.12`
  - `distance(left_wrist, left_elbow) > 0.08`
- RIGHT（右转）：镜像 LEFT。
- BACKWARD（后退）：双腕高于各自肩并靠近各自肩的水平线：
  - `wrist.y < shoulder.y - 0.05`
  - `abs(wrist.x - shoulder.x) < 0.20`

阈值均为归一化坐标（相对图像宽/高）。如果需要更严或更松，可在 `recognize_gesture` 中微调相应数值。

### 命令持续与切换
- 变量 `max_command_frames = 30` 控制持续帧数。
- 期间检测到新手势会立即发送新命令并重置计数。
- 若持续满 `max_command_frames`，自动发送 `STOP`。

### 常见问题
- 仍显示 Using CPU inference：
  - Windows 下 GPU delegate 往往不可用（与显卡性能无关），程序会打印并回退 CPU。
  - 避免远程桌面，直连显示器；更新显卡驱动；尝试较新版本的 `mediapipe`。
- 识别不稳定或误判：
  - 调整 LEFT/RIGHT 的水平外展、腕-肘距离、垂直高度容差阈值。
  - 调整 FORWARD 的手腕与肩的高度/水平阈值。

### 依赖的控制服务
程序通过 `POST {control_url}` 发送 JSON：`{"command": "<GESTURE>"}`，其中 `<GESTURE>` 为 `FORWARD|LEFT|RIGHT|BACKWARD|STOP`。请确保树莓派端已实现对应控制接口。

### 许可证
本目录下代码遵循与项目相同的许可证条款。*** End Patch

