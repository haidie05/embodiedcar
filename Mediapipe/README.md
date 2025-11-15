# 交通手势识别系统

基于MediaPipe姿态检测的交通手势识别系统，可以识别交警的停车、直行、左转、右转手势，并控制树莓派小车。

## 功能特性

- 从网络流读取视频（支持RTSP、HTTP等协议）
- 10帧/秒的处理速度
- 识别四种交通手势：
  - **停车 (STOP)**: 双手平举（T字姿势）
  - **直行 (FORWARD)**: 单手指向前方
  - **左转 (LEFT)**: 左手向左指
  - **右转 (RIGHT)**: 右手向右指
- 自动向树莓派小车发送控制命令

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python mediapipe_detect.py
```

### 自定义参数

```bash
python mediapipe_detect.py \
    --streamUrl rtsp://172.20.10.7:8554/stream \
    --targetFPS 10 \
    --raspberryPiIp 172.20.10.7 \
    --raspberryPiPort 8080 \
    --model pose_landmarker.task
```

### 参数说明

- `--streamUrl`: 视频流URL（默认: `rtsp://172.20.10.7:8554/stream`）
  - RTSP流: `rtsp://172.20.10.7:8554/stream`
  - HTTP流: `http://172.20.10.7:8080/stream`
  - MJPEG流: `http://172.20.10.7:8080/mjpeg_stream`
- `--targetFPS`: 目标处理帧率（默认: 10）
- `--raspberryPiIp`: 树莓派IP地址（默认: `172.20.10.7`）
- `--raspberryPiPort`: 树莓派命令API端口（默认: 8080）
- `--model`: 姿态检测模型文件（默认: `pose_landmarker.task`）
- `--minPoseDetectionConfidence`: 姿态检测最小置信度（默认: 0.5）
- `--minPosePresenceConfidence`: 姿态存在最小置信度（默认: 0.5）
- `--minTrackingConfidence`: 跟踪最小置信度（默认: 0.5）

## 手势识别说明

### 停车手势 (STOP)
- 双手平举，形成T字姿势
- 双手腕高度相近，高于肩膀
- 双手水平伸展

### 直行手势 (FORWARD)
- 单手指向前方（手臂向前伸出）
- 手腕在肩膀前方（z坐标更小）

### 左转手势 (LEFT)
- 左手向左指
- 左手腕在左肩左侧
- 左臂伸展

### 右转手势 (RIGHT)
- 右手向右指
- 右手腕在右肩右侧
- 右臂伸展

## 树莓派API要求

程序会向树莓派发送HTTP POST请求到 `http://<raspberryPiIp>:<raspberryPiPort>/command`

请求格式：
```json
{
    "command": "stop" | "forward" | "left" | "right"
}
```

树莓派需要实现相应的API端点来接收这些命令。

## 注意事项

1. 确保树莓派摄像头已正确配置并可以访问
2. 确保网络连接正常，可以访问树莓派IP
3. 确保树莓派上运行了接收命令的API服务
4. 手势需要保持稳定（连续3帧检测到相同手势）才会发送命令
5. 如果视频流无法连接，请检查流URL格式和网络设置

## 故障排除

### 无法连接视频流
- 检查IP地址和端口是否正确
- 尝试不同的流协议（RTSP、HTTP等）
- 检查防火墙设置

### 手势识别不准确
- 调整 `--minPoseDetectionConfidence` 参数
- 确保光线充足，人物清晰可见
- 确保手势动作标准

### 命令发送失败
- 检查树莓派IP和端口是否正确
- 确认树莓派API服务正在运行
- 检查网络连接

