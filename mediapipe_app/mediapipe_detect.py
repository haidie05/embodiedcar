import cv2
import time
import requests
import numpy as np
import sys; print(sys.executable)
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 检查 CUDA 是否可用（用于参考，MediaPipe 有自己的 GPU delegate）
try:
    import torch
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available")
except ImportError:
    use_gpu = False
    print("PyTorch not installed, cannot check CUDA availability")

# MediaPipe 0.10+ 不再有 solutions API，需要手动绘制关键点
# 定义姿态连接关系（用于绘制骨架）
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # 面部
    (0, 4), (4, 5), (5, 6), (6, 8),  # 面部
    (9, 10),  # 肩部
    (11, 12),  # 肩膀
    (11, 13), (13, 15),  # 左臂
    (12, 14), (14, 16),  # 右臂
    (11, 23), (12, 24),  # 躯干上部
    (23, 24),  # 躯干中部
    (23, 25), (25, 27),  # 左腿
    (24, 26), (26, 28),  # 右腿
    (27, 29), (27, 31),  # 左脚
    (28, 30), (28, 32),  # 右脚
]

# 这里请替换为你的树莓派 IP 地址
stream_url = "http://172.20.10.7:8080/?action=stream"    # 树莓派摄像头流地址
control_url = "http://172.20.10.7:5000/control"          # 树莓派控制API地址
# MediaPipe模型路径 - 使用相对于脚本文件的路径
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "pose_landmarker.task")

cap = cv2.VideoCapture(stream_url)                       # 获取视频的输入源，也就是这个网址

# 检查视频流连接
if not cap.isOpened():
    print(f"⚠️  警告: 无法打开视频流 {stream_url}")
    print("   请检查：1) 树莓派是否开机 2) IP地址是否正确 3) 视频流服务是否运行")
    print("   程序将继续运行，但可能无法获取视频帧")
else:
    print(f"✓ 视频流连接成功: {stream_url}")

# Gesture types (using uppercase to match YOLO code format)
GESTURE_NONE = "none"
GESTURE_STOP = "STOP"      # 停车
GESTURE_FORWARD = "FORWARD"  # 直行
GESTURE_LEFT = "LEFT"      # 左转
GESTURE_RIGHT = "RIGHT"    # 右转
GESTURE_BACKWARD = "BACKWARD"  # 后退（双手上举）

# MediaPipe Pose landmark indices (33 landmarks total, 0-32)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

# Global variable for detection result
DETECTION_RESULT = None

# 手势识别函数（仅三种姿态：前进/左转/右转；否则不输出手势，由上层默认停止）
def recognize_gesture(pose_landmarks):
    """识别三种姿态：
    - T 字型（双臂水平张开） => FORWARD
    - 左手在身体左侧水平平举 => LEFT
    - 右手在身体右侧水平平举 => RIGHT
    其它情况返回 none（由上层逻辑停止）
    """
    if not pose_landmarks or len(pose_landmarks) < 33:
        return GESTURE_NONE
    
    # Get key landmarks
    left_shoulder = pose_landmarks[LEFT_SHOULDER]
    right_shoulder = pose_landmarks[RIGHT_SHOULDER]
    left_elbow = pose_landmarks[LEFT_ELBOW]
    right_elbow = pose_landmarks[RIGHT_ELBOW]
    left_wrist = pose_landmarks[LEFT_WRIST]
    right_wrist = pose_landmarks[RIGHT_WRIST]
    
    # Check visibility（放宽可见度门槛）
    def is_visible(landmark):
        return landmark.visibility > 0.3
    
    # Calculate distances
    def get_distance(p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    # T 字型 => 前进（放宽判定）
    # 允许左右手腕高度差更大；手腕可略低于肩（+0.05）；水平伸展距离阈值稍小
    if (is_visible(left_wrist) and is_visible(right_wrist) and 
        is_visible(left_shoulder) and is_visible(right_shoulder)):
        wrist_height_diff = abs(left_wrist.y - right_wrist.y)
        left_raised = left_wrist.y < left_shoulder.y + 0.05
        right_raised = right_wrist.y < right_shoulder.y + 0.05
        left_extended = abs(left_wrist.x - left_shoulder.x) > 0.10
        right_extended = abs(right_wrist.x - right_shoulder.x) > 0.10
        
        if (wrist_height_diff < 0.18 and left_raised and right_raised and 
            left_extended and right_extended):
            return GESTURE_FORWARD
    
    # 左转：左手向左侧平举（以手腕与肩的相对位置为主）
    # 条件：左腕明显在左肩左侧，且高度接近或略高于肩；为避免误触，要求腕-肘略有伸展
    if is_visible(left_wrist) and is_visible(left_shoulder) and is_visible(left_elbow):
        left_horizontal = left_wrist.x < left_shoulder.x - 0.06
        left_vertical = left_wrist.y <= left_shoulder.y + 0.12
        left_min_extension = get_distance(left_wrist, left_elbow) > 0.08
        if left_horizontal and left_vertical and left_min_extension:
            return GESTURE_LEFT
    
    # 右转：右手向右侧平举（镜像条件）
    if is_visible(right_wrist) and is_visible(right_shoulder) and is_visible(right_elbow):
        right_horizontal = right_wrist.x > right_shoulder.x + 0.06
        right_vertical = right_wrist.y <= right_shoulder.y + 0.12
        right_min_extension = get_distance(right_wrist, right_elbow) > 0.08
        if right_horizontal and right_vertical and right_min_extension:
            return GESTURE_RIGHT

    # 后退：双手竖直上举过头顶（双腕显著高于肩，且水平方向接近肩，表示“上举”而非“外展”）
    if (is_visible(left_wrist) and is_visible(right_wrist) and
        is_visible(left_shoulder) and is_visible(right_shoulder)):
        left_up = left_wrist.y < left_shoulder.y - 0.05
        right_up = right_wrist.y < right_shoulder.y - 0.05
        left_near_shoulder_x = abs(left_wrist.x - left_shoulder.x) < 0.20
        right_near_shoulder_x = abs(right_wrist.x - right_shoulder.x) < 0.20
        if left_up and right_up and left_near_shoulder_x and right_near_shoulder_x:
            return GESTURE_BACKWARD
    
    return GESTURE_NONE

def send_command(gesture):
    """发送命令给树莓派"""
    try:
        response = requests.post(control_url, json={'command': gesture}, timeout=2)
        response.raise_for_status()  # 检查HTTP错误
        if gesture == GESTURE_STOP:
            print("Stop")
        elif gesture == GESTURE_LEFT:
            print("Turn left")
        elif gesture == GESTURE_RIGHT:
            print("Turn right")
        elif gesture == GESTURE_FORWARD:
            print("Go straight")
        elif gesture == GESTURE_BACKWARD:
            print("Go backward")
    except requests.exceptions.ConnectionError as e:
        print(f"⚠️  无法连接到树莓派控制API ({control_url}): 连接被拒绝")
        print(f"   请检查：1) 树莓派是否开机 2) IP地址是否正确 3) 控制服务是否运行")
    except requests.exceptions.Timeout:
        print(f"⚠️  连接超时: 无法在2秒内连接到 {control_url}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️  请求错误: {e}")
    except Exception as e:
        print(f"⚠️  发送命令时发生未知错误: {e}")

# 模型加载：如果你的电脑上有 GPU，会自动使用 GPU加速；如果没有 GPU，会使用 CPU
def create_detector():
    def save_result(result: vision.PoseLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global DETECTION_RESULT
        DETECTION_RESULT = result
    
    # 创建 options 的公共函数（避免重复代码）
    def create_options(use_gpu_delegate=False):
        base_options = python.BaseOptions(model_asset_path=model_path)
        if use_gpu_delegate:
            try:
                base_options.delegate = python.BaseOptions.Delegate.GPU
            except Exception:
                use_gpu_delegate = False
        return vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=save_result), use_gpu_delegate
    
    # 优先尝试使用 MediaPipe GPU delegate（与 CUDA 无强关联），失败再回退 CPU
    # MediaPipe 的 GPU delegate 在 Windows 上可能有限制，但先尝试
    print("Attempting to initialize MediaPipe with GPU delegate...")
    try:
        options, is_gpu = create_options(use_gpu_delegate=True)
        detector = vision.PoseLandmarker.create_from_options(options)
        if is_gpu:
            print("✓ GPU acceleration enabled successfully")
        else:
            print("✓ Using CPU inference (GPU delegate not available)")
        return detector
    except Exception as e:
        print(f"Failed to initialize GPU delegate: {type(e).__name__}: {e}")
        print("Falling back to CPU...")
    
    # Fallback to CPU
    options, _ = create_options(use_gpu_delegate=False)
    detector = vision.PoseLandmarker.create_from_options(options)
    print("✓ Using CPU inference")
    return detector

detector = create_detector()

# 发送状态
last_sent_gesture = GESTURE_NONE

# 命令执行帧数控制
command_frame_count = 0   # 当前命令已执行的帧数
max_command_frames = 30   # 每个命令最多执行的帧数（例如30帧，假设30fps约1.0秒），达到后自动停止

# 时间戳管理：确保时间戳单调递增
last_timestamp_ms = 0

while True:
    ret, frame = cap.read()                             # 不停地从输入源读取一帧图像
    if not ret:                                         # 如果读取失败，结束循环
        print("⚠️  无法读取视频帧，可能视频流已断开")
        print("   程序将在1秒后重试...")
        time.sleep(1)
        # 尝试重新连接
        cap.release()
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print("   重连失败，继续尝试...")
        continue
    
    # 将图片转换为 MediaPipe 需要的格式
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    # 运行 MediaPipe 模型，得到输出
    # 确保时间戳单调递增（MediaPipe 异步模式要求）
    current_timestamp_ms = time.time_ns() // 1_000_000
    if current_timestamp_ms <= last_timestamp_ms:
        # 如果时间戳没有增加，强制增加1毫秒
        current_timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = current_timestamp_ms
    
    try:
        detector.detect_async(mp_image, current_timestamp_ms)
    except ValueError as e:
        if "monotonically increasing" in str(e):
            print(f"⚠️  时间戳错误，跳过此帧: {e}")
            continue
        else:
            raise
    
    # 识别手势
    current_gesture = GESTURE_NONE
    if DETECTION_RESULT:
        for pose_landmarks in DETECTION_RESULT.pose_landmarks:
            # 绘制姿态关键点（使用 OpenCV 手动绘制，因为 MediaPipe 0.10+ 不再有 solutions API）
            h, w, _ = frame.shape
            landmark_points = []
            
            for landmark in pose_landmarks:
                if landmark.visibility > 0.5:  # 只绘制可见的关键点
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmark_points.append((x, y))
                    # 绘制关键点
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                else:
                    landmark_points.append(None)
            
            # 绘制连接线
            for connection in POSE_CONNECTIONS:
                pt1_idx, pt2_idx = connection
                if (pt1_idx < len(landmark_points) and pt2_idx < len(landmark_points) and
                    landmark_points[pt1_idx] is not None and landmark_points[pt2_idx] is not None):
                    pt1 = landmark_points[pt1_idx]
                    pt2 = landmark_points[pt2_idx]
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
            # 识别手势
            gesture = recognize_gesture(pose_landmarks)
            if gesture != GESTURE_NONE:
                current_gesture = gesture
                break

    # 检查当前命令是否已经执行了足够帧数，如果是则停止
    if (command_frame_count >= max_command_frames and 
        last_sent_gesture != GESTURE_NONE and 
        last_sent_gesture != GESTURE_STOP):
        # 执行帧数达到上限，发送停止命令
        send_command(GESTURE_STOP)
        print(f"Command '{last_sent_gesture}' executed for {command_frame_count} frames, stopping")
        last_sent_gesture = GESTURE_STOP
        command_frame_count = 0  # 重置计数
    
    # 发送命令：检测到手势即刻发送（移除稳定门限），与上次不同则更新并重置帧计数
    if (current_gesture != GESTURE_NONE and current_gesture != last_sent_gesture):
        send_command(current_gesture)
        last_sent_gesture = current_gesture
        command_frame_count = 0  # 重置帧计数，开始新的命令
        print(f"Command '{current_gesture}' sent, will execute for {max_command_frames} frames")
    
    # 如果命令正在执行，增加帧计数
    if (last_sent_gesture != GESTURE_NONE and 
        last_sent_gesture != GESTURE_STOP):
        command_frame_count += 1
    
    # 显示手势信息
    if current_gesture != GESTURE_NONE:
        gesture_text = f"Gesture: {current_gesture}"
        # 显示剩余帧数
        if (last_sent_gesture != GESTURE_NONE and 
            last_sent_gesture != GESTURE_STOP and
            command_frame_count > 0):
            remaining_frames = max(0, max_command_frames - command_frame_count)
            gesture_text += f" ({remaining_frames} frames left)"
        cv2.putText(frame, gesture_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Gesture Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
    
    # 显示图像
    cv2.imshow('Traffic Gesture Recognition', frame)      # 显示获取到的图像
    
    if cv2.waitKey(1) & 0xFF == ord('q'):               # 检测 q 键有没有按下，按下就退出程序
        break

detector.close()
cap.release()                                           # 释放视频输入源
cv2.destroyAllWindows()
