import os
import cv2
import time
import argparse
import requests
import torch
import model.detector
import utils.utils

stream_url = "http://172.20.10.7:8080/?action=stream"    # 这里请替换为你的树莓派 IP 地址
control_url = "http://172.20.10.7:5000/control"          # 这里请替换为你的树莓派 IP 地址
cap = cv2.VideoCapture(stream_url)                       # 获取视频的输入源，也就是这个网址

cfg = utils.utils.load_datafile(r'E:\embodiedcar\Yolo-FastestV2\data\coco.data')
weights = r'E:\embodiedcar\Yolo-FastestV2\modelzoo\coco2017-0.241078ap-model.pth'
assert os.path.exists(weights), "请指定正确的模型路径"

target_categories = ["chair"]                           # 检测椅子

# ========== 多椅子遍历配置 ==========
ARRIVED_THRESHOLD = 0.6  # 到达椅子的面积阈值（检测框面积/图像面积的比值）
MIN_DISTANCE_TO_CONSIDER_NEW = 0.1  # 两个椅子之间的距离阈值（用于判断是否为同一个椅子）
SEARCH_COMMANDS = ["LEFT", "STOP"]  # 搜索模式时的命令序列
search_command_index = 0

# 已访问的椅子列表（存储椅子的中心坐标，用于去重）
visited_chairs = []

# 当前目标椅子（None 表示正在搜索）
current_target_chair = None

# 模型加载：如果你的电脑上有 GPU，可以用第一行；如果你的电脑有 Apple M 系列芯片，可以用第二行
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"

model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
model.load_state_dict(torch.load(weights, map_location=device))
model.eval()

def get_chair_id(box):
    """获取椅子的唯一标识（使用中心坐标）"""
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    return (x_center, y_center)

def is_visited(chair_id):
    """检查椅子是否已被访问"""
    for visited_id in visited_chairs:
        # 计算两个椅子之间的距离
        distance = ((chair_id[0] - visited_id[0])**2 + (chair_id[1] - visited_id[1])**2)**0.5
        if distance < MIN_DISTANCE_TO_CONSIDER_NEW:
            return True
    return False

def get_distance_to_center(box, img_w, img_h):
    """计算椅子中心到图像中心的距离（归一化）"""
    chair_x = (box[0] + box[2]) / 2
    chair_y = (box[1] + box[3]) / 2
    img_center_x = img_w / 2
    img_center_y = img_h / 2
    distance = ((chair_x - img_center_x)**2 + (chair_y - img_center_y)**2)**0.5
    # 归一化到 [0, 1]
    max_distance = ((img_w/2)**2 + (img_h/2)**2)**0.5
    return distance / max_distance if max_distance > 0 else 0

def find_nearest_chair(chairs, img_w, img_h):
    """找到最近的椅子（面积最大且未被访问）"""
    unvisited_chairs = [chair for chair in chairs if not is_visited(get_chair_id(chair))]
    
    if len(unvisited_chairs) == 0:
        return None
    
    # 找到面积最大的椅子（面积越大说明越近）
    nearest = max(unvisited_chairs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return nearest

def send_command_to_chair(box, h, w):
    """向目标椅子移动"""
    x_center = (box[0] + box[2]) / 2 / w                          # 检测框的中心点
    box_area = (box[2] - box[0]) * (box[3] - box[1]) / (h * w)    # 检测框的面积
    print(f"Target chair - box area: {box_area:.3f}, x_center: {x_center:.3f}")
    
    # 如果已经到达椅子跟前，标记为已访问并开始搜索下一个
    if box_area > ARRIVED_THRESHOLD:      
        chair_id = get_chair_id(box)
        if not is_visited(chair_id):
            visited_chairs.append(chair_id)
            print(f"✓ Arrived at chair! Total visited: {len(visited_chairs)}")
            global current_target_chair
            current_target_chair = None
        response = requests.post(control_url, json={'command': "STOP"})
        print("Stop - Arrived at chair")
    else:
        # 如果椅子的中心点在左边，就左转
        if x_center < 0.3:
            response = requests.post(control_url, json={'command': "LEFT"})            
            print("Turn left")
        # 如果椅子的中心点在右边，就右转
        elif x_center > 0.7:
            response = requests.post(control_url, json={'command': "RIGHT"})            
            print("Turn right")
        # 如果椅子的中心点在中间，就前进
        else:
            response = requests.post(control_url, json={'command': "FORWARD"})            
            print("Go straight")

def search_next_chair():
    """搜索下一个椅子的模式（原地转圈）"""
    global search_command_index
    command = SEARCH_COMMANDS[search_command_index]
    search_command_index = (search_command_index + 1) % len(SEARCH_COMMANDS)
    response = requests.post(control_url, json={'command': command})
    print(f"Searching for next chair: {command}")
    

while True:
    ret, frame = cap.read()                             # 不停地从输入源读取一帧图像
    if not ret:                                         # 如果读取失败，结束循环
        break
    # 将图片处理成 YOLO 模型的输入
    res_img = cv2.resize(frame, (cfg["width"], cfg["height"]), interpolation = cv2.INTER_LINEAR) 
    img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
    img = torch.from_numpy(img.transpose(0, 3, 1, 2))
    img = img.to(device).float() / 255.0
    # 运行 YOLO 模型，得到输出
    preds = model(img)
    # 对输出进行后处理，例如过滤掉重叠的框和置信度低的框
    output = utils.utils.handel_preds(preds, cfg, device)
    output_boxes = utils.utils.non_max_suppression(output, conf_thres = 0.3, iou_thres = 0.4)

    #加载label names
    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())
    h, w, _ = frame.shape
    scale_h, scale_w = h / cfg["height"], w / cfg["width"]

    # 筛选出所有检测到的椅子
    detected_chairs = []
    all_boxes_info = []
    
    for box in output_boxes[0]:
        box = box.tolist()
        obj_score = box[4]
        category = LABEL_NAMES[int(box[5])]
        x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
        x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)
        
        all_boxes_info.append({
            'box': box,
            'category': category,
            'score': obj_score,
            'coords': (x1, y1, x2, y2)
        })
        
        if category in target_categories:
            detected_chairs.append(box)

    # 如果检测到椅子
    if len(detected_chairs) > 0:
        # 如果有当前目标且该目标仍在视野中，继续追踪
        if current_target_chair is not None:
            target_id = get_chair_id(current_target_chair)
            # 检查当前目标是否还在视野中
            found_target = False
            for chair in detected_chairs:
                chair_id = get_chair_id(chair)
                distance = ((chair_id[0] - target_id[0])**2 + (chair_id[1] - target_id[1])**2)**0.5
                if distance < MIN_DISTANCE_TO_CONSIDER_NEW:
                    found_target = True
                    current_target_chair = chair
                    send_command_to_chair(chair, cfg["height"], cfg["width"])
                    break
            
            # 如果目标丢失或已完成，寻找新的最近椅子
            if not found_target or current_target_chair is None:
                nearest_chair = find_nearest_chair(detected_chairs, cfg["width"], cfg["height"])
                if nearest_chair is not None:
                    current_target_chair = nearest_chair
                    chair_id = get_chair_id(nearest_chair)
                    print(f"New target chair selected. Visited: {len(visited_chairs)}")
                    send_command_to_chair(nearest_chair, cfg["height"], cfg["width"])
                else:
                    # 所有椅子都已访问，或者正在搜索
                    current_target_chair = None
                    search_next_chair()
        else:
            # 没有当前目标，寻找最近的椅子
            nearest_chair = find_nearest_chair(detected_chairs, cfg["width"], cfg["height"])
            if nearest_chair is not None:
                current_target_chair = nearest_chair
                chair_id = get_chair_id(nearest_chair)
                print(f"Target chair found. Visited: {len(visited_chairs)}")
                send_command_to_chair(nearest_chair, cfg["height"], cfg["width"])
            else:
                # 所有椅子都已访问，继续搜索
                search_next_chair()
    else:
        # 没有检测到椅子，搜索模式
        current_target_chair = None
        search_next_chair()

    # 绘制所有检测框
    for info in all_boxes_info:
        x1, y1, x2, y2 = info['coords']
        category = info['category']
        score = info['score']
        box = info['box']
        
        # 判断是否为当前目标椅子
        is_target = False
        if current_target_chair is not None and category in target_categories:
            target_id = get_chair_id(current_target_chair)
            chair_id = get_chair_id(box)
            distance = ((chair_id[0] - target_id[0])**2 + (chair_id[1] - target_id[1])**2)**0.5
            if distance < MIN_DISTANCE_TO_CONSIDER_NEW:
                is_target = True
        
        # 判断是否已访问
        is_visited_chair = False
        if category in target_categories:
            chair_id = get_chair_id(box)
            is_visited_chair = is_visited(chair_id)
        
        # 根据状态选择颜色
        if is_target:
            color = (0, 255, 255)  # 黄色：当前目标
            label = f"{category} (TARGET)"
        elif is_visited_chair:
            color = (128, 128, 128)  # 灰色：已访问
            label = f"{category} (VISITED)"
        else:
            color = (255, 255, 0)  # 青色：未访问
            label = category
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, '%.2f' % score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)    
        cv2.putText(frame, label, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
    
    # 在图像上显示状态信息
    status_text = f"Visited: {len(visited_chairs)} | Target: {'Found' if current_target_chair is not None else 'Searching'}"
    cv2.putText(frame, status_text, (10, 30), 0, 0.7, (255, 255, 255), 2)
    processed_frame = frame
    # 显示图像
    cv2.imshow('Processed Frame', processed_frame)      # 显示获取到的图像

    end = time.perf_counter()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):               # 检测 q 键有没有按下，按下就退出程序
        break

cap.release()                                           # 释放视频输入源
cv2.destroyAllWindows() 