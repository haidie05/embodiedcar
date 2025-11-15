# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main scripts to run pose landmarker with traffic gesture recognition."""

import argparse
import sys
import time
import requests

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variable for detection result
DETECTION_RESULT = None

# Gesture types (using uppercase to match YOLO code format)
GESTURE_NONE = "none"
GESTURE_STOP = "STOP"      # 停车
GESTURE_FORWARD = "FORWARD"  # 直行
GESTURE_LEFT = "LEFT"      # 左转
GESTURE_RIGHT = "RIGHT"    # 右转

# MediaPipe Pose landmark indices (33 landmarks total, 0-32)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16


def recognize_gesture(pose_landmarks):
    """Recognize traffic gestures from pose landmarks.
    
    Args:
        pose_landmarks: List of pose landmarks from MediaPipe
        
    Returns:
        str: Recognized gesture (stop, forward, left, right, none)
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
    
    # Check visibility (z coordinate indicates depth, lower z means closer)
    def is_visible(landmark):
        return landmark.visibility > 0.5
    
    # Calculate distances and positions
    def get_distance(p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    # Check for STOP gesture: Both arms extended horizontally (T-pose)
    if (is_visible(left_wrist) and is_visible(right_wrist) and 
        is_visible(left_shoulder) and is_visible(right_shoulder)):
        # Both wrists should be at similar height (within threshold)
        wrist_height_diff = abs(left_wrist.y - right_wrist.y)
        # Both wrists should be higher than shoulders (arms raised)
        left_raised = left_wrist.y < left_shoulder.y
        right_raised = right_wrist.y < right_shoulder.y
        # Wrists should be horizontally extended (x distance from shoulders)
        left_extended = abs(left_wrist.x - left_shoulder.x) > 0.15
        right_extended = abs(right_wrist.x - right_shoulder.x) > 0.15
        
        if (wrist_height_diff < 0.1 and left_raised and right_raised and 
            left_extended and right_extended):
            return GESTURE_STOP
    
    # Check for LEFT turn: Left arm extended to the left
    if is_visible(left_wrist) and is_visible(left_shoulder) and is_visible(left_elbow):
        # Left wrist should be significantly to the left of left shoulder
        left_horizontal = left_wrist.x < left_shoulder.x - 0.1
        # Left arm should be extended (wrist far from elbow)
        left_arm_extended = get_distance(left_wrist, left_elbow) > 0.15
        # Left wrist should be at similar or higher height than shoulder
        left_vertical = left_wrist.y <= left_shoulder.y + 0.1
        
        if left_horizontal and left_arm_extended and left_vertical:
            return GESTURE_LEFT
    
    # Check for RIGHT turn: Right arm extended to the right
    if is_visible(right_wrist) and is_visible(right_shoulder) and is_visible(right_elbow):
        # Right wrist should be significantly to the right of right shoulder
        right_horizontal = right_wrist.x > right_shoulder.x + 0.1
        # Right arm should be extended (wrist far from elbow)
        right_arm_extended = get_distance(right_wrist, right_elbow) > 0.15
        # Right wrist should be at similar or higher height than shoulder
        right_vertical = right_wrist.y <= right_shoulder.y + 0.1
        
        if right_horizontal and right_arm_extended and right_vertical:
            return GESTURE_RIGHT
    
    # Check for FORWARD gesture: One arm extended forward
    # Forward gesture: arm extended forward (wrist closer to camera than shoulder)
    if is_visible(left_wrist) and is_visible(left_shoulder) and is_visible(left_elbow):
        # Left wrist should be forward (z coordinate indicates depth)
        # Lower z means closer to camera (forward)
        left_forward = left_wrist.z < left_shoulder.z - 0.05
        # Left arm should be extended (wrist far from elbow)
        left_arm_extended = get_distance(left_wrist, left_elbow) > 0.12
        # Wrist should be roughly in front of shoulder (similar x, similar or lower y)
        left_aligned = abs(left_wrist.x - left_shoulder.x) < 0.15
        
        if left_forward and left_arm_extended and left_aligned:
            return GESTURE_FORWARD
    
    if is_visible(right_wrist) and is_visible(right_shoulder) and is_visible(right_elbow):
        # Right wrist should be forward
        right_forward = right_wrist.z < right_shoulder.z - 0.05
        # Right arm should be extended
        right_arm_extended = get_distance(right_wrist, right_elbow) > 0.12
        # Wrist should be roughly in front of shoulder
        right_aligned = abs(right_wrist.x - right_shoulder.x) < 0.15
        
        if right_forward and right_arm_extended and right_aligned:
            return GESTURE_FORWARD
    
    return GESTURE_NONE


def send_command_to_raspberry_pi(control_url, command, timeout=1.0):
    """Send command to Raspberry Pi car.
    
    Args:
        control_url: Full URL for control API (e.g., http://172.20.10.7:5000/control)
        command: Command to send (STOP, FORWARD, LEFT, RIGHT)
        timeout: Request timeout in seconds
    """
    try:
        response = requests.post(control_url, json={"command": command}, timeout=timeout)
        if response.status_code == 200:
            print(f"Command '{command}' sent successfully")
        else:
            print(f"Failed to send command: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending command to Raspberry Pi: {e}")


def run(model: str, num_poses: int,
        min_pose_detection_confidence: float,
        min_pose_presence_confidence: float, min_tracking_confidence: float,
        output_segmentation_masks: bool,
        stream_url: str, target_fps: int,
        control_url: str, command_duration: float) -> None:
    """Continuously run inference on images acquired from network stream.

  Args:
      model: Name of the pose landmarker model bundle.
      num_poses: Max number of poses that can be detected by the landmarker.
      min_pose_detection_confidence: The minimum confidence score for pose
        detection to be considered successful.
      min_pose_presence_confidence: The minimum confidence score of pose
        presence score in the pose landmark detection.
      min_tracking_confidence: The minimum confidence score for the pose
        tracking to be considered successful.
      output_segmentation_masks: Choose whether to visualize the segmentation
        mask or not.
      stream_url: URL or path to video stream (network or local).
      target_fps: Target FPS for processing (10 fps).
      control_url: Full URL for control API (e.g., http://172.20.10.7:5000/control).
      command_duration: Duration in seconds for car to execute command before stopping.
  """

    # Start capturing video input from network stream
    print(f"Connecting to stream: {stream_url}")
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print(f"ERROR: Unable to open stream: {stream_url}")
        print("Trying common stream formats...")
        print("Please check the stream URL format (RTSP, HTTP, etc.)")
        sys.exit(1)
    
    # Calculate frame interval for target FPS
    frame_interval = 1.0 / target_fps
    last_frame_time = time.time()
    
    # Gesture tracking
    last_gesture = GESTURE_NONE
    last_sent_gesture = GESTURE_NONE  # Track last sent command to avoid duplicates
    gesture_stable_count = 0
    gesture_stable_threshold = 3  # Require 3 consecutive detections
    
    # Command execution timing
    command_start_time = None  # Time when command was sent

    # Visualization parameters
    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 255, 0)  # green for text
    font_size = 0.7
    font_thickness = 2
    fps_avg_frame_count = 10
    overlay_alpha = 0.5
    mask_color = (100, 100, 0)  # cyan
    
    # FPS calculation
    fps_start_time = time.perf_counter()
    fps_frame_count = 0
    current_fps = 0.0

    def save_result(result: vision.PoseLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global DETECTION_RESULT
        DETECTION_RESULT = result

    # Initialize the pose landmarker model
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=output_segmentation_masks,
        result_callback=save_result)
    detector = vision.PoseLandmarker.create_from_options(options)

    # Continuously capture images from the stream and run inference
    while cap.isOpened():
        # Control FPS by skipping frames
        current_time = time.time()
        if current_time - last_frame_time < frame_interval:
            # Skip this frame to maintain target FPS
            cap.grab()  # Discard frame without decoding
            continue
        last_frame_time = current_time
        
        success, image = cap.read()
        if not success:
            print("Warning: Failed to read frame, retrying...")
            time.sleep(0.1)
            continue

        # Don't flip network stream (already correct orientation)
        # image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run pose landmarker using the model.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Calculate FPS
        fps_frame_count += 1
        if fps_frame_count >= fps_avg_frame_count:
            fps_end_time = time.perf_counter()
            current_fps = fps_avg_frame_count / (fps_end_time - fps_start_time)
            fps_start_time = fps_end_time
            fps_frame_count = 0

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(current_fps)
        text_location = (left_margin, row_size)
        current_frame = image.copy()
        cv2.putText(current_frame, fps_text, text_location,
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)

        current_gesture = GESTURE_NONE
        
        if DETECTION_RESULT:
            # Draw landmarks and recognize gestures.
            for pose_landmarks in DETECTION_RESULT.pose_landmarks:
                # Draw the pose landmarks.
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                                    z=landmark.z) for landmark
                    in pose_landmarks
                ])
                mp_drawing.draw_landmarks(
                    current_frame,
                    pose_landmarks_proto,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style())
                
                # Recognize gesture
                gesture = recognize_gesture(pose_landmarks)
                if gesture != GESTURE_NONE:
                    current_gesture = gesture
                    break
        
        # Gesture stability check (avoid flickering)
        if current_gesture == last_gesture:
            gesture_stable_count += 1
        else:
            gesture_stable_count = 0
            last_gesture = current_gesture
        
        # Handle gesture detection and command sending
        current_time = time.time()
        
        # Check if current command has been executing long enough, then stop
        if (command_start_time is not None and 
            last_sent_gesture != GESTURE_NONE and 
            last_sent_gesture != GESTURE_STOP):
            elapsed_time = current_time - command_start_time
            if elapsed_time >= command_duration:
                # Command execution time reached, send STOP
                send_command_to_raspberry_pi(control_url, GESTURE_STOP)
                print(f"Command '{last_sent_gesture}' executed for {elapsed_time:.1f}s, stopping")
                last_sent_gesture = GESTURE_STOP
                command_start_time = None
        
        # Only send commands when gesture is detected
        if current_gesture != GESTURE_NONE:
            # Send command if gesture is stable and different from last sent command
            if (gesture_stable_count >= gesture_stable_threshold and 
                current_gesture != last_sent_gesture):
                send_command_to_raspberry_pi(control_url, current_gesture)
                last_sent_gesture = current_gesture  # Update last sent gesture
                command_start_time = current_time  # Record command start time
                print(f"Command '{current_gesture}' sent, will execute for {command_duration}s")
        # When no gesture detected, do nothing - car maintains current state
        
        # Display gesture on frame
        if current_gesture != GESTURE_NONE:
            gesture_text = f"Gesture: {current_gesture}"
            gesture_location = (left_margin, row_size + 30)
            gesture_color = (0, 255, 0)  # green
            
            # Show remaining execution time if command is active
            if (command_start_time is not None and 
                last_sent_gesture != GESTURE_NONE and 
                last_sent_gesture != GESTURE_STOP):
                elapsed = current_time - command_start_time
                remaining = max(0, command_duration - elapsed)
                time_text = f" ({remaining:.1f}s)"
                gesture_text += time_text
            
            cv2.putText(current_frame, gesture_text, gesture_location,
                        cv2.FONT_HERSHEY_DUPLEX,
                        font_size, gesture_color, font_thickness, cv2.LINE_AA)
        else:
            no_gesture_text = "No Gesture Detected"
            no_gesture_location = (left_margin, row_size + 30)
            no_gesture_color = (0, 165, 255)  # orange
            cv2.putText(current_frame, no_gesture_text, no_gesture_location,
                        cv2.FONT_HERSHEY_DUPLEX,
                        font_size, no_gesture_color, font_thickness, cv2.LINE_AA)

        if (output_segmentation_masks and DETECTION_RESULT):
            if DETECTION_RESULT.segmentation_masks is not None:
                segmentation_mask = DETECTION_RESULT.segmentation_masks[0].numpy_view()
                mask_image = np.zeros(image.shape, dtype=np.uint8)
                mask_image[:] = mask_color
                condition = np.stack((segmentation_mask,) * 3, axis=-1) > 0.1
                visualized_mask = np.where(condition, mask_image, current_frame)
                current_frame = cv2.addWeighted(current_frame, overlay_alpha,
                                                visualized_mask, overlay_alpha,
                                                0)

        cv2.imshow('Traffic Gesture Recognition', current_frame)

        # Stop the program if the 'q' key is pressed (matching YOLO code)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Traffic gesture recognition using MediaPipe Pose')
    parser.add_argument(
        '--model',
        help='Name of the pose landmarker model bundle.',
        required=False,
        default='pose_landmarker.task')
    parser.add_argument(
        '--numPoses',
        help='Max number of poses that can be detected by the landmarker.',
        required=False,
        default=1)
    parser.add_argument(
        '--minPoseDetectionConfidence',
        help='The minimum confidence score for pose detection to be considered '
             'successful.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minPosePresenceConfidence',
        help='The minimum confidence score of pose presence score in the pose '
             'landmark detection.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minTrackingConfidence',
        help='The minimum confidence score for the pose tracking to be '
             'considered successful.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--outputSegmentationMasks',
        help='Set this if you would also like to visualize the segmentation '
             'mask.',
        required=False,
        action='store_true')
    parser.add_argument(
        '--streamUrl',
        help='URL or path to video stream (e.g., http://172.20.10.7:8080/?action=stream)',
        required=False,
        default='http://172.20.10.7:8080/?action=stream')
    parser.add_argument(
        '--targetFPS',
        help='Target FPS for processing (default: 10)',
        required=False,
        default=10,
        type=int)
    parser.add_argument(
        '--controlUrl',
        help='Full URL for control API (e.g., http://172.20.10.7:5000/control)',
        required=False,
        default='http://172.20.10.7:5000/control')
    parser.add_argument(
        '--commandDuration',
        help='Duration in seconds for car to execute command before stopping (default: 2.0)',
        required=False,
        default=2.0,
        type=float)
    args = parser.parse_args()

    run(args.model, int(args.numPoses), args.minPoseDetectionConfidence,
        args.minPosePresenceConfidence, args.minTrackingConfidence,
        args.outputSegmentationMasks,
        args.streamUrl, args.targetFPS,
        args.controlUrl, args.commandDuration)


if __name__ == '__main__':
    main()