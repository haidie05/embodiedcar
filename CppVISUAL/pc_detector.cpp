#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cpprest/http_client.h>
#include <cpprest/json.h>

using namespace web;
using namespace web::http;
using namespace web::http::client;

class YOLODetector {
private:
    cv::dnn::Net net_;
    std::vector<std::string> class_names_;
    float conf_threshold_ = 0.3;
    float nms_threshold_ = 0.4;
    int input_width_ = 416;
    int input_height_ = 416;
    
    http_client control_client_;
    
public:
    YOLODetector(const std::string& model_path, const std::string& classes_path, 
                 const std::string& control_url)
        : control_client_(U(control_url)) {
        
        // 加载YOLO模型
        net_ = cv::dnn::readNet(model_path);
        
        // 设置计算后端（根据你的环境选择）
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        // 加载类别名称
        loadClassNames(classes_path);
    }
    
    void loadClassNames(const std::string& classes_path) {
        std::ifstream file(classes_path);
        std::string line;
        while (std::getline(file, line)) {
            class_names_.push_back(line);
        }
    }
    
    void sendCommand(const std::string& command) {
        json::value post_data;
        post_data["command"] = json::value::string(command);
        
        control_client_.request(methods::POST, U("/control"), post_data)
            .then([](http_response response) {
                if (response.status_code() == status_codes::OK) {
                    std::cout << "Command sent successfully" << std::endl;
                } else {
                    std::cout << "Failed to send command: " << response.status_code() << std::endl;
                }
            })
            .wait();
    }
    
    void processFrame(cv::Mat& frame) {
        // 预处理
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(input_width_, input_height_), 
                              cv::Scalar(0,0,0), true, false);
        
        net_.setInput(blob);
        
        // 推理
        std::vector<cv::Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());
        
        // 后处理
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        
        float* data = (float*)outputs[0].data;
        
        for (int i = 0; i < outputs[0].rows; i++) {
            float confidence = data[4];
            if (confidence > conf_threshold_) {
                // 找到最大概率的类别
                float* classes_scores = data + 5;
                cv::Mat scores(1, class_names_.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;
                cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
                
                if (max_class_score > conf_threshold_) {
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);
                    
                    // 计算边界框坐标
                    float center_x = data[0] * frame.cols;
                    float center_y = data[1] * frame.rows;
                    float width = data[2] * frame.cols;
                    float height = data[3] * frame.rows;
                    
                    int left = int(center_x - width / 2);
                    int top = int(center_y - height / 2);
                    
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
            data += outputs[0].cols;
        }
        
        // 非极大值抑制
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);
        
        // 处理检测结果
        bool person_detected = false;
        for (int idx : indices) {
            int class_id = class_ids[idx];
            if (class_names_[class_id] == "person") {
                person_detected = true;
                cv::Rect box = boxes[idx];
                
                // 计算中心点和面积
                float x_center = (box.x + box.width / 2.0) / frame.cols;
                float box_area = (box.width * box.height) / (frame.cols * frame.rows);
                
                std::cout << "Box area: " << box_area << ", X center: " << x_center << std::endl;
                
                // 决策逻辑
                if (box_area > 0.7) {
                    sendCommand("STOP");
                    std::cout << "Stop" << std::endl;
                } else {
                    if (x_center < 0.3) {
                        sendCommand("LEFT");
                        std::cout << "Turn left" << std::endl;
                    } else if (x_center > 0.7) {
                        sendCommand("RIGHT");
                        std::cout << "Turn right" << std::endl;
                    } else {
                        sendCommand("FORWARD");
                        std::cout << "Go straight" << std::endl;
                    }
                }
                
                // 绘制边界框
                cv::rectangle(frame, box, cv::Scalar(255, 255, 0), 2);
                std::string label = class_names_[class_id] + ": " + std::to_string(confidences[idx]);
                cv::putText(frame, label, cv::Point(box.x, box.y - 10), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }
        }
        
        if (!person_detected) {
            sendCommand("STOP");
            std::cout << "No person detected - Stop" << std::endl;
        }
    }
    
    void runDetection(const std::string& stream_url) {
        cv::VideoCapture cap(stream_url);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open video stream: " << stream_url << std::endl;
            return;
        }
        
        cv::Mat frame;
        while (true) {
            cap >> frame;
            if (frame.empty()) break;
            
            processFrame(frame);
            
            cv::imshow("YOLO Detection", frame);
            
            if (cv::waitKey(1) == 'q') break;
        }
        
        cap.release();
        cv::destroyAllWindows();
    }
};

int main() {
    // 配置参数 - 根据你的实际路径修改
    std::string model_path = "path/to/your/yolo_model.onnx";  // 需要转换为ONNX格式
    std::string classes_path = "D:\embodiedcar\embodiedcar\CppVISUAL\data\coco.names";
    std::string stream_url = "http://172.20.10.7:8080/?action=stream";
    std::string control_url = "http://172.20.10.7:5000";
    
    YOLODetector detector(model_path, classes_path, control_url);
    detector.runDetection(stream_url);
    
    return 0;
}