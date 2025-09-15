#include <fstream>
#include <sstream>
#include <iostream>

#include "../../include/yolov5x/yolo_detector.hpp"
#include "../../include/device_type.hpp"

YOLODetector::YOLODetector(const std::string& model_path, 
                           DeviceType device_type,
                           float conf_threshold,
                           float nms_threshold)
    : conf_threshold_(conf_threshold), 
      nms_threshold_(nms_threshold),
      device_type_(device_type) {
    
    // Load network
    try {
        // Load network
        net_ = cv::dnn::readNetFromONNX(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        throw;
    }
    
    // Set backend
    std::vector<std::pair<cv::dnn::Backend, cv::dnn::Target>> availableBackends = cv::dnn::getAvailableBackends();
    bool cudaBackendAvailable = false;
    for (const auto& pair : availableBackends) {
        if (pair.first == cv::dnn::DNN_BACKEND_CUDA && pair.second == cv::dnn::DNN_TARGET_CUDA) {
            cudaBackendAvailable = true;
            break;
        }
    }
    std::cout << "CUDA BACKEND AVAILABLITY: " << cudaBackendAvailable << std::endl;

    // Set backend based on availability, not just request
    if (device_type == DeviceType::CUDA && cudaBackendAvailable) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        std::cout << "Using CUDA backend" << std::endl;
    } else {
        if (device_type == DeviceType::CUDA) {
            std::cerr << "CUDA backend requested but not available. Falling back to CPU." << std::endl;
        }
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        device_type_ = DeviceType::CPU;
        std::cout << "Using CPU backend" << std::endl;
    }
    
    // Load COCO class names
    classes_ = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };
}

cv::Mat YOLODetector::format_yolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

std::vector<Detection> YOLODetector::parse_detections(const cv::Mat& frame, 
                                                     const std::vector<cv::Mat>& outputs) {
    std::vector<Detection> detections;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    float x_factor = frame.cols / input_width_;
    float y_factor = frame.rows / input_height_;
    
    float* data = (float*)outputs[0].data;
    const int dimensions = 85;
    const int rows = 25200;
    
    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= conf_threshold_) {
            float* classes_scores = data + 5;
            cv::Mat scores(1, classes_.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            
            if (max_class_score > conf_threshold_) {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                
                int left = int((x - w / 2) * x_factor);
                int top = int((y - h / 2) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                
                boxes.emplace_back(left, top, width, height);
            }
        }
        data += dimensions;
    }
    
    // Apply NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);
    
    for (int idx : indices) {
        Detection det;
        det.class_id = class_ids[idx];
        det.confidence = confidences[idx];
        det.bbox = boxes[idx];
        detections.push_back(det);
    }
    
    return detections;
}

std::vector<Detection> YOLODetector::detect(cv::Mat& frame)
{
    auto input_image = format_yolov5(frame);
    
    // Create blob from image
    cv::Mat blob;
    cv::dnn::blobFromImage(input_image, blob, 1./255., 
                          cv::Size(input_width_, input_height_), 
                          cv::Scalar(), true, false);
    
    net_.setInput(blob);
    
    // Forward pass
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    
    return parse_detections(frame, outputs);
}

void YOLODetector::draw_detections(cv::Mat& frame, std::vector<Detection>& detections) {
    for (auto& detection : detections) {
        auto box = detection.bbox;
        auto classId = detection.class_id;
        auto confidence = detection.confidence;
        
        // Only draw vehicles
        if (classId == 2 || classId == 3 || classId == 5 || classId == 7) {
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
            
            std::string label = classes_[classId] + ": " + 
                               std::to_string(confidence).substr(0, 4);
            
            cv::putText(frame, label, cv::Point(box.x, box.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
    }
}