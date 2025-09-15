#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include "../tracking/detection.hpp"
#include "../device_type.hpp"

class YOLODetector {
public:
    YOLODetector(const std::string& model_path, 
                 DeviceType device_type = DeviceType::CPU,
                 float conf_threshold = 0.5f,
                 float nms_threshold = 0.4f);
    
    std::vector<Detection> detect(cv::Mat& frame);
    void draw_detections(cv::Mat& frame, std::vector<Detection>& detections);
    void resetDevice()
    {
        if (device_type_ == DeviceType::CUDA)
            cv::cuda::resetDevice();
    }
    DeviceType device_type() const { return device_type_; }
    
private:
    cv::dnn::Net net_;
    std::vector<std::string> classes_;
    float conf_threshold_;
    float nms_threshold_;
    DeviceType device_type_;
    
    const float input_width_ = 640.0f;
    const float input_height_ = 640.0f;
    
    cv::Mat format_yolov5(const cv::Mat& source);
    std::vector<Detection> parse_detections(const cv::Mat& frame, 
                                           const std::vector<cv::Mat>& outputs);
};