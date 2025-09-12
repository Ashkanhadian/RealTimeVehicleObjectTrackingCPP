#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

#include <include/device_type.hpp>
#include <include/tracking/detection.hpp>

class YOLODetector
{
    public:
        YOLODetector(const std::string& model_path,
                     DeviceType device_type = DeviceType::CUDA,
                     const std::string& classes_file = "",
                     float conf_threshold = 0.5,
                     float nms_threshold = 0.4,
                     float score_threshold = 0.5);
        
        std::vector<Detection> detect(cv::Mat& frame);
        void draw_detections(cv::Mat& frame, std::vector<Detection>& detections);
        std::vector<std::string> load_class_list();
        DeviceType get_device_type() const {return device_type_; }

    private:
        cv::dnn::Net net_;
        std::vector<std::string> classes_;
        float conf_threshold_;
        float nms_threshold_;
        float score_threshold_;
        DeviceType device_type_;

        const float input_width_ = 640.0;
        const float input_height_ = 640.0;

        cv::Mat format_yolov5(const cv::Mat& source);
};