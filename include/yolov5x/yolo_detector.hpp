#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include "../tracking/detection.hpp"
#include "../device_type.hpp"

class YOLODetector
{
    private:
        cv::dnn::Net net_;
        std::vector<std::string> classes_;
        float conf_threshold_;
        float nms_threshold_;
        DeviceType device_type_;
        
        const int input_width_ = 640;
        const int input_height_ = 640;
        
        [[nodiscard]] cv::Mat format_yolov5(const cv::Mat& source) noexcept;
        [[nodiscard]] std::vector<Detection> parse_detections
        (
            const cv::Mat& frame, 
            const std::vector<cv::Mat>& outputs
        );
        void load_classes_from_file(const std::string& path);

    public:
        YOLODetector(const std::string& model_path, 
                    DeviceType device_type = DeviceType::CPU,
                    float conf_threshold = 0.5f,
                    float nms_threshold = 0.4f);
        
        [[nodiscard]] std::vector<Detection> detect(cv::Mat& frame);
        void draw_detections(cv::Mat& frame, const std::vector<Detection>& detections) noexcept;
        void resetDevice() noexcept
        {
            if (device_type_ == DeviceType::CUDA)
                cv::cuda::resetDevice();
        }

        [[nodiscard]] DeviceType device_type() const noexcept { return device_type_; }
        [[nodiscard]] float getConfidenceThreshold() const noexcept { return conf_threshold_; }
        [[nodiscard]] float getNMSThreshold() const noexcept { return nms_threshold_; }
};