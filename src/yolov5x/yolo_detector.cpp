#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/cudaarithm.hpp>

#include "../../include/yolov5x/yolo_detector.hpp"
#include "../../include/device_type.hpp"

YOLODetector::YOLODetector
(
    const std::string& model_path,
    DeviceType device_type,
    float conf_threshold,
    float nms_threshold
)   : conf_threshold_(conf_threshold),
      nms_threshold_(nms_threshold),
      device_type_(device_type)
{
    load_classes_from_file("..\\models\\coco.names");

    try
    {
        net_ = cv::dnn::readNetFromONNX(model_path);
    } catch (const std::exception& e)
    {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        throw;
    }

    std::vector<std::pair<cv::dnn::Backend, cv::dnn::Target>> availableBackends = cv::dnn::getAvailableBackends();

    bool cudaBackendAvailable = false;
    for (const auto& pair : availableBackends)
    {
        if (pair.first == cv::dnn::DNN_BACKEND_CUDA && 
            pair.second == cv::dnn::DNN_TARGET_CUDA)
        {
            cudaBackendAvailable = true;
            break;
        }
    }

    // Set the preferred backend
    if (device_type == DeviceType::CUDA && cudaBackendAvailable)
    {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        std::cout << "Using CUDA backend" << std::endl;
    } else
    {
        if (device_type == DeviceType::CUDA)
        {
            std::cerr << "CUDA backend requested but not available. Falling back to CPU." << std::endl;
        }
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        device_type_ = DeviceType::CPU;
        std::cout << "Using CPU backend" << std::endl;
    }
}

void YOLODetector::load_classes_from_file(const std::string& path)
{
    std::ifstream ifs(path);

    if (!ifs.is_open())
    {
        std::cerr << "ERROR: Invalid input!" << std::endl;
        return;
    }

    std::string line;
    while (std::getline(ifs, line))
    {
        classes_.push_back(line);
    }
    std::cout << "Loaded " << classes_.size() << " classes from " << path << std::endl;
}

cv::Mat YOLODetector::format_yolov5(const cv::Mat& source) noexcept
{
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

std::vector<Detection> YOLODetector::parse_detections
(
    const cv::Mat& frame, 
    const std::vector<cv::Mat>& outputs
)
{
    std::vector<Detection> detections;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    // Resizing factor
    float x_factor = static_cast<float>(frame.cols) / input_width_;
    float y_factor = static_cast<float>(frame.rows) / input_height_;
    
    // Get output dimensions
    const auto& dimensions = outputs[0].size;
    if (dimensions.dims() < 3) {
        std::cerr << "Unexpected output dimensions: " << dimensions << std::endl;
        return detections;
    }
    
    const int rows = dimensions[1];
    const int dimensions_size = dimensions[2];
    
    // Get pointer to output data
    float* data = (float*)outputs[0].data;
    
    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= conf_threshold_) {
            // Get class scores
            float* classes_scores = data + 5;
            int class_id = std::max_element(classes_scores, classes_scores + classes_.size()) - classes_scores;
            float max_class_score = classes_scores[class_id];
            
            if (max_class_score > conf_threshold_) {
                confidences.push_back(confidence * max_class_score);
                class_ids.push_back(class_id);
                
                // Extract bounding box coordinates
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                
                // Convert to image coordinates
                int left = static_cast<int>((x - w / 2) * x_factor);
                int top = static_cast<int>((y - h / 2) * y_factor);
                int width = static_cast<int>(w * x_factor);
                int height = static_cast<int>(h * y_factor);
                
                // Ensure coordinates are within image bounds
                left = std::max(0, left);
                top = std::max(0, top);
                width = std::min(width, frame.cols - left);
                height = std::min(height, frame.rows - top);
                
                if (width > 0 && height > 0) {
                    boxes.emplace_back(left, top, width, height);
                }
            }
        }
        // Move to next row
        data += dimensions_size;
    }
    
    // Apply Non-Maximum Suppression (NMS)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);
    
    // Create Detection objects for the filtered results
    for (int idx : indices) {
        cv::Rect_<float> bbox(
            static_cast<float>(boxes[idx].x),
            static_cast<float>(boxes[idx].y),
            static_cast<float>(boxes[idx].width),
            static_cast<float>(boxes[idx].height)
        );
        detections.emplace_back(class_ids[idx], confidences[idx], bbox);
    }
    
    return detections;
}

std::vector<Detection> YOLODetector::detect(cv::Mat& frame)
{
    try {
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
    } catch (const std::exception& e) {
        std::cerr << "Detection error: " << e.what() << std::endl;
        if (device_type_ == DeviceType::CUDA) {
            cv::cuda::resetDevice();
        }
        return {};
    }
}

void YOLODetector::draw_detections
(
    cv::Mat& frame, 
    const std::vector<Detection>& detections
) noexcept
{
    for (const auto& detection : detections) {
        auto box = detection.getBbox();
        auto classId = detection.getClassId();
        auto confidence = detection.getConfidence();
        
        // Only draw vehicles (car, motorcycle, bus, truck)
        if (classId == 2 || classId == 3 || classId == 5 || classId == 7) {
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
            
            std::string label = classes_[classId] + ": " + 
                               std::to_string(confidence).substr(0, 4);
            
            cv::putText(frame, label, cv::Point(box.x, box.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
    }
}