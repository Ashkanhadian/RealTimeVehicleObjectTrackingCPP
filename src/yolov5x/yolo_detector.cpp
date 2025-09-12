#include <include/yolo_detector.hpp>
#include <opencv2/core/cuda.hpp>
#include <fstream>
#include <sstream>
#include <iostream>

YOLODetector::YOLODetector
(
    const std::string& model_path,
    DeviceType device_type,
    const std::string& classes_file,
    float conf_threshold,
    float nms_threshold,
    float score_threshold
) : conf_threshold_(conf_threshold),
    nms_threshold_(nms_threshold),
    score_threshold_(score_threshold),
    device_type_(device_type)
{
    net_ = cv::dnn::readNetFromONNX(model_path);

    if (device_type == DeviceType::CUDA && is_cuda_available())
    {
        std::cout << "Using CUDA" << std::endl;
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else
    {
        if (device_type == DeviceType::CUDA)
        {
            std::cout << "CUDA requested but not available. Falling back to CPU." << std::endl;
        }
        std::cout << "Using CPU" << std::endl;
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        device_type_ = DeviceType::CPU;
    }

    if (!classes_file.empty())
    {
        classes_ = load_class_list();
    } else
    {
        classes_ = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"};
    }

    std::cout << "YOLO detector initialized with " << classes_.size() << "classes on " << device_type_to_string(device_type_) << std::endl;
}

std::vector<std::string> YOLODetector::load_class_list()
{
    std::vector<std::string> class_list;
    std::ifstream ifs("models/coco.names"); // Update path if needed
    std::string line;
    while (std::getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}

cv::Mat YOLODetector::format_yolov5(const cv::Mat& sources)
{
    int col = sources.cols;
    int row = sources.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    sources.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

std::vector<Detection> YOLODetector::detect(cv::Mat& frame)
{
    std::vector<Detection> output;

    auto start = std::chrono::high_resolution_clock::now();

    auto input_image = format_yolov5(frame);

    cv::Mat blob;
    cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(input_width_, input_height_), cv::Scalar(), true, false);
    net_.setInput(blob);

    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    float x_factor = input_image.cols / input_width_;
    float y_factor = input_image.rows / input_height_;

    float* data = (float*)outputs[0].data;

    const int dimensions = 85; // x, y, w, h, confidence + 80 classes
    const int rows = 25200; // YOLOv5 output size

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // Process all detections
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];

        if (confidence >= conf_threshold_)
        {
            float* classes_scores = data + 5;
            cv::Mat scores(1, classes_.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            if (max_class_score > score_threshold_)
            {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                // Calculate bounding box coordinates
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }

        data += dimensions;
    }

    // Apply Non-Maximum Suppression
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, nms_result);

    // Store filtered detections
    for (size_t i = 0; i < nms_result.size(); i++)
    {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }

    std::cout << "Inference time: " << duration.count() << " ms, Detections: " << output.size() << std::endl;

    return output;
}

void YOLODetector::draw_detections(cv::Mat& frame, std::vector<Detection>& detections)
{
    // Colors for different vehicle types
    std::map<int, cv::Scalar> colors =
    {
        {2, cv::Scalar(0, 255, 0)},     // Car - Green
        {3, cv::Scalar(255, 0, 0)},     // Motorcycle - Blue
        {5, cv::Scalar(0, 0, 255)},     // Bus - Red
        {7, cv::Scalar(0, 255, 255)}    // Truck - Yellow
    };

    for (auto& detection : detections)
    {
        auto box = detection.box;
        auto classId = detection.class_id;
        auto confidence = detection.confidence;

        // Only draw vehicles
        if (classId == 2 || classId == 3 || classId == 5 || classId == 7)
        {
            cv::Scalar color = colors.count(classId) ? colors[classId] : cv::Scalar(255, 255, 255);

            cv::rectangle(frame, box, color, 2);

            std::string label = classes_[classId] + ": " + std::to_string(confidence).substr(0, 4);

            // Draw background for text
            cv::rectangle(frame,
                          cv::Point(box.x, box.y - 20),
                          cv::Point(box.x + label.size() * 0, box.y), color, -1);

            // Draw text
            cv::putText(frame, label, cv::Point(box.x, box.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
    }
}