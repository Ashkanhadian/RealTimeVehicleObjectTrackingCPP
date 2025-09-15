#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>
#include <chrono>
#include "../include/yolov5x/yolo_detector.hpp"
#include "../include/tracking/tracker.hpp"
#include "../include/device_type.hpp"

int main(int argc, char* argv[]) {
    // Parse command line arguments
    DeviceType device_type = DeviceType::CPU;
    std::string video_path = "data/traffic.mp4";
    float conf_threshold = 0.5f;
    float iou_threshold = 0.3f;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cuda" && device::is_cuda_available()) {
            device_type = DeviceType::CUDA;
        } else if (arg == "--cpu") {
            device_type = DeviceType::CPU;
        } else if (arg == "--video" && i + 1 < argc) {
            video_path = argv[++i];
        } else if (arg == "--conf" && i + 1 < argc) {
            conf_threshold = std::stof(argv[++i]);
        } else if (arg == "--iou" && i + 1 < argc) {
            iou_threshold = std::stof(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [--cuda] [--cpu] [--video <path>] [--conf <threshold>] [--iou <threshold>] [--help]" << std::endl;
            return 0;
        }
    }
    
    // Initialize YOLO detector
    YOLODetector detector("C:\\Users\\admin\\Desktop\\Projects\\RealTimeVehicleObjectTrackingCPP\\models\\yolov5s.onnx", device_type, conf_threshold);
    
    // Initialize tracker
    Tracker tracker(iou_threshold);
    
    // Open video capture
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) 
    {
        // cap.open(0); // Try webcam
        // if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or camera" << std::endl;
        return -1;
        // }
    }
    
    // Get video properties
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    std::cout << "Video resolution: " << frame_width << "x" << frame_height << std::endl;
    std::cout << "Video FPS: " << fps << std::endl;
    std::cout << "Using device: " << device::to_string(detector.device_type()) << std::endl;
    std::cout << "Confidence threshold: " << conf_threshold << std::endl;
    std::cout << "IOU threshold: " << iou_threshold << std::endl;
    
    cv::Mat frame;
    auto last_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    
    while (true) {
        cap >> frame;
        if (frame.empty())
        {
            std::cout << "Frame is empty" << std::endl;
            break;
        }

        std::cout << "Processing frame" << std::endl;
        
        // Detect objects
        auto detections = detector.detect(frame);
        
        // Update tracker
        auto tracks = tracker.update(detections);
        
        // Draw tracks
        for (const auto& track : tracks) {
            auto box = track.current_bbox();
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
            
            std::string label = "ID: " + std::to_string(track.id());
            cv::putText(frame, label, cv::Point(box.x, box.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
        
        // Calculate and display FPS
        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_time).count();
        
        if (elapsed >= 1) {
            double fps = frame_count / static_cast<double>(elapsed);
            std::string fps_text = "FPS: " + std::to_string(fps).substr(0, 4);
            cv::putText(frame, fps_text, cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            
            // Reset counters
            frame_count = 0;
            last_time = current_time;
        }
        
        // Display device info
        std::string device_text = "Device: " + device::to_string(detector.device_type());
        cv::putText(frame, device_text, cv::Point(10, 70), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        // Display track count
        std::string track_text = "Tracks: " + std::to_string(tracks.size());
        cv::putText(frame, track_text, cv::Point(10, 110), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        // Display result
        cv::imshow("YOLOv5 Vehicle Tracking", frame);
        
        // Handle keyboard input
        int key = cv::waitKey(1);
        if (key == 27) break; // ESC to exit
        else if (key == 'p') cv::waitKey(0); // Pause
    }
    
    return 0;
}