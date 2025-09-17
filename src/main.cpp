// #include <opencv2/opencv.hpp>
// #include <opencv2/cudaimgproc.hpp>
// #include <iostream>
// #include <chrono>
// #include "../include/yolov5x/yolo_detector.hpp"
// #include "../include/tracking/tracker.hpp"
// #include "../include/device_type.hpp"

// int main(int argc, char* argv[]) {
//     // Parse command line arguments
//     DeviceType device_type = DeviceType::CPU;
//     std::string video_path = "data/traffic.mp4";
//     float conf_threshold = 0.5f;
//     float iou_threshold = 0.3f;
    
//     for (int i = 1; i < argc; ++i) {
//         std::string arg = argv[i];
//         if (arg == "--cuda" && device::is_cuda_available()) {
//             device_type = DeviceType::CUDA;
//         } else if (arg == "--cpu") {
//             device_type = DeviceType::CPU;
//         } else if (arg == "--video" && i + 1 < argc) {
//             video_path = argv[++i];
//         } else if (arg == "--conf" && i + 1 < argc) {
//             conf_threshold = std::stof(argv[++i]);
//         } else if (arg == "--iou" && i + 1 < argc) {
//             iou_threshold = std::stof(argv[++i]);
//         } else if (arg == "--help") {
//             std::cout << "Usage: " << argv[0] << " [--cuda] [--cpu] [--video <path>] [--conf <threshold>] [--iou <threshold>] [--help]" << std::endl;
//             return 0;
//         }
//     }
    
//     try {
//         // Initialize YOLO detector
//         std::cout << "Initializing YOLO detector..." << std::endl;
//         YOLODetector detector("C:\\Users\\admin\\Desktop\\Projects\\RealTimeVehicleObjectTrackingCPP\\models\\yolov5s.onnx", 
//                              device_type, conf_threshold);
        
//         // Initialize tracker
//         std::cout << "Initializing tracker..." << std::endl;
//         Tracker tracker(iou_threshold);
        
//         // Open video capture
//         std::cout << "Opening video: " << video_path << std::endl;
//         cv::VideoCapture cap(video_path);
//         if (!cap.isOpened()) 
//         {
//             std::cerr << "Error opening video stream: " << video_path << std::endl;
//             return -1;
//         }
        
//         // Get video properties
//         int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
//         int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
//         double fps = cap.get(cv::CAP_PROP_FPS);
        
//         std::cout << "Video resolution: " << frame_width << "x" << frame_height << std::endl;
//         std::cout << "Video FPS: " << fps << std::endl;
//         std::cout << "Using device: " << device::to_string(detector.device_type()) << std::endl;
//         std::cout << "Confidence threshold: " << conf_threshold << std::endl;
//         std::cout << "IOU threshold: " << iou_threshold << std::endl;
        
//         cv::Mat frame;
//         auto last_time = std::chrono::high_resolution_clock::now();
//         int frame_count = 0;
        
//         std::cout << "Starting video processing..." << std::endl;
        
//         while (true) {
//             try {
//                 cap >> frame;
//                 if (frame.empty())
//                 {
//                     std::cout << "End of video stream" << std::endl;
//                     break;
//                 }

//                 std::cout << "Processing frame " << frame_count << std::endl;
                
//                 // Detect objects
//                 auto detections = detector.detect(frame);
//                 std::cout << "Detected " << detections.size() << " objects" << std::endl;
                
//                 // Update tracker
//                 auto tracks = tracker.update(detections);
//                 std::cout << "Tracking " << tracks.size() << " objects" << std::endl;
                
//                 // Draw tracks
//                 for (const auto& track : tracks) {
//                     auto box = track.current_bbox();
//                     cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
                    
//                     std::string label = "ID: " + std::to_string(track.id());
//                     cv::putText(frame, label, cv::Point(box.x, box.y - 5),
//                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
//                 }
                
//                 // Calculate and display FPS
//                 frame_count++;
//                 auto current_time = std::chrono::high_resolution_clock::now();
//                 auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_time).count();
                
//                 if (elapsed >= 1) {
//                     double current_fps = frame_count / static_cast<double>(elapsed);
//                     std::string fps_text = "FPS: " + std::to_string(current_fps).substr(0, 4);
//                     cv::putText(frame, fps_text, cv::Point(10, 30), 
//                                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
                    
//                     // Reset counters
//                     frame_count = 0;
//                     last_time = current_time;
//                 }
                
//                 // Display device info
//                 std::string device_text = "Device: " + device::to_string(detector.device_type());
//                 cv::putText(frame, device_text, cv::Point(10, 70), 
//                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                
//                 // Display track count
//                 std::string track_text = "Tracks: " + std::to_string(tracks.size());
//                 cv::putText(frame, track_text, cv::Point(10, 110), 
//                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                
//                 // Display result
//                 cv::imshow("YOLOv5 Vehicle Tracking", frame);
                
//                 // Reset CUDA device periodically to prevent memory issues
//                 if (device_type == DeviceType::CUDA && frame_count % 10 == 0) {
//                     cv::cuda::resetDevice();
//                     std::cout << "CUDA device reset" << std::endl;
//                 }
                
//                 // Handle keyboard input
//                 int key = cv::waitKey(1);
//                 if (key == 27) break; // ESC to exit
//                 else if (key == 'p') cv::waitKey(0); // Pause
                
//             } catch (const std::exception& e) {
//                 std::cerr << "Error processing frame: " << e.what() << std::endl;
//                 // Reset CUDA device on error
//                 if (device_type == DeviceType::CUDA) {
//                     cv::cuda::resetDevice();
//                     std::cout << "CUDA device reset after error" << std::endl;
//                 }
//             }
//         }
//     } catch (const std::exception& e) {
//         std::cerr << "Initialization error: " << e.what() << std::endl;
//         return -1;
//     }
    
//     std::cout << "Program completed successfully" << std::endl;
//     return 0;
// }

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <memory>

#include "../include/yolov5x/yolo_detector.hpp"
#include "../include/tracking/tracker.hpp"
#include "../include/device_type.hpp"

int main(int argc, char** argv)
{
    // Parse command line arguments
    cv::String modelPath = "../models/yolov5x.onnx"; // Default model path
    cv::String videoPath = "../data/vecteezy_out-of-focus-street-traffic-high-angle-view.mov";
    DeviceType deviceType = DeviceType::CPU; // Default to CPU
    float confThreshold = 0.5f;
    float nmsThreshold = 0.4f;
    float iouThreshold = 0.3f;

    if (argc > 1)
    {
        videoPath = argv[1];
    }
    if (argc > 2)
    {
        modelPath = argv[2];
    }
    if (argc > 3)
    {
        std::string deviceArg = argv[3];
        if (deviceArg == "cuda" || deviceArg == "CUDA")
        {
            deviceType = DeviceType::CUDA;
        }
    }

    // Check if CUDA is available if requested
    if (deviceType == DeviceType::CUDA && !device::is_cuda_available())
    {
        std::cout << "CUDA requested but not available. Falling back to CPU." << std::endl;
        deviceType = DeviceType::CPU;
    }

    YOLODetector detector(modelPath, deviceType, confThreshold, nmsThreshold);

    Tracker tracker(iouThreshold);

    cv::VideoCapture cap;
    cap.open(videoPath);

    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open video source: " << videoPath << std::endl;
        return -1;
    }

    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::namedWindow("YOLOv5 Object Detection & Tracking", cv::WINDOW_NORMAL);
    cv::resizeWindow("YOLOv5 Object Detection & Tracking", frameWidth, frameHeight);

    auto startTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;

    cv::Mat frame;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            break;
        }

        auto detections = detector.detect(frame);

        auto tracks = tracker.update(detections);
        
        detector.draw_detections(frame, detections);

        for (const auto& track : tracks)
        {
            auto bbox = track.current_bbox();
            int trackId = track.id();
            
            // Only draw vehicles (car, motorcycle, bus, truck)
            if (track.id() == 2 || track.id() == 3 || 
                track.id() == 5 || track.id() == 7) {
                // Draw bounding box
                rectangle(frame, bbox, cv::Scalar(0, 255, 255), 2);
                
                // Draw track ID
                std::string label = "ID: " + std::to_string(trackId);
                putText(frame, label, cv::Point(bbox.x, bbox.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
            }
        }

        // Calculate and display FPS
        frameCount++;
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count();
        
        if (elapsedTime >= 1)
        {
            double currentFps = frameCount / static_cast<double>(elapsedTime);
            std::string fpsText = "FPS: " + std::to_string(currentFps).substr(0, 4);
            putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            
            // Reset counters
            frameCount = 0;
            startTime = currentTime;
        }

        imshow("YOLOv5 Object Detection & Tracking", frame);

        if (cv::waitKey(1) == 27) { // ESC key
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
}