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

    try
    {
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

        int targetWidth = 640;
        int targetHeight = static_cast<int>(frameHeight * (targetWidth / static_cast<float>(frameWidth)));

        std::cout << "Original resolution: " << frameWidth << "x" << frameHeight << std::endl;
        std::cout << "Reduced resolution: " << targetWidth << "x" << targetHeight << std::endl;
        std::cout << "Video FPS: " << fps << std::endl;

        cv::namedWindow("YOLOv5 Object Detection & Tracking", cv::WINDOW_NORMAL);
        cv::resizeWindow("YOLOv5 Object Detection & Tracking", targetWidth, targetHeight);

        auto startTime = std::chrono::high_resolution_clock::now();
        int frameCount = 0;
        int processedFrames = 0;

        cv::Mat frame, resizedFrame;
        while (true)
        {
            std::cout << "Reading frame " << processedFrames << "..." << std::endl;

            cap >> frame;
            if (frame.empty())
            {
                std::cout << "End of video stream" << std::endl;
                break;
            }

            cv::resize(frame, resizedFrame, cv::Size(targetWidth, targetHeight));

            auto detections = detector.detect(resizedFrame);

            // auto tracks = tracker.update(detections);

            
            detector.draw_detections(resizedFrame, detections);

            // for (const auto& track : tracks)
            // {
            //     auto bbox = track.current_bbox();
            //     int trackId = track.id();
                
            //     // Only draw vehicles (car, motorcycle, bus, truck)
            //     // if (track.id() == 2 || track.id() == 3 || 
            //     //     track.id() == 5 || track.id() == 7) {
            //     //     // Draw bounding box
            //     rectangle(resizedFrame, bbox, cv::Scalar(0, 255, 255), 2);
                
            //     // Draw track ID
            //     std::string label = "ID: " + std::to_string(trackId);
            //     putText(resizedFrame, label, cv::Point(bbox.x, bbox.y - 10),
            //             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
            //     // }
            // }

            // Calculate and display FPS
            frameCount++;
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto elapsedTime = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count();
            
            if (elapsedTime >= 1)
            {
                double currentFps = frameCount / static_cast<double>(elapsedTime);
                std::string fpsText = "FPS: " + std::to_string(currentFps).substr(0, 4);
                putText(resizedFrame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
                
                // Reset counters
                frameCount = 0;
                startTime = currentTime;
            }

            std::cout << "Displaying frame..." << std::endl;
            imshow("YOLOv5 Object Detection & Tracking", resizedFrame);

            if (cv::waitKey(1) == 27)
            { // ESC key
                break;
            }

            processedFrames++;
        }

        cap.release();
        cv::destroyAllWindows();
    } catch (const std::exception& e)
    {
        std::cerr << "Error in main: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "Program completed successfully" << std::endl;
    return 0;
}