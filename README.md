# RealTimeVehicleObjectTrackingCPP

C++ implementation of object detection using YOLO with Kalman filtering and Hungarian algorithm for multi-object tracking.

# Features

- YOLOv5 Object Detection: Leverages ONNX model for fast and accurate vehicle detection

- Multi-Object Tracking: Implements Kalman filters for motion prediction and state estimation

- Data Association: Uses Hungarian algorithm for optimal detection-to-track assignment

- Real-time Performance: Optimized for both CPU and CUDA acceleration

- Vehicle Counting: Maintains persistent IDs and counts total vehicles in the video

- Cross-Platform: Built with CMake and OpenCV for portability

# System Architecture

Video Input → YOLOv5 Detection → Detection Scaling → Hungarian Matching → Kalman Filter Tracking → Output Visualization

# Components

- YOLODetector: Handles object detection using YOLOv5 ONNX model

- Tracker: Manages track lifecycle and data association

- KalmanFilter: Predicts and corrects object positions

- HungarianAlgorithm: Solves optimal assignment problem

- Detection & Track: Data structures for detection and track representation

# Data used in this project

The source code is available for learning and development purposes. The sample video data is owned by Vecteezy and is used under their Standard License. Please ensure you have the right to use any video you process with this code.

The video file can be downloaded from https://www.vecteezy.com/video/14055564-out-of-focus-street-traffic-high-angle-view.

# Supported Vehicle Classes

* Cars (class 2)

* Motorcycles (class 3)

* Buses (class 5)

* Trucks (class 7)

# Installation & Usage

**Prerequisites**

* OpenCV 4.12.0+ with CUDA support (optional)

* CMake 3.25+

* C++20 compatible compiler

**Building the Project**

```bash
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -Wno-dev
cmake --build . --config Release
```

**Running the Application**

```bash
./bin/Release/RealTimeVehicleObjectTrackingCPP.exe ./data/vecteezy_out-of-focus-street-traffic-high-angle-view.mov ./models/yolov5x.onnx cuda
```

Parameters:

- video_path: Path to input video file (default: ./data/vecteezy_out-of-focus-street-traffic-high-angle-view.mov)

- model_path: Path to YOLOv5 ONNX model (default: ./models/yolov5x.onnx)

- device_type: Processing device - "cpu" or "cuda" (default: cpu)