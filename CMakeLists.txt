cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(RealTimeVehicleObjectTrackingCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==================================================
# Find OpenCV with CUDA support
# ==================================================
set(OpenCV_DIR "C:/OpenCV-4.12.0/x64/vc17/lib")
find_package(OpenCV REQUIRED COMPONENTS
    core
    highgui
    imgproc
    dnn
    cudaarithm
    cudabgsegm
    cudafilters
    cudaimgproc
    cudawarping
)

if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
    
    # Check if CUDA is available
    if(OpenCV_HAVE_CUDA)
        message(STATUS "CUDA support detected in OpenCV")
        add_definitions(-DHAVE_CUDA=1)
    else()
        message(WARNING "OpenCV was compiled without CUDA support")
        add_definitions(-DHAVE_CUDA=0)
    endif()
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

# ==================================================
# Target Configuration
# ==================================================
add_executable(RealTimeVehicleObjectTrackingCPP
    src/main.cpp
    src/yolov5x/yolo_detector.cpp
    src/tracking/hungarian.cpp
    src/tracking/kalman_filter.cpp
    src/tracking/track.cpp
    src/tracking/tracker.cpp
    src/tracking/detection.cpp
)

# ==================================================
# Include Directories
# ==================================================
target_include_directories(RealTimeVehicleObjectTrackingCPP PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
)

# ==================================================
# Link Libraries
# ==================================================
target_link_libraries(RealTimeVehicleObjectTrackingCPP PRIVATE
    ${OpenCV_LIBS}
)

# ==================================================
# Compiler Settings
# ==================================================
set_target_properties(RealTimeVehicleObjectTrackingCPP PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# ==================================================
# Output Configuration
# ==================================================
set_target_properties(RealTimeVehicleObjectTrackingCPP PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../lib"
)

# ==================================================
# Diagnostic Output
# ==================================================
message(STATUS "OpenCV Include Directory: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV CUDA Support: ${OpenCV_HAVE_CUDA}")