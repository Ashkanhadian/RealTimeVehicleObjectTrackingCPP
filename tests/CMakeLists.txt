cmake_minimum_required(VERSION 3.25)
project(RealTimeVehicleObjectTrackingCPP_Tests)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GTEST_ROOT "C:/GoogleTest")
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

set(OpenCV_DIR "C:/OpenCV-4.12.0/x64/vc17/lib")
find_package(OpenCV REQUIRED COMPONENTS
    core
    highgui
    imgproc
    dnn
    video
)

# Detection tests
add_executable(DetectionTest 
    detection_test/detection_test.cpp 
    ../src/tracking/detection.cpp
    ../src/tracking/track.cpp
    ../src/tracking/kalman_filter.cpp
    )
target_include_directories(DetectionTest PRIVATE 
    ../include 
    ${OpenCV_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
)
target_link_libraries(DetectionTest PRIVATE 
    GTest::gtest
    GTest::gtest_main 
    ${OpenCV_LIBS}
)

add_executable(KalmanTest 
    kalman_test/kalman_test.cpp
    ../src/tracking/kalman_filter.cpp
    ../src/tracking/detection.cpp
)
target_include_directories(KalmanTest PRIVATE 
    ../include 
    ${OpenCV_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
)
target_compile_definitions(KalmanTest PRIVATE EXCLUDE_TRACK_DEPENDENT)
target_link_libraries(KalmanTest PRIVATE 
    GTest::gtest 
    GTest::gtest_main 
    ${OpenCV_LIBS}
)

add_executable(HungarianTest 
    hungarian_test/HungarianTest.cpp
    ../src/tracking/hungarian.cpp
)
target_include_directories(HungarianTest PRIVATE 
    ../include 
    ${OpenCV_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
)
target_link_libraries(HungarianTest PRIVATE 
    GTest::gtest 
    GTest::gtest_main 
    ${OpenCV_LIBS}
)

add_executable(TrackTest 
    track_test/TrackTest.cpp
    ../src/tracking/track.cpp
    ../src/tracking/tracker.cpp
    ../src/tracking/kalman_filter.cpp
    ../src/tracking/detection.cpp
    ../src/tracking/hungarian.cpp
)
target_include_directories(TrackTest PRIVATE 
    ../include 
    ${OpenCV_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
)
target_link_libraries(TrackTest PRIVATE 
    GTest::gtest
    GTest::gtest_main 
    ${OpenCV_LIBS}
)

# Add all tests to a custom target
add_custom_target(run_all_tests
    COMMAND DetectionTest
    COMMAND KalmanTest
    COMMAND HungarianTest
    COMMAND TrackTest
    DEPENDS DetectionTest TrackTest HungarianTest KalmanTest
    COMMENT "Running tests..."
)