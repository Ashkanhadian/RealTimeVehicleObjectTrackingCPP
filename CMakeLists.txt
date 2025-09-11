cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(RealTimeVehicleObjectTrackingCPP LANGUAGES CXX)

# ==================================================
# Find OpenCV with CUDA support
# ==================================================
set(OpenCV_DIR "C:/OpenCV/x64/vc17/lib")
find_package(OpenCV REQUIRED COMPONENTS
    core
    highgui
    imgproc
    cudaarithm
    cudabgsegm
    cudafilters
    cudaimgproc
    cudawarping
)

# Verify OpenCV was built with CUDA support
if(NOT OpenCV_HAVE_CUDA)
    message(WARNING "OpenCV was not built with CUDA support.")
endif()

# ==================================================
# Target Configuration
# ==================================================
add_executable(RealTimeVehicleObjectTrackingCPP
    src/main.cpp
    src/hungarian.cpp
    src/kalman_filter.cpp
    src/nn.cpp
    src/utils.cpp
    include/hungarian.hpp
    include/kalman_filter.hpp
    include/nn.hpp
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