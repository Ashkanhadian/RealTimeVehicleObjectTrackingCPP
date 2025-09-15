#pragma once

#include <opencv2/opencv.hpp>
#include <string>

enum class DeviceType
{
    CPU,
    CUDA
};

namespace device
{
    inline std::string to_string(DeviceType type)
    {
        switch (type)
        {
            case DeviceType::CPU: return "CPU";
            case DeviceType::CUDA: return "CUDA";
            default: return "Unknown";
        }
    }

    inline bool is_cuda_available()
    {
        #ifdef HAVE_CUDA
        return cv::cuda::getCudaEnabledDeviceCount() > 0;
        #else
        return false;
        #endif
    }

    inline DeviceType auto_select()
    {
        return is_cuda_available() ? DeviceType::CUDA : DeviceType::CPU;
    }
} // end namespace device