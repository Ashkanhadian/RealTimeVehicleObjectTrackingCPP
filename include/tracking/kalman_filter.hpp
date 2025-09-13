#pragma once

#include <opencv2/opencv.hpp>
#include "detection.hpp"

class KalmanFilter
{
    public:
        KalmanFilter();
        explicit KalmanFilter(const Detection& initial_detection);
        
        void init(const Detection& initial_detection);
        void predict();
        void update(const Detection& detection);

        cv::Rect_<float> predicted_bbox() const;
        cv::Rect_<float> corrected_bbox() const;
        cv::Mat state() const;
        cv::Mat covariance() const;

        // Constant velocity model state: [x, y, w, h, vx, vy, vw, vh]
        static constexpr int state_size = 8;
        static constexpr int measurement_size = 4;

    private:
        cv::KalmanFilter kf_;
        cv::Mat measurement_;
        
        void setup_kalman_filter();
};