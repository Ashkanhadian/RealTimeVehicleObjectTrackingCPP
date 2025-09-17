#pragma once

#include "detection.hpp"
#include <opencv2/opencv.hpp>

class KalmanFilter
{
    private:
        cv::KalmanFilter kf_;
        cv::Mat measurement_;

        void setup_kalman_filter();

    public:
        KalmanFilter();
        explicit KalmanFilter(const Detection& initial_detection);

        void init(const Detection& initial_detection);
        void predict();
        void update(const Detection& detection);

        [[nodiscard]] cv::Rect_<float> predicted_bbox() const;
        [[nodiscard]] cv::Rect_<float> corrected_bbox() const;
        [[nodiscard]] cv::Mat state() const;
        [[nodiscard]] cv::Mat covariance() const;

        // Constant velocity model state: [x, y, w, h, vx, vy, vw, vh]
        static constexpr int state_size = 8;
        static constexpr int measurement_size = 4;
};