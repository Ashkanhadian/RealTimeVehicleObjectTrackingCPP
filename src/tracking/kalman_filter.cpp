#include "../../include/tracking/kalman_filter.hpp"
#include <iostream>

KalmanFilter::KalmanFilter()
{
    kf_ = cv::KalmanFilter(state_size, measurement_size, 0);
    measurement_ = cv::Mat::zeros(measurement_size, 1, CV_32F);
    setup_kalman_filter();
}

KalmanFilter::KalmanFilter(const Detection& initial_detection)
    : KalmanFilter()
{
    init(initial_detection);
}

void KalmanFilter::init(const Detection& initial_detection)
{
    // Initialize state with detection
    auto initial_detection_bbox = initial_detection.getBbox();
    kf_.statePost.at<float>(0) = initial_detection_bbox.x;
    kf_.statePost.at<float>(1) = initial_detection_bbox.y;
    kf_.statePost.at<float>(2) = initial_detection_bbox.width;
    kf_.statePost.at<float>(3) = initial_detection_bbox.height;
    // Initialize velocities to 0
    kf_.statePost.at<float>(4) = 0;
    kf_.statePost.at<float>(5) = 0;
    kf_.statePost.at<float>(6) = 0;
    kf_.statePost.at<float>(7) = 0;
}

void KalmanFilter::setup_kalman_filter()
{
    // Transition matrix (A)
    kf_.transitionMatrix = cv::Mat::eye(state_size, state_size, CV_32F);
    kf_.transitionMatrix.at<float>(0, 4) = 1;
    kf_.transitionMatrix.at<float>(1, 5) = 1;
    kf_.transitionMatrix.at<float>(2, 6) = 1;
    kf_.transitionMatrix.at<float>(3, 7) = 1;

    // Measurement matrix (H)
    kf_.measurementMatrix = cv::Mat::zeros(measurement_size, state_size, CV_32F);
    kf_.measurementMatrix.at<float>(0, 0) = 1;
    kf_.measurementMatrix.at<float>(1, 1) = 1;
    kf_.measurementMatrix.at<float>(2, 2) = 1;
    kf_.measurementMatrix.at<float>(3, 3) = 1;

    // Process noise covariance (Q)
    kf_.processNoiseCov = cv::Mat::eye(state_size, state_size, CV_32F);
    kf_.processNoiseCov.at<float>(4, 4) = 0.01f;    // vx
    kf_.processNoiseCov.at<float>(5, 5) = 0.01f;    // vy
    kf_.processNoiseCov.at<float>(6, 6) = 0.0001f;   // vw
    kf_.processNoiseCov.at<float>(7, 7) = 0.0001f;   // vh
    kf_.processNoiseCov *= 1e-2;

    // Measurement noise covariance (R)
    kf_.measurementNoiseCov = cv::Mat::eye(measurement_size, measurement_size, CV_32F);
    kf_.measurementNoiseCov.at<float>(0, 0) = 0.1f;
    kf_.measurementNoiseCov.at<float>(1, 1) = 0.1f;
    kf_.measurementNoiseCov.at<float>(2, 2) = 0.05f;
    kf_.measurementNoiseCov.at<float>(3, 3) = 0.05f;
    kf_.measurementNoiseCov *= 1e-1;

    // State covariance (P)
    kf_.errorCovPost = cv::Mat::eye(state_size, state_size, CV_32F);
}

void KalmanFilter::predict()
{
    kf_.predict();
}

void KalmanFilter::update(const Detection& detection)
{
    auto detection_bbox = detection.getBbox();
    measurement_.at<float>(0) = detection_bbox.x;
    measurement_.at<float>(1) = detection_bbox.y;
    measurement_.at<float>(2) = detection_bbox.width;
    measurement_.at<float>(3) = detection_bbox.height;
    
    kf_.correct(measurement_);
}

[[nodiscard]] cv::Rect_<float> KalmanFilter::predicted_bbox() const
{
    const cv::Mat& state = kf_.statePre;
    return cv::Rect_<float>(
        state.at<float>(0),
        state.at<float>(1),
        state.at<float>(2),
        state.at<float>(3)
    );
}

[[nodiscard]] cv::Rect_<float> KalmanFilter::corrected_bbox() const
{
    const cv::Mat& state = kf_.statePost;
    return cv::Rect_<float>(
        state.at<float>(0),
        state.at<float>(1),
        state.at<float>(2),
        state.at<float>(3)
    );
}

[[nodiscard]] cv::Mat KalmanFilter::state() const
{
    return kf_.statePost;
}

[[nodiscard]] cv::Mat KalmanFilter::covariance() const
{
    return kf_.errorCovPost;
}