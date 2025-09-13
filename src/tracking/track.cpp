#include "tracking/track.hpp"
#include <iostream>

Track::Track(int id, const Detection& initial_detection, 
             std::chrono::steady_clock::time_point timestamp)
    : id_(id), hits_(1), age_(0), time_since_update_(0),
      last_detection_(initial_detection), last_update_time_(timestamp) {
    kf_.init(initial_detection);
}

void Track::predict() {
    kf_.predict();
    age_++;
}

void Track::update(const Detection& detection, 
                  std::chrono::steady_clock::time_point timestamp) {
    kf_.update(detection);
    last_detection_ = detection;
    hits_++;
    time_since_update_ = 0;
    last_update_time_ = timestamp;
}

void Track::mark_missed() {
    time_since_update_++;
}

cv::Rect_<float> Track::predicted_bbox() const {
    return kf_.predicted_bbox();
}

cv::Rect_<float> Track::current_bbox() const {
    return kf_.corrected_bbox();
}