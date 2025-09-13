#pragma once
#include "kalman_filter.hpp"
#include "detection.hpp"
#include <chrono>

class Track {
public:
    Track(int id, const Detection& initial_detection, 
          std::chrono::steady_clock::time_point timestamp);
    
    void predict();
    void update(const Detection& detection, 
                std::chrono::steady_clock::time_point timestamp);
    void mark_missed();
    
    // Getters
    int id() const { return id_; }
    int hits() const { return hits_; }
    int age() const { return age_; }
    int time_since_update() const { return time_since_update_; }
    Detection last_detection() const { return last_detection_; }
    cv::Rect_<float> predicted_bbox() const;
    cv::Rect_<float> current_bbox() const;
    bool is_confirmed() const { return hits_ >= min_hits_; }
    bool is_dead() const { return time_since_update_ > max_age_; }
    
    // Static configuration
    static inline int min_hits_ = 3;
    static inline int max_age_ = 30;
    
private:
    int id_;
    int hits_;
    int age_;
    int time_since_update_;
    Detection last_detection_;
    KalmanFilter kf_;
    std::chrono::steady_clock::time_point last_update_time_;
};