#pragma once

#include "kalman_filter.hpp"
#include "detection.hpp"
#include <chrono>

class Track
{
    private:
        int id_;
        int hits_;
        int age_;
        int time_since_update_;
        Detection last_detection_;
        KalmanFilter kf_;
        std::chrono::steady_clock::time_point last_update_time_;

    public:
        Track(int id,
              const Detection& initial_detection,
              std::chrono::steady_clock::time_point timestamp) noexcept;

        void predict() noexcept;
        void update(const Detection& detection,
                    std::chrono::steady_clock::time_point timestamp) noexcept;
        void mark_missed() noexcept;

        [[nodiscard]] int id() const noexcept { return id_; }
        [[nodiscard]] int hits() const noexcept { return hits_; }
        [[nodiscard]] int age() const noexcept { return age_; }
        [[nodiscard]] int time_since_update() const noexcept { return time_since_update_; }
        [[nodiscard]] Detection last_detection() const noexcept {return last_detection_; }
        [[nodiscard]] cv::Rect_<float> predicted_bbox() const noexcept;
        [[nodiscard]] cv::Rect_<float> current_bbox() const noexcept;
        [[nodiscard]] bool is_confirmed() const noexcept;
        [[nodiscard]] bool is_dead() const noexcept;

        static int min_hits_;
        static int max_age_;
};