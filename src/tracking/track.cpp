#include "tracking/track.hpp"
#include <iostream>

int Track::min_hits_ = 1;
int Track::max_age_ = 5;

Track::Track
(
    int id,
    const Detection& initial_detection,
    std::chrono::steady_clock::time_point timestamp
) noexcept
    : id_(id),
      hits_(1),
      age_(0),
      time_since_update_(0),
      last_detection_(initial_detection),
      last_update_time_(timestamp)
{
    kf_.init(initial_detection);
}

void Track::predict() noexcept
{
    kf_.predict();
    age_++;
}

void Track::update
(
    const Detection& detection,
    std::chrono::steady_clock::time_point timestamp
) noexcept
{
    kf_.update(detection);
    last_detection_ = detection;
    hits_++;
    time_since_update_ = 0;
    last_update_time_ = timestamp;
}

void Track::mark_missed() noexcept
{
    time_since_update_++;
}

[[nodiscard]] cv::Rect_<float> Track::predicted_bbox() const noexcept
{
    return kf_.predicted_bbox();
}

[[nodiscard]] cv::Rect_<float> Track::current_bbox() const noexcept
{
    return kf_.corrected_bbox();
}

[[nodiscard]] bool Track::is_confirmed() const noexcept
{
    return hits_ >= min_hits_;
}

[[nodiscard]] bool Track::is_dead() const noexcept
{
    return time_since_update_ > max_age_;
}