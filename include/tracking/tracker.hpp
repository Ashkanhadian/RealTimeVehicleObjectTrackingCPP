#pragma once

#include "track.hpp"
#include "hungarian.hpp"
#include "detection.hpp"

#include <vector>
#include <memory>
#include <chrono>

class Tracker
{
    private:
        std::vector<std::shared_ptr<Track>> tracks_;
        int next_id_;
        float iou_threshold_;
        HungarianAlgorithm hungarian_;
        std::chrono::steady_clock::time_point last_update_time_;

        [[nodiscard]] std::vector<std::vector<float>> create_cost_matrix
        (
            const std::vector<Detection>& detection,
            const std::vector<std::shared_ptr<Track>>& tracks
        ) const noexcept;
        
        [[nodiscard]] float calculate_iou
        (
            const Detection& det1,
            const Detection& det2
        ) const noexcept;

        void associate_detections_to_tracks
        (
            const std::vector<Detection>& detections,
            const std::vector<std::vector<float>>& cost_matrix,
            std::vector<int>& assignment,
            std::vector<int>& unmatched_detections,
            std::vector<int>& unmatched_tracks
        ) noexcept;

    public:
        explicit Tracker(float iou_threshold = 0.3f) noexcept;

        [[nodiscard]] std::vector<Track> update(const std::vector<Detection>& detections) noexcept;

        [[nodiscard]] const std::vector<std::shared_ptr<Track>>& tracks() const noexcept
        {
            return tracks_;
        }
        
        [[nodiscard]] int next_id() const noexcept { return next_id_; }
};