#pragma once
#include "track.hpp"
#include "hungarian.hpp"
#include "detection.hpp"
#include <vector>
#include <memory>
#include <chrono>

class Tracker {
public:
    Tracker(float iou_threshold = 0.3f);
    
    std::vector<Track> update(const std::vector<Detection>& detections);
    
    // Getters
    const std::vector<std::shared_ptr<Track>>& tracks() const { return tracks_; }
    int next_id() const { return next_id_; }
    
private:
    std::vector<std::shared_ptr<Track>> tracks_;
    int next_id_;
    float iou_threshold_;
    HungarianAlgorithm hungarian_;
    std::chrono::steady_clock::time_point last_update_time_;
    
    std::vector<std::vector<float>> create_cost_matrix(
        const std::vector<Detection>& detections,
        const std::vector<std::shared_ptr<Track>>& tracks);
    
    void associate_detections_to_tracks(
        const std::vector<Detection>& detections,
        const std::vector<std::vector<float>>& cost_matrix,
        std::vector<int>& assignment,
        std::vector<int>& unmatched_detections,
        std::vector<int>& unmatched_tracks);
};