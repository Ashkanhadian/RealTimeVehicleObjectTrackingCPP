#include "../../include/tracking/tracker.hpp"

#include <algorithm>
#include <iostream>

Tracker::Tracker(float iou_threshold) noexcept
    : next_id_(0),
      iou_threshold_(iou_threshold),
      last_update_time_(std::chrono::steady_clock::now()) {}

std::vector<Track> Tracker::update(const std::vector<Detection>& detections) noexcept
{
    auto current_time = std::chrono::steady_clock::now();

    // Predict current tracks
    for (auto& track : tracks_)
    {
        track->predict();
    }

    auto cost_matrix = create_cost_matrix(detections, tracks_);

    // Solve assignment problem
    std::vector<int> assignment;
    std::vector<int> unmatched_detections;
    std::vector<int> unmatched_tracks;

    if (!cost_matrix.empty())
    {
        // Convert cost matrix to cv::Mat
        cv::Mat cost_mat(static_cast<int>(cost_matrix.size()),
                         static_cast<int>(cost_matrix[0].size()), CV_32F);

        for (size_t i = 0; i < cost_matrix.size(); ++i)
        {
            for (size_t j = 0; j < cost_matrix.size(); ++j)
            {
                cost_mat.at<float>(static_cast<int>(i), static_cast<int>(j)) = cost_matrix[i][j];
            }
        }

        // Use Hungarian algorithm for assignment
        assignment = hungarian_.solve(cost_mat);
        associate_detections_to_tracks
        (
            detections,
            cost_matrix,
            assignment,
            unmatched_detections,
            unmatched_tracks
        );
    } else
    {
        // All detections are unmatched if no tracks exist
        for (size_t i = 0; i < detections.size(); ++i)
        {
            unmatched_detections.push_back(static_cast<int>(i));
        }
    }

    // Update matched tracks
    for (size_t i = 0; i < assignment.size(); ++i)
    {
        if (assignment[i] != -1)
        {
            tracks_[i]->update(detections[assignment[i]], current_time);
        } else 
        {
            tracks_[i]->mark_missed();
        }
    }

    // Create new tracks for unmatched detections
    for (int idx : unmatched_detections)
    {
        tracks_.push_back(std::make_shared<Track>(next_id_++, detections[idx], current_time));
    }

    // Remove dead tracks
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [](const std::shared_ptr<Track>& track)
                                 {
                                    return track->is_dead();
                                 }), tracks_.end());

    // Return confirmed tracks
    std::vector<Track> result;
    for (const auto& track : tracks_)
    {
        if (track->is_confirmed())
        {
            result.push_back(*track);
        }
    }

    last_update_time_ = current_time;
    return result;
}

std::vector<std::vector<float>> Tracker::create_cost_matrix
(
    const std::vector<Detection>& detections,
    const std::vector<std::shared_ptr<Track>>& tracks
) const noexcept
{
    if (tracks.empty() || detections.empty())
    {
        return {};
    }

    std::vector<std::vector<float>> cost_matrix(tracks.size(),
                                                std::vector<float>(detections.size()));

    for (size_t i = 0; i < tracks.size(); ++i)
    {
        for (size_t j = 0; j < detections.size(); ++j)
        {
            // Using 1 - IoU as cost (lower cost means better match)
            cost_matrix[i][j] = 1.0f - calculate_iou
            (
                {0, 0.0f, tracks[i]->predicted_bbox()},
                detections[j]
            );
        }
    }

    return cost_matrix;
}

float Tracker::calculate_iou
(
    const Detection& det1,
    const Detection& det2
) const noexcept
{
    const auto& rect1 = det1.getBbox();
    const auto& rect2 = det2.getBbox();

    float x1 = std::max(rect1.x, rect2.x);
    float y1 = std::max(rect1.y, rect2.y);
    float x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    float y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);

    float area1 = rect1.width * rect1.height;
    float area2 = rect2.width * rect2.height;
    float union_area = area1 + area2 - intersection;

    return union_area > 0 ? intersection / union_area : 0.0f;
}

void Tracker::associate_detections_to_tracks(
    const std::vector<Detection>& detections,
    const std::vector<std::vector<float>>& cost_matrix,
    std::vector<int>& assignment,
    std::vector<int>& unmatched_detections,
    std::vector<int>& unmatched_tracks) noexcept {
    
    // Initialize all detections as unmatched
    for (size_t i = 0; i < detections.size(); ++i) {
        unmatched_detections.push_back(static_cast<int>(i));
    }
    
    // Find matches based on assignment and cost threshold
    for (size_t i = 0; i < assignment.size(); ++i) {
        int j = assignment[i];
        if (j != -1 && cost_matrix[i][j] < (1.0f - iou_threshold_)) {
            // Valid assignment
            assignment[i] = j;
            // Remove from unmatched detections
            unmatched_detections.erase(
                std::remove(unmatched_detections.begin(), 
                           unmatched_detections.end(), j),
                unmatched_detections.end());
        } else {
            // Invalid assignment
            assignment[i] = -1;
            unmatched_tracks.push_back(static_cast<int>(i));
        }
    }
}