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

    if (tracks_.empty())
    {
        // Create new tracks for all detections
        for (const auto& detection : detections)
        {
            tracks_.push_back(std::make_shared<Track>(next_id_++, detection, current_time));
        }

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

    if (detections.empty())
    {
        // Mark all tracks as missed
        for (auto& track : tracks_)
        {
            track->mark_missed();
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

    auto cost_matrix = create_cost_matrix(detections, tracks_);

    // std::cout << "Cost matrix size: " << cost_matrix.size() << "x" 
    //           << (cost_matrix.empty() ? 0 : cost_matrix[0].size()) << std::endl;
    // std::cout << "Tracks count: " << tracks_.size() << std::endl;
    // std::cout << "Detections count: " << detections.size() << std::endl;

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
            for (size_t j = 0; j < cost_matrix[0].size(); ++j)
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

        // std::cout << "Assignment result size: " << assignment.size() << std::endl;
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
        if (assignment[i] != -1 && assignment[i] < detections.size())
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
                                                std::vector<float>(detections.size(), 1.0f));

    for (size_t i = 0; i < tracks.size(); ++i)
    {
        for (size_t j = 0; j < detections.size(); ++j)
        {
            // Using 1 - IoU as cost (lower cost means better match)
            float iou = detection_utils::calculateIoU
            (
                tracks[i]->predicted_bbox(),
                detections[j].getBbox()
            );
            cost_matrix[i][j] = 1.0f - iou;
        }
    }

    return cost_matrix;
}

void Tracker::associate_detections_to_tracks
(
    const std::vector<Detection>& detections,
    const std::vector<std::vector<float>>& cost_matrix,
    std::vector<int>& assignment,
    std::vector<int>& unmatched_detections,
    std::vector<int>& unmatched_tracks
) noexcept
{
    unmatched_detections.clear();
    unmatched_tracks.clear();

    // Initialize all detections as unmatched
    for (int i = 0; i < detections.size(); i++)
    {
        unmatched_detections.push_back(i);
    }

    // Process assignments
    for (int i = 0; i < assignment.size(); i++)
    {
        int j = assignment[i];

        if (j != -1 && 
            j < detections.size() && 
            cost_matrix[i][j] <= iou_threshold_)
        {
            // Valid assignment - remove from unmatched detections
            auto it = std::find(unmatched_detections.begin(), unmatched_detections.end(), j);
            if (it != unmatched_detections.end())
            {
                unmatched_detections.erase(it);
            }
        } else
        {
            // Invalid assignment - mark track as unmatched
            unmatched_tracks.push_back(i);
        }
    }
}