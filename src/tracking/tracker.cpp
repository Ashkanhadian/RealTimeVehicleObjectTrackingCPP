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
            // Filter by confidence and vehicle classes
            if (detection.getConfidence() > 0.75f &&
                (detection.getClassId() == 2 || detection.getClassId() == 3 ||
                 detection.getClassId() == 5 || detection.getClassId() == 7))
            {
                tracks_.push_back(std::make_shared<Track>(next_id_++, detection, current_time));
                std::cout << "Created new track ID: " << (next_id_ - 1) << std::endl;
            }
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

        // Remove dead tracks and invalid tracks
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                     [](const std::shared_ptr<Track>& track)
                                     {
                                        return track->is_dead() ||
                                               !track->is_valid();
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

    std::cout << "Before matching - Tracks: " << tracks_.size() << ", Detections: " << detections.size() << std::endl;

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

        std::cout << "After matching - Matched: " << (assignment.size() - std::count(assignment.begin(), assignment.end(), -1)) 
                  << ", Unmatched detections: " << unmatched_detections.size() 
                  << ", Unmatched tracks: " << unmatched_tracks.size() << std::endl;
    } else
    {
        // All detections are unmatched if no tracks exist
        for (size_t i = 0; i < detections.size(); ++i)
        {
            unmatched_detections.push_back(static_cast<int>(i));
        }
        // All tracks are unmatched
        for (size_t i = 0; i < tracks_.size(); ++i)
        {
            unmatched_tracks.push_back(static_cast<int>(i));
        }
    }

    // Update matched tracks
    for (size_t i = 0; i < assignment.size(); ++i)
    {
        if (assignment[i] != -1 && assignment[i] < detections.size())
        {
            tracks_[i]->update(detections[assignment[i]], current_time);
            std::cout << "Updated track " << tracks_[i]->id() << " with detection " << assignment[i] << std::endl;
        } else 
        {
            tracks_[i]->mark_missed();
        }
    }

    // Create new tracks only for high_confidence unmatched detections
    for (int idx : unmatched_detections)
    {
        const auto& detection = detections[idx];

        // Filter by confidence and vehicle classes
        if (detection.getConfidence() > 0.75f &&
            (detection.getClassId() == 2 || detection.getClassId() == 3 ||
             detection.getClassId() == 5 || detection.getClassId() == 7))
        {
            // Additional check: ensure the detection doesn't overlap too much with existing tracks
            bool is_duplicate = false;
            for (const auto& track : tracks_)
            {
                float iou = detection_utils::calculateIoU(detection.getBbox(), track->current_bbox());

                if (iou > 0.5f) // If overlap is too high, it's likely a duplicate
                {
                    is_duplicate = true;
                    break;
                }
            }

            if (!is_duplicate)
            {
                tracks_.push_back(std::make_shared<Track>(next_id_++, detection, current_time));
                std::cout << "Created new track ID: " << (next_id_ - 1) << " from unmatched detection" << std::endl;
            }
        }
    }

    // Mark unmatched tracks as missed
    for (int idx : unmatched_tracks)
    {
        if (idx < tracks_.size())
        {
            tracks_[idx]->mark_missed();
            std::cout << "Marked track " << tracks_[idx]->id() << " as missed" << std::endl;
        }
    }

    // Remove dead tracks, invalid tracks, and tracks with too few hits
    size_t before_removal = tracks_.size();
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [](const std::shared_ptr<Track>& track)
                                 {
                                    return track->is_dead() ||
                                           !track->is_valid();
                                 }), tracks_.end());

    if (tracks_.size() != before_removal)
    {
        std::cout << "Removed " << (before_removal - tracks_.size()) << " tracks" << std::endl;
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

    std::cout << "Final - Active tracks: " << tracks_.size() << ", Confirmed tracks: " << result.size() << std::endl;

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
        auto track_bbox = tracks[i]->predicted_bbox();
        float track_area = track_bbox.width * track_bbox.height;

        for (size_t j = 0; j < detections.size(); ++j)
        {
            auto det_bbox = detections[j].getBbox();
            float det_area = det_bbox.width * det_bbox.height;

            // Calculate IoU cost
            float iou = 1.0f - detection_utils::calculateIoU(track_bbox, det_bbox);

            // Calculate size consistency cost
            float size_ratio = std::max(track_area, det_area) / std::min(track_area, det_area);
            float size_cost = std::min((size_ratio - 1.0f) * 0.5f, 1.0f);

            // Calculate center distance cost (normalized by image size)
            auto track_center = cv::Point2f(track_bbox.x + track_bbox.width / 2.0f,
                                            track_bbox.y + track_bbox.height / 2.0f);
            auto det_center = detections[j].getCenter();
            float distance = detection_utils::calculateDistance(track_center, det_center);
            float distance_cost = std::min(distance / 100.0f, 1.0f);    // Normalize by 100 pixels

            // Combined cost (weighted)
            cost_matrix[i][j] = 0.5f * iou + 0.3f * size_cost + 0.2f * distance_cost;
        }
    }

    return cost_matrix;
}

// void Tracker::associate_detections_to_tracks(
//     const std::vector<Detection>& detections,
//     const std::vector<std::vector<float>>& cost_matrix,
//     std::vector<int>& assignment,
//     std::vector<int>& unmatched_detections,
//     std::vector<int>& unmatched_tracks
// ) noexcept
// {
//     unmatched_detections.clear();
//     unmatched_tracks.clear();

//     // Initialize all detections as unmatched
//     for (int i = 0; i < detections.size(); i++)
//     {
//         unmatched_detections.push_back(i);
//     }

//     // Initialize all tracks as unmatched
//     for (int i = 0; i < tracks_.size(); i++)
//     {
//         unmatched_tracks.push_back(i);
//     }

//     // Process assignments with dynamic threshold
//     for (int i = 0; i < assignment.size(); i++)
//     {
//         int j = assignment[i];

//         if (j != -1 && j < detections.size())
//         {
//             float cost = cost_matrix[i][j];
//             float iou = 1.0f - cost; // Convert cost back to IoU
            
//             // Use a simple threshold for now
//             if (iou >= iou_threshold_)
//             {
//                 // Valid assignment - remove from unmatched lists
//                 auto det_it = std::find(unmatched_detections.begin(), unmatched_detections.end(), j);
//                 if (det_it != unmatched_detections.end())
//                 {
//                     unmatched_detections.erase(det_it);
//                 }
                
//                 auto trk_it = std::find(unmatched_tracks.begin(), unmatched_tracks.end(), i);
//                 if (trk_it != unmatched_tracks.end())
//                 {
//                     unmatched_tracks.erase(trk_it);
//                 }
                
//                 std::cout << "Matched track " << i << " with detection " << j << " (IoU: " << iou << ")" << std::endl;
//             }
//         }
//     }
// }

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

    // Initialize all tracks as unmatched initially
    for (int i = 0; i < tracks_.size(); i++)
    {
        unmatched_tracks.push_back(i);
    }

    // Process assignments
    for (int i = 0; i < assignment.size(); i++)
    {
        int j = assignment[i];

        if (j != -1 && j < detections.size())
        {
            float iou = 1.0f - cost_matrix[i][j];

            // Dynamic threshold based on track confidence
            float dynamic_threshold = iou_threshold_;
            if (tracks_[i]->is_confirmed())
            {
                dynamic_threshold *= 0.8f; // Stricter for confirmed tracks
            } else
            {
                dynamic_threshold *= 1.2f; // More lenient for new tracks
            }

            if (iou >= dynamic_threshold)
            {
                // Valid assignment - remove from unmatched lists
                auto det_it = std::find(unmatched_detections.begin(),
                                        unmatched_detections.end(), j);
                if (det_it != unmatched_detections.end())
                {
                    unmatched_detections.erase(det_it);
                }

                auto trk_it = std::find(unmatched_tracks.begin(),
                                        unmatched_tracks.end(), i);

                if (trk_it != unmatched_tracks.end())
                {
                    unmatched_tracks.erase(trk_it);
                }
            }
        }
    }
}

// void Tracker::associate_detections_to_tracks
// (
//     const std::vector<Detection>& detections,
//     const std::vector<std::vector<float>>& cost_matrix,
//     std::vector<int>& assignment,
//     std::vector<int>& unmatched_detections,
//     std::vector<int>& unmatched_tracks
// ) noexcept
// {
//     unmatched_detections.clear();
//     unmatched_tracks.clear();

//     // Initialize all detections as unmatched
//     for (int i = 0; i < detections.size(); i++)
//     {
//         unmatched_detections.push_back(i);
//     }

//     // Initialize all tracks as unmatched initially
//     for (int i = 0; i < tracks_.size(); i++)
//     {
//         unmatched_tracks.push_back(i);
//     }

//     // Process assignments
//     for (int i = 0; i < assignment.size(); i++)
//     {
//         int j = assignment[i];

//         if (j != -1 && 
//             j < detections.size() && 
//             cost_matrix[i][j] <= iou_threshold_)
//         {
//             // Valid assignment - remove from unmatched detections
//             auto it = std::find(unmatched_detections.begin(), unmatched_detections.end(), j);
//             if (it != unmatched_detections.end())
//             {
//                 unmatched_detections.erase(it);
//             }
//         } else
//         {
//             // Invalid assignment - mark track as unmatched
//             unmatched_tracks.push_back(i);
//         }
//     }
// }