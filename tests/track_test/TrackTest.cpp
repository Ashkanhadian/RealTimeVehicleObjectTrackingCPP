#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "../../include/tracking/tracker.hpp"
#include "../../include/tracking/detection.hpp"

TEST(TrackerTest, Initialization)
{
    Tracker tracker(0.3f);

    EXPECT_EQ(tracker.next_id(), 0);
    EXPECT_TRUE(tracker.tracks().empty());
}

TEST(TrackerTest, FirstUpdateCreatesTracks)
{
    Tracker tracker(0.3f);
    
    std::vector<Detection> detections;

    cv::Rect_<float> bbox(10.0f, 20.0f, 30.0f, 40.0f);
    detections.emplace_back(1, 0.8f, bbox);

    auto confirmed_tracks = tracker.update(detections);

    EXPECT_TRUE(confirmed_tracks.empty());
    EXPECT_EQ(tracker.tracks().size(), 1);
    EXPECT_EQ(tracker.next_id(), 1);
}

TEST(TrackerTest, UpdateWithMatchingDetection)
{
    Tracker tracker(0.3f);

    std::vector<Detection> detections;

    cv::Rect_<float> bbox(10.0f, 20.0f, 30.0f, 40.0f);
    detections.emplace_back(1, 0.8f, bbox);

    tracker.update(detections);

    auto confirmed_tracks = tracker.update(detections);

    EXPECT_TRUE(confirmed_tracks.empty());
    EXPECT_EQ(tracker.tracks().size(), 1);
    EXPECT_EQ(tracker.tracks()[0]->hits(), 2);
}

TEST(TrackerTest, UpdateWithNonMatchingDetectionCreatesNewTrack)
{
    Tracker tracker(0.3f);

    std::vector<Detection> detections;

    cv::Rect_<float> bbox1(10.0f, 20.0f, 30.0f, 40.0f);
    detections.emplace_back(1, 0.8f, bbox1);

    tracker.update(detections);

    detections.clear();
    
    cv::Rect_<float> bbox2(1000.0f, 2000.0f, 30.0f, 40.f);
    detections.emplace_back(1, 0.8f, bbox2);

    auto confirmed_tracks = tracker.update(detections);

    EXPECT_EQ(tracker.tracks().size(), 2);
    EXPECT_EQ(tracker.next_id(), 2);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}