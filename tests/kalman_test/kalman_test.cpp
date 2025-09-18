#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "../../include/tracking/kalman_filter.hpp"
#include "../../include/tracking/detection.hpp"

TEST(KalmanFilterTest, Initialization)
{
    cv::Rect_<float> bbox(10.0f, 20.0f, 30.0f, 40.0f);
    Detection detection(1, 0.8f, bbox);
    KalmanFilter kf(detection);

    auto state = kf.state();
    EXPECT_FLOAT_EQ(state.at<float>(0), 10.0f);
    EXPECT_FLOAT_EQ(state.at<float>(1), 20.0f);
    EXPECT_FLOAT_EQ(state.at<float>(2), 30.0f);
    EXPECT_FLOAT_EQ(state.at<float>(3), 40.0f);
}

TEST(KalmanFilterTest, Prediction)
{
    cv::Rect_<float> bbox(10.0f, 20.0f, 30.0f, 40.0f);
    Detection detection(1, 0.8f, bbox);
    KalmanFilter kf(detection);

    kf.predict();
    auto predicted_bbox = kf.predicted_bbox();

    EXPECT_NEAR(predicted_bbox.x, 10.0f, 1.0f);
    EXPECT_NEAR(predicted_bbox.y, 20.0f, 1.0f);
    EXPECT_NEAR(predicted_bbox.width, 30.0f, 1.0f);
    EXPECT_NEAR(predicted_bbox.height, 40.0f, 1.0f);
}

TEST(KalmanFilterTest, Update)
{
    cv::Rect_<float> initial_bbox(10.0f, 20.0f, 30.0f, 40.0f);
    Detection initial_detection(1, 0.8f, initial_bbox);
    KalmanFilter kf(initial_detection);

    cv::Rect_<float> new_bbox(15.0f, 25.0f, 35.0f, 45.0f);
    Detection new_detections(1, 0.9f, new_bbox);

    kf.predict();
    kf.update(new_detections);

    auto corrected_bbox = kf.corrected_bbox();

    EXPECT_NEAR(corrected_bbox.x, 15.0f, 5.0f);
    EXPECT_NEAR(corrected_bbox.y, 25.0f, 5.0f);
    EXPECT_NEAR(corrected_bbox.width, 35.0f, 5.0f);
    EXPECT_NEAR(corrected_bbox.height, 45.0f, 5.0f);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}