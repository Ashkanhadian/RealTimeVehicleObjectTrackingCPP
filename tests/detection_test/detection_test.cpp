#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "../../include/tracking/detection.hpp"

TEST(DetectionTest, ConstructionAndGetters)
{
    cv::Rect_<float> bbox(10.0f, 20.0f, 30.0f, 40.0f);
    Detection detection(1, 0.8f, bbox);

    EXPECT_EQ(detection.getClassId(), 1);
    EXPECT_FLOAT_EQ(detection.getConfidence(), 0.8f);
    EXPECT_EQ(detection.getBbox(), bbox);
    EXPECT_EQ(detection.getCenter(), cv::Point2f(25.0f, 40.0f));
    EXPECT_FLOAT_EQ(detection.getArea(), 1200.0f);
}

TEST(DetectionTest, UpdateMethods)
{
    cv::Rect_<float> initial_bbox(10.0f, 20.0f, 30.0f, 40.0f);
    Detection detection(1, 0.8f, initial_bbox);

    cv::Rect_<float> new_bbox(15.0f, 25.0f, 35.0f, 45.0f);
    detection.updateBbox(new_bbox);
    detection.updateConfidence(0.9f);

    EXPECT_EQ(detection.getBbox(), new_bbox);
    EXPECT_FLOAT_EQ(detection.getConfidence(), 0.9f);
}

TEST(DetectionUtilsTest, IoUCalculation)
{
    cv::Rect_<float> rect1(0.0f, 0.0f, 10.0f, 10.0f);
    cv::Rect_<float> rect2(5.0f, 5.0f, 10.0f, 10.0f);

    float iou = detection_utils::calculateIoU(rect1, rect2);
    EXPECT_FLOAT_EQ(iou, 25.0f / 175.0f);
}

TEST(DetectionUtilsTest, DistanceCalculation)
{
    cv::Point2f point1(0.0f, 0.0f);
    cv::Point2f point2(3.0f, 4.0f);

    float distance = detection_utils::calculateDistance(point1, point2);
    EXPECT_FLOAT_EQ(distance, 5.0f);
}

TEST(DetectionUtilsTest, BBoxConversion)
{
    cv::Rect_<float> xywh_bbox(10.0f, 20.0f, 30.0f, 40.0f);
    cv::Rect_<float> x1y1x2y2_bbox = detection_utils::toX1Y1X2Y2(xywh_bbox);
    
    EXPECT_FLOAT_EQ(x1y1x2y2_bbox.x, 10.0f);
    EXPECT_FLOAT_EQ(x1y1x2y2_bbox.y, 20.0f);
    EXPECT_FLOAT_EQ(x1y1x2y2_bbox.width, 40.0f);
    EXPECT_FLOAT_EQ(x1y1x2y2_bbox.height, 60.0f);
}


int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}