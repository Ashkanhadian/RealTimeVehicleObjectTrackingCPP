#pragma once

#include <opencv2/opencv.hpp>

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect_<float> bbox;

    // For Hungarian algorithm
    cv::Point2f center() const
    {
        return cv::Point2f(bbox.x + bbox.width / 2.0f,
                           bbox.y + bbox.height / 2.0f);
    }

    float area() const
    {
        return bbox.width * bbox.height;
    }
};

// IoU calculation
inline float calculate_iou(const Detection& det1, const Detection& det2)
{
    const auto& rect1 = det1.bbox;
    const auto& rect2 = det2.bbox;

    float x1 = std::max(rect1.x, rect2.x);
    float y1 = std::max(rect1.y, rect2.y);
    float x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    float y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float union_area = det1.area() + det2.area() - intersection;

    return union_area > 0 ? intersection / union_area : 0.0f;
}