#include "../../include/tracking/detection.hpp"

float detection_utils::calculateIoU
(
    const Detection& det1,
    const Detection& det2
)
{
    return calculateIoU(det1.getBbox(), det2.getBbox());
}

float detection_utils::calculateIoU
(
    const cv::Rect_<float>& rect1,
    const cv::Rect_<float>& rect2
)
{
    float x1 = std::max(rect1.x, rect2.x);
    float y1 = std::max(rect1.y, rect2.y);
    float x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    float y2 = std::min(rect1.x + rect1.height, rect2.y + rect2.height);

    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);

    float area1 = rect1.width * rect1.height;
    float area2 = rect2.width * rect2.height;
    float union_area = area1 + area2 - intersection;

    return union_area > 0 ? intersection / union_area : 0.0f;
}