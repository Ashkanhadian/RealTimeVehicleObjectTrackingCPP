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

float detection_utils::calculateDistance
(
    const Detection& det1,
    const Detection& det2
)
{
    return calculateDistance(det1.getCenter(), det2.getCenter());
}

float detection_utils::calculateDistance
(
    const cv::Point2f& point1,
    const cv::Point2f& point2
)
{
    float dx = point2.x - point1.x;
    float dy = point2.y - point1.y;
    return std::sqrt(dx * dx + dy * dy);
}

cv::Rect_<float> detection_utils::toX1Y1X2Y2
(
    const cv::Rect_<float>& bbox
)
{
    return cv::Rect_<float>(bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height);
}

cv::Rect_<float> detection_utils::toXYWH
(
    const cv::Rect_<float>& bbox
)
{
    return cv::Rect_<float>(bbox.x, bbox.y, bbox.width - bbox.x, bbox.height - bbox.y);
}

cv::Mat detection_utils::createIoUCostMatrix
(
    const std::vector<Detection>& detections,
    const std::vector<std::shared_ptr<Track>>& tracks
)
{

}

cv::Mat detection_utils::createDistanceCostMatrix
(
    const std::vector<Detection>& detections,
    const std::vector<std::shared_ptr<Track>>& tracks,
    float max_distance = 100.0f
)
{
    
}