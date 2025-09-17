#include "../../include/tracking/detection.hpp"
#include "../../include/tracking/track.hpp"

#include <cmath>

Detection::Detection(int class_id, float confidence, const cv::Rect_<float>& bbox)
    : class_id(class_id), confidence(confidence), bbox(bbox) {}

[[nodiscard]] int Detection::getClassId() const noexcept
{
    return class_id;
}

[[nodiscard]] float Detection::getConfidence() const noexcept
{
    return confidence;
}

[[nodiscard]] cv::Rect_<float> Detection::getBbox() const noexcept
{
    return bbox;
}

[[nodiscard]] cv::Point2f Detection::getCenter() const noexcept
{
    return cv::Point2f(bbox.x + bbox.width / 2.0f, bbox.y + bbox.height / 2.0f);
}

[[nodiscard]] float Detection::getArea() const noexcept
{
    return bbox.width * bbox.height;
}

[[nodiscard]] void Detection::updateBbox(const cv::Rect_<float>& new_bbox) noexcept
{
    bbox = new_bbox;
}

void Detection::updateConfidence(float new_confidence) noexcept
{
    confidence = new_confidence;
}

[[nodiscard]] float detection_utils::calculateIoU
(
    const Detection& det1,
    const Detection& det2
) noexcept
{
    return calculateIoU(det1.getBbox(), det2.getBbox());
}

[[nodiscard]] float detection_utils::calculateIoU
(
    const cv::Rect_<float>& rect1,
    const cv::Rect_<float>& rect2
) noexcept
{
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

[[nodiscard]] float detection_utils::calculateDistance
(
    const Detection& det1,
    const Detection& det2
) noexcept
{
    return calculateDistance(det1.getCenter(), det2.getCenter());
}

[[nodiscard]] float detection_utils::calculateDistance
(
    const cv::Point2f& point1,
    const cv::Point2f& point2
) noexcept
{
    float dx = point2.x - point1.x;
    float dy = point2.y - point1.y;
    return std::sqrt(dx * dx + dy * dy);
}

[[nodiscard]] cv::Rect_<float> detection_utils::toX1Y1X2Y2
(
    const cv::Rect_<float>& bbox
) noexcept
{
    return cv::Rect_<float>(bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height);
}

[[nodiscard]] cv::Rect_<float> detection_utils::toXYWH
(
    const cv::Rect_<float>& bbox
) noexcept
{
    // NOTE: This assumes the input bbox is in (x1, y1, x2, y2) format
    return cv::Rect_<float>(bbox.x, bbox.y, bbox.width - bbox.x, bbox.height - bbox.y);
}

[[nodiscard]] cv::Mat detection_utils::createIoUCostMatrix
(
    const std::vector<Detection>& detections,
    const std::vector<std::shared_ptr<Track>>& tracks
)
{
    if (tracks.empty() || detections.empty())
    {
        return cv::Mat();
    }

    cv::Mat cost_matrix(tracks.size(), detections.size(), CV_32F);

    for (size_t i = 0; i < tracks.size(); ++i)
    {
        // Create a detection from the track's predicted bounding box
        Detection track_det(-1, 0.0f, tracks[i]->predicted_bbox());

        for (size_t j = 0; j < detections.size(); ++j)
        {
            // Calculate IoU and use 1 - IoU as cost (lower is better)
            float iou = calculateIoU(track_det, detections[j]);
            cost_matrix.at<float>(i, j) = 1.0f - iou;
        }
    }

    return cost_matrix;
}

[[nodiscard]] cv::Mat detection_utils::createDistanceCostMatrix
(
    const std::vector<Detection>& detections,
    const std::vector<std::shared_ptr<Track>>& tracks,
    float max_distance = 100.0f
)
{
    if (tracks.empty() || detections.empty())
    {
        return cv::Mat();
    }

    cv::Mat cost_matrix(tracks.size(), detections.size(), CV_32F);

    for (size_t i = 0; i < tracks.size(); ++i)
    {
        // Get the center of the track's predicted bounding box
        cv::Rect_<float> track_bbox = tracks[i]->predicted_bbox();
        cv::Point2f track_center(track_bbox.x + track_bbox.width / 2.0f,
                                 track_bbox.y + track_bbox.height / 2.0f);

        for (size_t j = 0; j < detections.size(); ++j)
        {
            // Calculate distance and normalize to [0, 1] range
            float distance = calculateDistance(track_center, detections[j].getCenter());
            cost_matrix.at<float>(i, j) = std::min(distance / max_distance, 1.0f);
        }
    }

    return cost_matrix;
}