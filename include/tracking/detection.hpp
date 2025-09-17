#pragma once

#include <opencv2/opencv.hpp>
#include <memory>

class Track;

class Detection
{
    private:
        int class_id;
        float confidence;
        cv::Rect_<float> bbox;

    public:
        Detection(int class_id, float confidence, const cv::Rect_<float>& bbox);

        int getClassId() const noexcept;
        float getConfidence() const noexcept;
        cv::Rect_<float> getBbox() const noexcept;

        cv::Point2f getCenter() const noexcept;
        float getArea() const noexcept;

        void updateBbox(const cv::Rect_<float>& new_bbox);
        void updateConfidence(float new_confidence);
};

namespace detection_utils
{
    float calculateIoU(const Detection& det1, const Detection& det2);
    float calculateIoU(const cv::Rect_<float>& rect1, const cv::Rect_<float>& rect2);

    float calculateDistance(const Detection& det1, const Detection& det2);
    float calculateDistance(const cv::Point2f& point1, const cv::Point2f& point2);

    cv::Rect_<float> toX1Y1X2Y2(const cv::Rect_<float>& bbox);
    cv::Rect_<float> toXYWH(const cv::Rect_<float>& bbox);

    cv::Mat createIoUCostMatrix
    (
        const std::vector<Detection>& detections,
        const std::vector<std::shared_ptr<Track>>& tracks
    );

    cv::Mat createDistanceCostMatrix
    (
        const std::vector<Detection>& detections,
        const std::vector<std::shared_ptr<Track>>& tracks,
        float max_distance = 100.0f
    );
}