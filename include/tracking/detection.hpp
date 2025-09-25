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

        [[nodiscard]] int getClassId() const noexcept;
        [[nodiscard]] float getConfidence() const noexcept;
        [[nodiscard]] cv::Rect_<float> getBbox() const noexcept;

        [[nodiscard]] cv::Point2f getCenter() const noexcept;
        [[nodiscard]] float getArea() const noexcept;

        void updateBbox(const cv::Rect_<float>& new_bbox) noexcept;
        void updateConfidence(float new_confidence) noexcept;
};

namespace detection_utils
{
    [[nodiscard]] float calculateIoU(const Detection& det1, const Detection& det2) noexcept;
    [[nodiscard]] float calculateIoU(const cv::Rect_<float>& rect1, const cv::Rect_<float>& rect2) noexcept;

    [[nodiscard]] float calculateDistance(const Detection& det1, const Detection& det2) noexcept;
    [[nodiscard]] float calculateDistance(const cv::Point2f& point1, const cv::Point2f& point2) noexcept;

    [[nodiscard]] cv::Rect_<float> toX1Y1X2Y2(const cv::Rect_<float>& bbox) noexcept;
    // [[nodiscard]] cv::Rect_<float> toXYWH(const cv::Rect_<float>& bbox) noexcept;

    [[nodiscard]] cv::Mat createIoUCostMatrix
    (
        const std::vector<Detection>& detections,
        const std::vector<std::shared_ptr<Track>>& tracks
    );

    [[nodiscard]] cv::Mat createDistanceCostMatrix
    (
        const std::vector<Detection>& detections,
        const std::vector<std::shared_ptr<Track>>& tracks,
        float max_distance = 100.0f
    );
}