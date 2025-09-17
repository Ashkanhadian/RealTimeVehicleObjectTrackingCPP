#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

class HungarianAlgorithm
{
    public:
        HungarianAlgorithm() = default;

        [[nodiscard]] std::vector<int> solve(const cv::Mat& cost_matrix);

    private:
        void step1(int& step, cv::Mat& cost_matrix) noexcept;

        void step2(int& step, 
                   const cv::Mat& cost_matrix,
                   cv::Mat& mask,
                   cv::Mat& row_cover,
                   cv::Mat& col_cover) noexcept;

        void step3(int& step,
                   const cv::Mat& mask,
                   cv::Mat& col_cover) noexcept;

        void step4(int& step, 
                   const cv::Mat& cost_matrix,
                   cv::Mat& mask,
                   cv::Mat& row_cover,
                   cv::Mat& col_cover,
                   int& row,
                   int& col) noexcept;

        void step5(int& step, 
                   cv::Mat& mask,
                   cv::Mat& row_cover,
                   cv::Mat& col_cover) noexcept;

        void step6(int& step, 
                   cv::Mat& cost_matrix,
                   const cv::Mat& row_cover,
                   const cv::Mat& col_cover) noexcept;

        [[nodiscard]] bool found_a_zero(int& row, 
                         int& col, 
                         const cv::Mat& cost_matrix, 
                         const cv::Mat& row_cover, 
                         const cv::Mat& col_cover) const noexcept;

        [[nodiscard]] bool star_in_row(int row, const cv::Mat& mask_matrix) const noexcept;
        void find_star_in_row(int row, int& col, const cv::Mat& mask_matrix) const noexcept;
        [[nodiscard]] bool star_in_col(int col, const cv::Mat& mask_matrix) const noexcept;
        void find_star_in_col(int col, int& row, const cv::Mat& mask_matrix) const noexcept;
        [[nodiscard]] bool prime_in_row(int row, const cv::Mat& mask_matrix) const noexcept;
        void find_prime_in_row(int row, int& col, const cv::Mat& mask_matrix) const noexcept;
};