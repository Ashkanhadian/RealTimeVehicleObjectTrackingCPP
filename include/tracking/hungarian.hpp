#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

class HungarianAlgorithm
{
    public:
        HungarianAlgorithm() = default;

        [[nodiscard]] std::vector<int> solve(const cv::Mat& cost_matrix);

    private:
        void step1(int& step, cv::Mat& cost_matrix);

        void step2(int& step, 
                   const cv::Mat& cost_matrix,
                   cv::Mat& mask,
                   cv::Mat& row_cover,
                   cv::Mat& col_cover);

        void step3(int& step,
                   const cv::Mat& mask,
                   cv::Mat& col_cover);

        void step4(int& step, 
                   const cv::Mat& cost_matrix,
                   cv::Mat& mask,
                   cv::Mat& row_cover,
                   cv::Mat& col_cover,
                   int& row,
                   int& col);

        void step5(int& step, 
                   cv::Mat& mask,
                   cv::Mat& row_cover,
                   cv::Mat& col_cover);

        void step6(int& step, 
                   cv::Mat& cost_matrix,
                   const cv::Mat& row_cover,
                   const cv::Mat& col_cover);

        [[nodiscard]] bool found_a_zero(int& row, 
                         int& col, 
                         const cv::Mat& cost_matrix, 
                         const cv::Mat& row_cover, 
                         const cv::Mat& col_cover);

        [[nodiscard]] bool star_in_row(int row, const cv::Mat& mask_matrix);
        void find_star_in_row(int row, int& col, const cv::Mat& mask_matrix);
        [[nodiscard]] bool star_in_col(int col, const cv::Mat& mask_matrix);
        void find_star_in_col(int col, int& row, const cv::Mat& mask_matrix);
};