#include "../../include/tracking/hungarian.hpp"
#include <algorithm>
#include <limits>
#include <vector>
#include <functional>

[[nodiscard]] std::vector<int> HungarianAlgorithm::solve(const cv::Mat& cost_matrix)
{
    if (cost_matrix.empty()) return {};
    
    const int n = cost_matrix.rows;
    const int m = cost_matrix.cols;
    
    // Create a copy of the cost matrix
    cv::Mat cost = cost_matrix.clone();
    
    // Initialize mask matrix (0: no assignment, 1: starred, 2: primed)
    cv::Mat mask = cv::Mat::zeros(n, m, CV_8S);
    
    // Initialize row and column covers
    cv::Mat row_cover = cv::Mat::zeros(1, n, CV_8S);
    cv::Mat col_cover = cv::Mat::zeros(1, m, CV_8S);
    
    // Steps of the algorithm
    int step = 1;
    int row = -1, col = -1;
    
    while (step != -1) {
        switch (step) {
            case 1: step1(step, cost); break;
            case 2: step2(step, cost, mask, row_cover, col_cover); break;
            case 3: step3(step, mask, col_cover); break;
            case 4: step4(step, cost, mask, row_cover, col_cover, row, col); break;
            case 5: step5(step, mask, row_cover, col_cover); break;
            case 6: step6(step, cost, row_cover, col_cover); break;
            default: step = -1; break;
        }
    }
    
    // Extract assignments
    std::vector<int> assignments(n, -1);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (mask.at<char>(i, j) == 1) {
                assignments[i] = j;
                break;
            }
        }
    }
    
    return assignments;
}

void HungarianAlgorithm::step1(int& step, cv::Mat& cost_matrix)
{
    // Subtract row minima
    for (int i = 0; i < cost_matrix.rows; ++i)
    {
        double min_val;
        cv::minMaxLoc(cost_matrix.row(i), &min_val);
        cost_matrix.row(i) -= min_val;
    }
    step = 2;
}

void HungarianAlgorithm::step2
(
    int& step, 
    const cv::Mat& cost_matrix,
    cv::Mat& mask,
    cv::Mat& row_cover, cv::Mat& col_cover
)