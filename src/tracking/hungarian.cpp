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
    
    const int k = std::max(n, m);
    
    // Create a square cost matrix by padding with zeros
    cv::Mat square_cost = cv::Mat::zeros(k, k, CV_32F);

    // Copy the original cost values
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            square_cost.at<float>(i, j) = cost_matrix.at<float>(i, j);
        }
    }

    // Use the square matrix for the Hungarian Algorithm
    cv::Mat cost = square_cost.clone();

    // Initialize mask matrix (0: no assignment, 1: starred, 2: primed)
    cv::Mat mask = cv::Mat::zeros(k, k, CV_8S);
    
    // Initialize row and column covers
    cv::Mat row_cover = cv::Mat::zeros(1, k, CV_8S);
    cv::Mat col_cover = cv::Mat::zeros(1, k, CV_8S);
    
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
    
    // Extract assignments (only for the original rows)
    std::vector<int> assignments(n, -1);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {   // Only consider original columns
            if (mask.at<char>(i, j) == 1) {
                assignments[i] = j;
                break;
            }
        }
    }
    
    return assignments;
}

void HungarianAlgorithm::step1(int& step, cv::Mat& cost_matrix) noexcept
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
    cv::Mat& row_cover, 
    cv::Mat& col_cover
) noexcept
{
    for (int i = 0; i < cost_matrix.rows; ++i)
    {
        for (int j = 0; j < cost_matrix.cols; ++j)
        {
            if (
                cost_matrix.at<float>(i, j) == 0 && 
                row_cover.at<char>(i) == 0 &&
                col_cover.at<char>(j) == 0
            )
            {
                mask.at<char>(i, j) = 1;
                row_cover.at<char>(i) = 1;
                col_cover.at<char>(j) = 1;
            }
        }
    }

    row_cover.setTo(0);
    col_cover.setTo(0);
    step = 3;
}

void HungarianAlgorithm::step3
(
    int& step,
    const cv::Mat& mask,
    cv::Mat& col_cover
) noexcept
{
    int col_count = 0;
    for (int i = 0; i < mask.rows; ++i)
    {
        for (int j = 0; j < mask.cols; ++j)
        {
            if (mask.at<char>(i, j) == 1)
            {
                col_cover.at<char>(j) = 1;
                col_count++;
            }
        }
    }

    step = (col_count >= mask.rows) ? -1 : 4;
}

void HungarianAlgorithm::step4
(
    int& step,
    const cv::Mat& cost_matrix,
    cv::Mat& mask,
    cv::Mat& row_cover,
    cv::Mat& col_cover,
    int& row,
    int& col
) noexcept
{
    // Find a non-covered zero and prime it
    if (found_a_zero(row, col, cost_matrix, row_cover, col_cover))
    {
        mask.at<char>(row, col) = 2;
        
        if (star_in_row(row, mask))
        {
            find_star_in_row(row, col, mask);
            row_cover.at<char>(row) = 1;
            col_cover.at<char>(col) = 0;
            step = 4;
        } else
        {
            step = 5;
        }
    } else
    {
        step = 6;
    }
}

void HungarianAlgorithm::step5
(
    int& step,
    cv::Mat& mask,
    cv::Mat& row_cover,
    cv::Mat& col_cover
) noexcept
{
    // Construct a series of alternating primed and starred zeros
    std::vector<cv::Point> path;
    int row = -1, col = -1;

    // Find the initial primed zero (which should be uncovered)
    bool found = false;
    for (int i = 0; i < mask.rows && !found; ++i)
    {
        for (int j = 0; j < mask.cols && !found; ++j)
        {
            if (mask.at<char>(i, j) == 2 && row_cover.at<char>(i) == 0 && col_cover.at<char>(j) == 0)
            {
                row = i;
                col = j;
                found = true;
                path.emplace_back(j, i);
            }
        }
    }

    if (!found) {
        step = 4;
        return;
    }

    // Follow the path of alternating primed and starred zeros
    bool done = false;
    while (!done)
    {
        // Find starred zero in the same column
        if (star_in_col(path.back().x, mask))
        {
            int star_row;
            find_star_in_col(path.back().x, star_row, mask);
            path.emplace_back(path.back().x, star_row);
        } else
        {
            done = true;
            break;
        }

        // Find primed zero in the same row
        if (prime_in_row(path.back().y, mask))
        {
            int prime_col;
            find_prime_in_row(path.back().y, prime_col, mask);
            path.emplace_back(prime_col, path.back().y);
        } else
        {
            done = true;
        }
    }

    // Augment the path
    for (const auto& p : path)
    {
        if (mask.at<char>(p.y, p.x) == 1)
        {
            mask.at<char>(p.y, p.x) = 0;
        } else
        {
            mask.at<char>(p.y, p.x) = 1;
        }
    }

    // Reset covers and clear primes
    row_cover.setTo(0);
    col_cover.setTo(0);
    
    for (int i = 0; i < mask.rows; ++i)
    {
        for (int j = 0; j < mask.cols; ++j)
        {
            if (mask.at<char>(i, j) == 2)
            {
                mask.at<char>(i, j) = 0;
            }
        }
    }

    step = 3;
}

void HungarianAlgorithm::step6
(
    int& step,
    cv::Mat& cost_matrix,
    const cv::Mat& row_cover,
    const cv::Mat& col_cover
) noexcept
{
    // Find the minimum uncovered value
    float min_val = std::numeric_limits<float>::max();

    for (int i = 0; i < cost_matrix.rows; ++i)
    {
        for (int j = 0; j < cost_matrix.cols; ++j)
        {
            if (row_cover.at<char>(i) == 0 && col_cover.at<char>(j) == 0)
            {
                if (cost_matrix.at<float>(i, j) < min_val)
                {
                    min_val = cost_matrix.at<float>(i, j);
                }
            }
        }
    }

    // Add the minimum value to covered rows and subtract it from uncovered columns
    for (int i = 0; i < cost_matrix.rows; ++i)
    {
        for (int j = 0; j < cost_matrix.cols; ++j)
        {
            if (row_cover.at<char>(i) == 1)
            {
                cost_matrix.at<float>(i, j) += min_val;
            }
            if (col_cover.at<char>(j) == 0)
            {
                cost_matrix.at<float>(i, j) -= min_val;
            }
        }
    }

    step = 4;
}

[[nodiscard]] bool HungarianAlgorithm::found_a_zero
(
    int& row,
    int& col,
    const cv::Mat& cost_matrix,
    const cv::Mat& row_cover,
    const cv::Mat& col_cover
) const noexcept
{
    row = -1;
    col = -1;

    for (int i = 0; i < cost_matrix.rows; ++i)
    {
        if (row_cover.at<char>(i) == 0)
        {
            for (int j = 0; j < cost_matrix.cols; ++j)
            {
                if (col_cover.at<char>(j) == 0 && cost_matrix.at<float>(i, j) == 0)
                {
                    row = i;
                    col = j;
                    return true;
                }
            }
        }
    }

    return false;
}

[[nodiscard]] bool HungarianAlgorithm::star_in_row
(
    int row,
    const cv::Mat& mask_matrix
) const noexcept
{
    for (int j = 0; j < mask_matrix.cols; ++j)
    {
        if (mask_matrix.at<char>(row, j) == 1)
            return true;
    }
    return false;
}

void HungarianAlgorithm::find_star_in_row
(
    int row,
    int& col,
    const cv::Mat& mask_matrix
) const noexcept
{
    col = -1;
    for (int j = 0; j < mask_matrix.cols; ++j)
    {
        if (mask_matrix.at<char>(row, j) == 1)
        {
            col = j;
            return;
        }
    }
}

[[nodiscard]] bool HungarianAlgorithm::star_in_col
(
    int col,
    const cv::Mat& mask_matrix
) const noexcept
{
    for (int i = 0; i < mask_matrix.rows; ++i)
    {
        if (mask_matrix.at<char>(i, col) == 1)
        {
            return true;
        }
    }
    return false;
}

void HungarianAlgorithm::find_star_in_col
(
    int col,
    int& row,
    const cv::Mat& mask_matrix
) const noexcept
{
    row = -1;
    for (int i = 0; i < mask_matrix.rows; ++i)
    {
        if (mask_matrix.at<char>(i, col) == 1)
        {
            row = i;
            return;
        }
    }
}

[[nodiscard]] bool HungarianAlgorithm::prime_in_row
(
    int row,
    const cv::Mat& mask_matrix
) const noexcept
{
    for (int j = 0; j < mask_matrix.cols; ++j)
    {
        if (mask_matrix.at<char>(row, j) == 2)
            return true;
    }
    return false;
}

void HungarianAlgorithm::find_prime_in_row
(
    int row,
    int& col,
    const cv::Mat& mask_matrix
) const noexcept
{
    col = -1;
    for (int j = 0; j < mask_matrix.cols; ++j)
    {
        if (mask_matrix.at<char>(row, j) == 2)
        {
            col = j;
            return;
        }
    }
}