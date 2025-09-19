#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "../../include/tracking/hungarian.hpp"

TEST(HungarianTest, SquareMatrixAssignment)
{
    cv::Mat cost_matrix = (cv::Mat_<float>(3, 3) << 
        1.0f, 2.0f, 3.0f,
        2.0f, 4.0f, 6.0f,
        3.0f, 6.0f, 9.0f
    );

    HungarianAlgorithm hungarian;
    std::vector<int> assignment = hungarian.solve(cost_matrix);

    EXPECT_EQ(assignment.size(), 3);
    EXPECT_EQ(assignment[0], 2);
    EXPECT_EQ(assignment[1], 1);
    EXPECT_EQ(assignment[2], 0);
}

TEST(HungarianTest, RectangularMatrixAssignment)
{
    cv::Mat cost_matrix = (cv::Mat_<float>(2, 3) <<
        1.0f, 2.0f, 3.0f,
        4.0f, 1.0f, 6.0f
    );

    HungarianAlgorithm hungarian;
    std::vector<int> assignment = hungarian.solve(cost_matrix);

    EXPECT_EQ(assignment.size(), 2);
    EXPECT_EQ(assignment[0], 0);
    EXPECT_EQ(assignment[1], 1);
}

TEST(HungarianTest, OptimalAssignment)
{
    cv::Mat cost_matrix = (cv::Mat_<float>(3, 3) <<
        1.0f, 3.0f, 2.0f,
        2.0f, 1.0f, 3.0f,
        3.0f, 2.0f, 1.0f
    );

    HungarianAlgorithm hungarian;
    std::vector<int> assignment = hungarian.solve(cost_matrix);

    EXPECT_EQ(assignment.size(), 3);
    EXPECT_EQ(assignment[0], 0);
    EXPECT_EQ(assignment[1], 1);
    EXPECT_EQ(assignment[2], 2);

    float total_cost = 0.0f;
    for (int i = 0; i < assignment.size(); ++i)
    {
        total_cost += cost_matrix.at<float>(i, assignment[i]);
    }

    EXPECT_FLOAT_EQ(total_cost, 3.0f);
}

TEST(HungarianTest, EmptyMatrix)
{
    cv::Mat cost_matrix;

    HungarianAlgorithm hungarian;
    std::vector<int> assignment = hungarian.solve(cost_matrix);

    EXPECT_TRUE(assignment.empty());
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}