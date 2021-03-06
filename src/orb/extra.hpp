#ifndef EXTRA_H
#define EXTRA_H

#include "includes.ihh"

namespace orb
{

//Constants
enum constants 
{
    PATCH_SIZE = 31,
    HALF_PATCH_SIZE = 15,
    EDGE_THRESHOLD = 19
};

/**
 * @struct ic_angle
 * @brief Calculate the Intensity centroid angle of a keypoint
 */
struct ic_angle
{
    ///@brief operator to calculate intensity centroid angle of a keypoint
    float operator()(const cv::Mat& image,
                     const cv::Point2f pt,
                     const std::vector<int> & u_max); 
};

/**
 * @struct compute_descriptor
 * @brief Compute the descriptor of a keypoint
 */
struct compute_descriptor
{
    ///@brief operator to compute the descriptor of a keypoint
    void operator()(const cv::KeyPoint& kpt,
                    const cv::Mat& img,
                    const cv::Point* pattern,
                    uchar* desc);
};

/**
 * @struct compute_all_descriptor
 * @brief Compute the all descriptors 
 */
struct compute_all_descriptors
{
    ///@brief operator to compute the descriptor of a keypoint
    void operator()(const cv::Mat& image, 
                    std::vector<cv::KeyPoint>& keypoints, 
                    cv::Mat& descriptors,
                    const std::vector<cv::Point>& pattern);

};
/**
 * @struct compute_orientation
 * @brief Compute the orientation of a keypoint using @see ic_angle
 */
struct compute_orientation
{
    void operator()(const cv::Mat& image, 
                    std::vector<cv::KeyPoint>& keypoints, 
                    const std::vector<int>& umax);
};
}

#endif
