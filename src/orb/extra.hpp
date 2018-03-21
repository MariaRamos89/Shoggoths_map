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

}

#endif
