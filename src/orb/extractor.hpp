#ifndef EXTRACTOR_H
#define EXTRACTOR_H

#include "includes.ihh"
#include "extra.hpp"

namespace orb
{

class extractor_node
{
public:
    ///@brief default constructor
    extractor_node();

    ///@brief divide in 4 different child nodes
    void DivideNode(extractor_node &n1, 
                    extractor_node &n2, 
                    extractor_node &n3, 
                    extractor_node &n4);

    /// @brief check if every node has at least one keypoint
    void check_keypoints(std::list<extractor_node> & nodes);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<extractor_node>::iterator lit;
    bool bNoMore;
};

class extractor
{
public:
    
    ///@brief orb methods to detect features
    enum { 
        HARRIS_SCORE=0, 
        FAST_SCORE=1  //faster but more inaccurate 
    };


    /**
     * @brief Constructor 
     * @param nfeatures Number of maximum features to look for
     * @param scaleFactor Pyramid decimation ratio 
     * @param nlevels Number of levels for the pyramid 
     * @param iniThFAST
     * @param minThFAST
     */
    extractor(int nfeatures = 2000, 
              float scaleFactor = 1.2, 
              int nlevels = 8,
              int iniThFAST = 20, 
              int minThFAST = 7);

    /// @brief  Compute the ORB features and descriptors on an image.
    /// ORB are dispersed on the image using an octree.
    /// Mask is ignored in the current implementation.
    void operator()( cv::InputArray image, 
                     std::vector<cv::KeyPoint>& keypoints,
                     cv::OutputArray descriptors);

    int get_levels();

    float get_scalefactor();

    std::vector<float> get_scalefactors();

    std::vector<float> get_inv_scalefactors();

    std::vector<float> get_sigma2();

    std::vector<float> get_inv_sigma2();

    ///@brief Vector of pyramid images
    std::vector<cv::Mat> image_pyramid;

private: 
    //@brief Init the scale factor variables 
    void init_scale_factors();

    //@brief Init number of features per level
    void init_features_level();

    //@brief Init vector umax
    void init_umax();

    //@brief create a pyramid of nlevels with the same image resize by inv_scalefactor^number_of_level
    void compute_pyramid(cv::Mat image);

    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);    

    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, 
                                                const int &minX,
                                                const int &maxX, 
                                                const int &minY, 
                                                const int &maxY, 
                                                const int &nFeatures, 
                                                const int &level);

    std::vector<cv::Point> pattern_;

    ///TODO: reduce the number of attributes
    int nfeatures_;
    double scalefactor_;
    int nlevels_;
    int iniThFAST_;
    int minThFAST_;

    std::vector<int> features_per_level_;

    std::vector<int> umax_;

    std::vector<float> vec_scalefactor_;
    std::vector<float> vec_inv_scalefactor_;    
    std::vector<float> vec_sigma2_;
    std::vector<float> vec_inv_sigma2_;


};

} //namespace orb

#endif

