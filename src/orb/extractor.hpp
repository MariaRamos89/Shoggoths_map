#ifndef EXTRACTOR_H
#define EXTRACTOR_H

#include "includes.ihh"
#include "extra.hpp"

namespace orb
{

//class ExtractorNode
//{
//public:
//    ExtractorNode():bNoMore(false){}
//
//    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);
//
//    std::vector<cv::KeyPoint> vKeys;
//    cv::Point2i UL, UR, BL, BR;
//    std::list<ExtractorNode>::iterator lit;
//    bool bNoMore;
//};

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
    extractor(int nfeatures, 
              float scaleFactor, 
              int nlevels,
              int iniThFAST, 
              int minThFAST);

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    void operator()( cv::InputArray image, cv::InputArray mask,
      std::vector<cv::KeyPoint>& keypoints,
      cv::OutputArray descriptors);

    int inline GetLevels(){
        return nlevels_;}

    float inline GetScaleFactor(){
        return scalefactor_;}

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;

protected:

    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);    
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                           const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

    std::vector<cv::Point> pattern;

    int nfeatures_;
    double scalefactor_;
    int nlevels_;
    int iniThFAST_;
    int minThFAST_;

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;    
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
};

} //namespace orb

#endif

