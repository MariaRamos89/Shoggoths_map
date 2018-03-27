#include <iostream>
#include "src/orb/extractor.hpp"
#include <chrono>

using namespace std;
using namespace cv;

int main()
{

    Mat gray_image1;
    gray_image1 = imread("../lenna.png", IMREAD_GRAYSCALE);

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    // Initiate ORB detector
    Ptr<FeatureDetector> detector = ORB::create(2000, 1.2, 8, 31, 0, 2, 1, 31, 12);
    vector<KeyPoint> keypoints_object;
    Mat descriptors_object;


    // find the keypoints and descriptors with ORB
    detector->detect(gray_image1, keypoints_object);

    Ptr<DescriptorExtractor> extractor = ORB::create(2000, 1.2, 8, 31, 0, 2, 1, 31, 12);
    extractor->compute(gray_image1, keypoints_object, descriptors_object );

    // Flann needs the descriptors to be of type CV_32F
    descriptors_object.convertTo(descriptors_object, CV_32F);

    //FlannBasedMatcher matcher;
    //vector<DMatch> matches;
    //matcher.match( descriptors_object, descriptors_scene, matches );

    //double max_dist = 0; double min_dist = 100;

    ////-- Quick calculation of max and min distances between keypoints
    //for( int i = 0; i < descriptors_object.rows; i++ )
    //{
    //    double dist = matches[i].distance;
    //    if( dist < min_dist ) min_dist = dist;
    //    if( dist > max_dist ) max_dist = dist;
    //}

    ////-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )
    //vector< DMatch > good_matches;

    //for( int i = 0; i < descriptors_object.rows; i++ )
    //{
    //    if( matches[i].distance < 3*min_dist )
    //    {
    //        good_matches.push_back( matches[i]);
    //    }
    //}


    //vector< Point2f > obj;

    //for( int i = 0; i < good_matches.size(); i++ )
    //{
    //    //-- Get the keypoints from the good matches
    //    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    //}

    end = std::chrono::system_clock::now();
    int elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>
                                     (end-start).count();

    cout << "Time: " << elapsed_seconds << " number of keypoints: " << keypoints_object.size() << endl;

    keypoints_object.clear();

    start = std::chrono::system_clock::now();
    orb::extractor Orb_slam;
    //Orb_slam.nfeatures = 2000;
    //Orb_slam.scaleFactor = 1.2;
    //Orb_slam.nlevels = 8;
    //Orb_slam.iniThFAST = 7;
    //Orb_slam.minThFAST = 12;
    (Orb_slam)(gray_image1, keypoints_object, descriptors_object);
    end = std::chrono::system_clock::now();
    elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>
                                     (end-start).count();


    cout << "Time: " << elapsed_seconds << " number of keypoints: " << keypoints_object.size() << endl;
    return 0;
}/*int main(int argc, char **argv)
{
    //cv::mat image = cv::imread("example.png");
    //orb::extractor ext;
    //ext.computepyramid(image);

    return 0;
}*/
