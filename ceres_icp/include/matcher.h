/**
 * 
 */

#ifndef CERES_PNP_MATCHER_
#define CERES_PNP_MATCHER_

#include <opencv2/opencv.hpp>
#include <iostream>

#include<Eigen/Core>

class Matcher
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
    Matcher(cv::Mat& img1, cv::Mat& img2, cv::Mat& depth1, cv::Mat& depth2);

    void extractORB(cv::Mat& im, std::vector<cv::KeyPoint>& kpts,
                    cv::Mat& descriptors);

    void find_feature_matches();

    void construct3d_to_2d(cv::Mat& depth1, cv::Mat& depth2);

    cv::Mat getCameraK();

    cv::Point2d pixel2cam(const cv::Point2d& p);

    virtual ~Matcher();
public:

    std::vector<cv::KeyPoint> kpts1;
    std::vector<cv::KeyPoint> kpts2;
    cv::Mat descriptors1;
    cv::Mat descriptors2;

    std::vector<cv::DMatch> good_matches;
    std::vector<cv::DMatch> matches;

    std::vector<cv::Point3d> pts1_3d; // img1
    std::vector<cv::Point2d> pts1_2d; // img1

    std::vector<cv::Point3d> pts2_3d; // img2
    std::vector<cv::Point2d> pts2_2d; // img2

private:

    // Camera intrinsics
    const float fx = 520.9;
    const float fy = 521.0;
    const float cx = 325.1;
    const float cy = 249.7;

    const float depthfactor = 5000.0;

    // ORB
    cv::Ptr<cv::ORB> orb;

    // BFMatcher
    cv::Ptr<cv::BFMatcher> bfmatcher;
};

#endif