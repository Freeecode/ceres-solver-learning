/**
 * @brief
 * 
 * Author:
 * Date:
 */

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "reprojection_factor.h"
#include "matcher.h"

void PoseOptimization(const std::vector<cv::Point3d>& points_3d,
                      const std::vector<cv::Point2d>& points_2d,
                      cv::Mat &rvec, cv::Mat &t);

int main(int argc, char* argv[])
{
    if(argc < 4)
    {
        std::cout << "Usage:\n"
                  << "  ceres_pnp img1 img2 depth1" << std::endl;
        exit(-1);
    }

    cv::Mat img1 =  cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat img2 =  cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat depth1 =  cv::imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);

    Matcher match(img1, img2, depth1);
    //std::cout << "Good matches: " << match.good_matches.size() << std::endl;
    std::cout << "3D-2D pairs: " << match.pts_3d.size() << std::endl;
    
    cv::Mat K = match.getCameraK();
    //std::cout << K.at<float>(0,0) << std::endl;
    
    cv::Mat rvec, t, R;
    cv::solvePnP(match.pts_3d, match.pts_2d, K, cv::Mat(), rvec, t, false, cv::SOLVEPNP_EPNP);

    cv::Rodrigues(rvec, R);
    std::cout << "rvec= \n" << rvec << std::endl;
    std::cout << "R= \n"    << R << std::endl;
    std::cout << "t= \n"    << t << std::endl;

    std::cout << "Calling bundle adjustment..." << std::endl;

    PoseOptimization(match.pts_3d, match.pts_2d, rvec, t);

    return 0;
}


void PoseOptimization(const std::vector<cv::Point3d>& points_3d,
                      const std::vector<cv::Point2d>& points_2d,
                      cv::Mat &rvec, cv::Mat &t)
{
    // Attention: cv::Mat::type
    assert(rvec.type() == CV_64F);
    assert(t.type() == CV_64F);

    double camera_rvec[3];
    camera_rvec[0] = rvec.at<double>(0,0); // can't use float
    camera_rvec[1] = rvec.at<double>(1,0);
    camera_rvec[2] = rvec.at<double>(2,0);
    
    double camera_t[3];
    camera_t[0] = t.at<double>(0,0);
    camera_t[1] = t.at<double>(1,0);
    camera_t[2] = t.at<double>(2,0);

    ceres::Problem problem;

    ceres::LossFunction* lossfunction = NULL;

    for(uint i = 0; i < points_3d.size(); i++)
    {
        Eigen::Vector3d p3d(points_3d[i].x, points_3d[i].y, points_3d[i].z);
        Eigen::Vector2d p2d(points_2d[i].x, points_2d[i].y);
        
        ceres::CostFunction* costfunction = ReprojectionError::Create(p3d, p2d);
        problem.AddResidualBlock(costfunction, lossfunction, camera_rvec, camera_t);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 100;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    //std::cout << camera_t[0] << "," << camera_t[1] << "," << camera_t[2] << std::endl;
    std::cout << "After Optimizing: " << std::endl;
    double quat[4];
    ceres::AngleAxisToQuaternion(camera_rvec, quat);
    Eigen::Quaterniond q(quat[0], quat[1], quat[2], quat[3]);
    //std::cout << "Rot= \n" << q.matrix() << std::endl;

    rvec.at<double>(0,0) = camera_rvec[0];
    rvec.at<double>(1,0) = camera_rvec[1];
    rvec.at<double>(2,0) = camera_rvec[2];
    t.at<double>(0,0) = camera_t[0];
    t.at<double>(1,0) = camera_t[1];
    t.at<double>(2,0) = camera_t[2];
    
    cv::Mat Rot;
    cv::Rodrigues(rvec, Rot);
    assert(Rot.type()== CV_64F);
    Eigen::Matrix3d Rotation;
    cv::cv2eigen(Rot, Rotation);
    Eigen::Isometry3d T(Rotation);
    T.pretranslate(Eigen::Vector3d(camera_t[0], camera_t[1], camera_t[2]));
    std::cout << "T= \n" << T.matrix() << std::endl;
}