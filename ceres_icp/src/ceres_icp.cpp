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

#include "reprojection3d.h"
#include "matcher.h"

void PoseOptimization(const std::vector<cv::Point3d>& pts1_3d, const std::vector<cv::Point2d>& pts1_2d,
                      const std::vector<cv::Point3d>& pts2_3d, const std::vector<cv::Point2d>& pts2_2d,
                      Eigen::Matrix4d &T);

int main(int argc, char* argv[])
{
    if(argc < 5)
    {
        std::cout << "Usage:\n"
                  << "  ceres_pnp img1 img2 depth1 depth2" << std::endl;
        exit(-1);
    }

    cv::Mat img1 =  cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat img2 =  cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat depth1 =  cv::imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat depth2 =  cv::imread(argv[4], CV_LOAD_IMAGE_UNCHANGED);

    Matcher match(img1, img2, depth1, depth2);
    std::cout << "Good matches: " << match.good_matches.size() << std::endl;
    std::cout << "3D-2D pairs: " << match.pts1_3d.size() << std::endl;
    
    cv::Mat K = match.getCameraK();
    //std::cout << K.at<float>(0,0) << std::endl;
    /*
    cv::Mat rvec, t, R;
    cv::solvePnP(match.pts_3d, match.pts_2d, K, cv::Mat(), rvec, t, false, cv::SOLVEPNP_EPNP);
    cv::Rodrigues(rvec, R);
    std::cout << "rvec= \n" << rvec << std::endl;
    std::cout << "R= \n"    << R << std::endl;
    std::cout << "t= \n"    << t << std::endl;
    */
    std::cout << "Calling bundle adjustment..." << std::endl;

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    PoseOptimization(match.pts1_3d, match.pts1_2d, match.pts2_3d, match.pts2_2d, T);

    return 0;
}

void PoseOptimization(const std::vector<cv::Point3d>& pts1_3d, const std::vector<cv::Point2d>& pts1_2d,
                      const std::vector<cv::Point3d>& pts2_3d, const std::vector<cv::Point2d>& pts2_2d,
                      Eigen::Matrix4d &T)
{
    double camera[6] = {0, 1, 2, 0, 0, 0};
    /*
    Eigen::Matrix3d Rot = T.block(0,0,3,3);
    //std::cout << Rot << std::endl;
    Eigen::Quaterniond q(Rot);
    double quat[4] = {q.w(), q.x(), q.y(), q.z()};
    */
    ceres::Problem problem;
    ceres::LossFunction* lossfunction = NULL;
    for(uint i = 0; i < pts1_3d.size(); i++)
    {
        Eigen::Vector3d p1(pts1_3d[i].x, pts1_3d[i].y, pts1_3d[i].z);
        Eigen::Vector3d p2(pts2_3d[i].x, pts2_3d[i].y, pts2_3d[i].z);
        Eigen::Vector2d pixel(pts2_2d[i].x, pts2_2d[i].y);
        // P2:observed
        ceres::CostFunction* costfunction = Reprojection3dError::Create(p1, p2);
        problem.AddResidualBlock(costfunction, lossfunction, camera);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 100;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "After Optimizing: "  << std::endl;
    
    double quat[4];
    ceres::AngleAxisToQuaternion(camera, quat);
    Eigen::Quaterniond q(quat[0], quat[1], quat[2], quat[3]);
    Eigen::Isometry3d Transform(q.matrix());
    Transform.pretranslate(Eigen::Vector3d(camera[3], camera[4], camera[5]));
    T = Transform.matrix();
    std::cout << "T=\n" << T << std::endl;
}