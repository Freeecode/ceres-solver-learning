/**
 * 
 */

#ifndef CERES_ICP_H_
#define CERES_ICP_H_

#include <ceres/ceres.h>
#include <ceres/rotation.h>

class Reprojection3dError
{
public:

    Reprojection3dError(Eigen::Vector3d point_, Eigen::Vector3d observed_)
        : point(point_), observed(observed_)
    {
    }

    template<typename T>
    bool operator()(const T* const camera, T* residuals) const
    {
        T pt1[3];
        pt1[0] = T(point.x());
        pt1[1] = T(point.y());
        pt1[2] = T(point.z());

        T pt2[3];
        ceres::AngleAxisRotatePoint(camera, pt1, pt2);

        pt2[0] = pt2[0] + camera[3];
        pt2[1] = pt2[1] + camera[4];
        pt2[2] = pt2[2] + camera[5];

        residuals[0] = observed[0] - pt2[0];
        residuals[1] = observed[1] - pt2[1];
        residuals[2] = observed[2] - pt2[2];

        return true;
    }

    static ceres::CostFunction* Create(Eigen::Vector3d point, Eigen::Vector3d observed)
    {
        return (new ceres::AutoDiffCostFunction<Reprojection3dError, 3, 6>(
                        new Reprojection3dError(point, observed)));
    }

private:
    Eigen::Vector3d point;
    Eigen::Vector3d observed;
    // Camera intrinsics
    double K[4] = {520.9, 521.0, 325.1, 249.7}; // fx,fy,cx,cy
};

#endif

