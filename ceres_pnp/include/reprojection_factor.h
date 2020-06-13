#ifndef CERES_PNP_H_
#define CERES_PNP_H_

#include <ceres/ceres.h>
#include <ceres/rotation.h>

class ReprojectionError
{
public:

    ReprojectionError(Eigen::Vector3d point_, Eigen::Vector2d observed_)
        : point(point_), observed(observed_)
    {
    }

    template<typename T>
    bool operator()(const T* const camera_r, const T* const camera_t, T* residuals) const
    {
        T pt1[3];
        pt1[0] = T(point.x());
        pt1[1] = T(point.y());
        pt1[2] = T(point.z());

        T pt2[3];
        ceres::AngleAxisRotatePoint(camera_r, pt1, pt2);

        pt2[0] = pt2[0] + camera_t[0];
        pt2[1] = pt2[1] + camera_t[1];
        pt2[2] = pt2[2] + camera_t[2];

        const T xp = T(K[0] * (pt2[0] / pt2[2]) + K[2]);
        const T yp = T(K[1] * (pt2[1] / pt2[2]) + K[3]);

        const T u = T(observed.x());
        const T v = T(observed.y());

        residuals[0] = u - xp;
        residuals[1] = v - yp;

        return true;
    }

    static ceres::CostFunction* Create(Eigen::Vector3d points, Eigen::Vector2d observed)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3>(
                        new ReprojectionError(points, observed)));
    }

private:
    Eigen::Vector3d point;
    Eigen::Vector2d observed;
    // Camera intrinsics
    double K[4] = {520.9, 521.0, 325.1, 249.7}; // fx,fy,cx,cy
};

#endif