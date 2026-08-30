#include "m2m_loransac/geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace loransac_app {
namespace {

double rad_to_deg(double radians) {
    return radians * 180.0 / M_PI;
}

double clamp_unit(double value) {
    return std::max(-1.0, std::min(1.0, value));
}

double rotation_error_deg(const poselib::CameraPose &estimated, const Eigen::Quaterniond &gt) {
    const Eigen::Matrix3d delta = estimated.R() * gt.toRotationMatrix().transpose();
    return rad_to_deg(std::acos(clamp_unit((delta.trace() - 1.0) * 0.5)));
}

double translation_error_deg(const Eigen::Vector3d &estimated, const Eigen::Vector3d &gt) {
    if (estimated.norm() == 0.0 || gt.norm() == 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return rad_to_deg(std::acos(clamp_unit(estimated.normalized().dot(gt.normalized()))));
}

double translation_error_sign_invariant_deg(const Eigen::Vector3d &estimated, const Eigen::Vector3d &gt) {
    const double err = translation_error_deg(estimated, gt);
    return std::min(err, 180.0 - err);
}

} // namespace

poselib::Camera make_simple_pinhole_camera(double f, double cx, double cy) {
    return poselib::Camera("SIMPLE_PINHOLE", {f, cx, cy}, 0, 0);
}

poselib::Camera make_pinhole_camera(double fx, double fy, double cx, double cy) {
    const double scale = std::max({std::abs(fx), std::abs(fy), 1.0});
    if (std::abs(fx - fy) <= 1e-9 * scale) {
        return make_simple_pinhole_camera(0.5 * (fx + fy), cx, cy);
    }
    return poselib::Camera("PINHOLE", {fx, fy, cx, cy}, 0, 0);
}

poselib::Camera make_camera_from_intrinsics(double fx, double fy, double cx, double cy,
                                            const std::vector<double> &distortion) {
    if (distortion.empty()) {
        return make_pinhole_camera(fx, fy, cx, cy);
    }
    if (distortion.size() == 4) {
        return poselib::Camera("OPENCV",
                                 {fx, fy, cx, cy, distortion[0], distortion[1], distortion[2], distortion[3]}, 0, 0);
    }
    if (distortion.size() == 8) {
        return poselib::Camera("FULL_OPENCV",
                                 {fx, fy, cx, cy, distortion[0], distortion[1], distortion[2], distortion[3],
                                  distortion[4], distortion[5], distortion[6], distortion[7]},
                                 0, 0);
    }
    throw std::runtime_error(
        "distortion must be empty, or length 4 (OPENCV), or 8 (FULL_OPENCV k1,k2,p1,p2,k3,k4,k5,k6)");
}

std::pair<poselib::Camera, poselib::Camera> make_cameras_from_pose_intrinsics(const PoseIntrinsics &intr) {
    return {make_camera_from_intrinsics(intr.fx1, intr.fy1, intr.cx1, intr.cy1, intr.distortion_cam1),
            make_camera_from_intrinsics(intr.fx2, intr.fy2, intr.cx2, intr.cy2, intr.distortion_cam2)};
}

poselib::CameraPose make_gt_pose(const PoseIntrinsics &intr) {
    return poselib::CameraPose(intr.q_gt.toRotationMatrix(), intr.t_gt);
}

GtError compute_gt_error(const poselib::CameraPose &estimated, const PoseIntrinsics &intr) {
    return {rotation_error_deg(estimated, intr.q_gt),
            translation_error_sign_invariant_deg(estimated.t, intr.t_gt)};
}

} // namespace loransac_app
