#pragma once

#include <PoseLib/poselib.h>

#include "io.h"

#include <utility>
#include <vector>

namespace loransac_app {

struct GtError {
    double rotation_deg;
    double translation_deg;
};

poselib::Camera make_simple_pinhole_camera(double f, double cx, double cy);
poselib::Camera make_pinhole_camera(double fx, double fy, double cx, double cy);
poselib::Camera make_camera_from_intrinsics(double fx, double fy, double cx, double cy,
                                            const std::vector<double> &distortion);
std::pair<poselib::Camera, poselib::Camera> make_cameras_from_pose_intrinsics(const PoseIntrinsics &intr);
// Ground-truth poses are always consumed exactly as stored in pose_intrinsics.csv.
poselib::CameraPose make_gt_pose(const PoseIntrinsics &intr);
GtError compute_gt_error(const poselib::CameraPose &estimated, const PoseIntrinsics &intr);

} // namespace loransac_app
