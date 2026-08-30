#pragma once

#include <PoseLib/poselib.h>

#include <Eigen/Geometry>

#include <filesystem>
#include <map>
#include <vector>

namespace loransac_app {

namespace fs = std::filesystem;

struct PoseIntrinsics {
    int pair_idx = -1;
    double fx1 = 0.0;
    double fy1 = 0.0;
    double cx1 = 0.0;
    double cy1 = 0.0;
    double fx2 = 0.0;
    double fy2 = 0.0;
    double cx2 = 0.0;
    double cy2 = 0.0;
    /** COLMAP OPENCV (4) or FULL_OPENCV (8) extra params; empty => undistorted pinhole. */
    std::vector<double> distortion_cam1;
    std::vector<double> distortion_cam2;
    Eigen::Quaterniond q_gt = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t_gt = Eigen::Vector3d::Zero();
};

struct MatchData {
    std::vector<poselib::Point2D> points1;
    std::vector<poselib::Point2D> points2;
    std::vector<int> idx1;
    std::vector<int> idx2;
    std::vector<double> prob;
    std::vector<double> similarity;
    std::vector<size_t> first_k;
    bool has_m2m_metadata = false;
    bool has_similarity = false;
    bool has_first_k = false;
};

std::map<int, PoseIntrinsics> load_pose_intrinsics(const fs::path &path);
void load_matches(const fs::path &path, std::vector<poselib::Point2D> *points1,
                  std::vector<poselib::Point2D> *points2);
MatchData load_match_data(const fs::path &path, bool require_first_k = false);
std::vector<double> assign_prob(const std::vector<int> &idx1, const std::vector<int> &idx2, double qx_ub,
                                double qy_ub);
int pair_index_from_path(const fs::path &path);

} // namespace loransac_app
