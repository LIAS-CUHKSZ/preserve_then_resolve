// Toy example: run many-to-many LO-RANSAC relative pose from a single
// association file plus hardwired intrinsics.
//
// This program deliberately depends ONLY on the reusable algorithm layer
// (estimate_relative_pose_m2m + its headers, PoseLib, Eigen). It does NOT touch
// the experiment harness (config.cpp / io.cpp / runner.cpp / geometry.cpp), so
// it doubles as a minimal, copy-pasteable template for using the algorithm in
// your own code.
//
// Usage:
//   ./toy_m2m [associations.csv] [ransac_mode] [q_ub]
//     associations.csv : defaults to examples/toy_matches.csv
//     ransac_mode      : HCM | MCM | HCM_MC   (default HCM_MC)
//     q_ub             : Assumption-1 prior upper bound, default 0.30
//
// The association CSV must contain, by header name:
//   idx1 (or left_idx / feature_idx1), idx2 (or right_idx / feature_idx2),
//   x1, y1, x2, y2
// and may optionally contain a `prob` column. When `prob` is absent it is
// derived from q_ub with the same Assumption-1 prior the project uses.

#include "m2m_loransac/relative_pose_m2m.h"

#include <PoseLib/poselib.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// ----------------------------------------------------------------------------
// EDIT ME: hardwired pinhole intrinsics for the two images.
// fx, fy = focal lengths (px); cx, cy = principal point (px).
// The shipped fixture was projected with identical pinhole cameras, identity
// rotation, and translation along the x axis. Every left feature has one true
// and one ambiguous candidate, making it a small genuine many-to-many graph.
// ----------------------------------------------------------------------------
constexpr double kFx1 = 800.0, kFy1 = 800.0, kCx1 = 320.0, kCy1 = 240.0;
constexpr double kFx2 = 800.0, kFy2 = 800.0, kCx2 = 320.0, kCy2 = 240.0;

// ----------------------------------------------------------------------------
// Optional ground-truth relative pose, used only to report estimation error.
// Set kHasGroundTruth = false if you don't have a reference pose.
// Translation is compared by direction (it is only recoverable up to scale).
// ----------------------------------------------------------------------------
constexpr bool kHasGroundTruth = true;
constexpr double kGtQw = 1.0, kGtQx = 0.0;
constexpr double kGtQy = 0.0, kGtQz = 0.0;
constexpr double kGtTx = 1.0, kGtTy = 0.0, kGtTz = 0.0;

// RANSAC / scoring parameters (mirror the project defaults).
constexpr double kMaxErrorPx = 1.0;     // inlier threshold in pixels
constexpr double kM2MDelta = 0.01;      // HCM score delta
constexpr size_t kTopNCandidates = 100; // HCM_MC bounded candidate pool size
constexpr size_t kMinIterations = 1000;
constexpr size_t kMaxIterations = 100000;
constexpr double kSuccessProb = 0.9999;
constexpr unsigned long kSeed = 0;

std::string trim(const std::string &s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
        return "";
    }
    const size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

std::vector<std::string> split_csv(const std::string &line) {
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        cols.push_back(trim(item));
    }
    return cols;
}

int find_col(const std::unordered_map<std::string, int> &cols, std::initializer_list<const char *> names) {
    for (const char *name : names) {
        const auto it = cols.find(name);
        if (it != cols.end()) {
            return it->second;
        }
    }
    return -1;
}

struct Associations {
    std::vector<poselib::Point2D> points1;
    std::vector<poselib::Point2D> points2;
    std::vector<int> idx1;
    std::vector<int> idx2;
    std::vector<double> prob; // empty unless the CSV had a prob column
};

Associations load_associations(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open association CSV: " + path);
    }

    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("Empty association CSV: " + path);
    }
    std::unordered_map<std::string, int> cols;
    {
        const auto header = split_csv(line);
        for (int i = 0; i < static_cast<int>(header.size()); ++i) {
            cols[header[i]] = i;
        }
    }
    const int c_idx1 = find_col(cols, {"idx1", "left_idx", "feature_idx1", "feature1"});
    const int c_idx2 = find_col(cols, {"idx2", "right_idx", "feature_idx2", "feature2"});
    const int c_x1 = find_col(cols, {"x1"});
    const int c_y1 = find_col(cols, {"y1"});
    const int c_x2 = find_col(cols, {"x2"});
    const int c_y2 = find_col(cols, {"y2"});
    const int c_prob = find_col(cols, {"prob", "probability"});
    if (c_idx1 < 0 || c_idx2 < 0 || c_x1 < 0 || c_y1 < 0 || c_x2 < 0 || c_y2 < 0) {
        throw std::runtime_error("CSV must have columns idx1,idx2,x1,y1,x2,y2 (or left_idx/right_idx aliases)");
    }

    Associations a;
    while (std::getline(in, line)) {
        if (trim(line).empty()) {
            continue;
        }
        const auto c = split_csv(line);
        const int max_col = std::max({c_idx1, c_idx2, c_x1, c_y1, c_x2, c_y2, c_prob});
        if (static_cast<int>(c.size()) <= max_col) {
            continue;
        }
        try {
            const int idx1 = std::stoi(c[c_idx1]);
            const int idx2 = std::stoi(c[c_idx2]);
            const poselib::Point2D point1(std::stod(c[c_x1]), std::stod(c[c_y1]));
            const poselib::Point2D point2(std::stod(c[c_x2]), std::stod(c[c_y2]));
            double probability = 0.0;
            if (c_prob >= 0) {
                probability = std::stod(c[c_prob]);
            }
            a.idx1.push_back(idx1);
            a.idx2.push_back(idx2);
            a.points1.push_back(point1);
            a.points2.push_back(point2);
            if (c_prob >= 0) {
                a.prob.push_back(probability);
            }
        } catch (const std::exception &) {
            // Drop malformed rows, matching the project's tolerant CSV reader.
            continue;
        }
    }
    return a;
}

// Assumption-1 prior: split each feature's q_ub budget evenly across the edges
// incident to it, then average the two directions. Identical to the project's
// assign_prob(); inlined here to keep the example self-contained.
std::vector<double> assign_prob(const std::vector<int> &idx1, const std::vector<int> &idx2, double q_ub) {
    std::unordered_map<int, int> deg1, deg2;
    for (size_t i = 0; i < idx1.size(); ++i) {
        deg1[idx1[i]]++;
        deg2[idx2[i]]++;
    }
    std::vector<double> prob(idx1.size());
    for (size_t i = 0; i < idx1.size(); ++i) {
        prob[i] = 0.5 * (q_ub / deg1[idx1[i]] + q_ub / deg2[idx2[i]]);
    }
    return prob;
}

// Angular errors against ground truth, matching geometry.cpp's convention:
// rotation via the relative rotation's trace, translation as the sign-invariant
// angle between translation directions (translation is recoverable up to scale).
struct PoseError {
    double rotation_deg = 0.0;
    double translation_deg = 0.0;
};

double clamp_unit(double v) { return std::max(-1.0, std::min(1.0, v)); }

PoseError pose_error(const poselib::CameraPose &est, const Eigen::Quaterniond &q_gt, const Eigen::Vector3d &t_gt) {
    constexpr double kRadToDeg = 180.0 / M_PI;
    const Eigen::Matrix3d delta = est.R() * q_gt.toRotationMatrix().transpose();
    const double rot = std::acos(clamp_unit((delta.trace() - 1.0) * 0.5)) * kRadToDeg;

    double trans = std::numeric_limits<double>::quiet_NaN();
    if (est.t.norm() > 0.0 && t_gt.norm() > 0.0) {
        const double a = std::acos(clamp_unit(est.t.normalized().dot(t_gt.normalized()))) * kRadToDeg;
        trans = std::min(a, 180.0 - a); // sign-invariant: t and -t are equivalent here
    }
    return {rot, trans};
}

loransac_app::RansacMode parse_mode(const std::string &s) {
    if (s == "HCM") {
        return loransac_app::RansacMode::HCM;
    }
    if (s == "MCM") {
        return loransac_app::RansacMode::MCM;
    }
    if (s == "HCM_MC") {
        return loransac_app::RansacMode::HCM_MC;
    }
    throw std::runtime_error("ransac_mode must be one of: HCM, MCM, HCM_MC (got '" + s + "')");
}

} // namespace

int main(int argc, char **argv) {
    try {
        const std::string csv_path = argc > 1 ? argv[1] : "examples/toy_matches.csv";
        const loransac_app::RansacMode mode = parse_mode(argc > 2 ? argv[2] : "HCM_MC");
        const double q_ub = argc > 3 ? std::stod(argv[3]) : 0.30;

        Associations a = load_associations(csv_path);
        std::cout << "Loaded " << a.points1.size() << " associations from " << csv_path << "\n";
        if (a.points1.size() < 5) {
            std::cerr << "Need at least 5 associations to estimate a relative pose.\n";
            return 1;
        }
        if (a.prob.empty()) {
            a.prob = assign_prob(a.idx1, a.idx2, q_ub);
            std::cout << "Derived per-association prob from q_ub=" << q_ub << " (no prob column present)\n";
        }

        // Hardwired pinhole cameras (edit kFx1.. above for your own data).
        const poselib::Camera camera1("PINHOLE", {kFx1, kFy1, kCx1, kCy1}, 0, 0);
        const poselib::Camera camera2("PINHOLE", {kFx2, kFy2, kCx2, kCy2}, 0, 0);

        poselib::RelativePoseOptions opt;
        opt.max_error = kMaxErrorPx;
        opt.bundle.loss_scale = kMaxErrorPx;
        opt.ransac.min_iterations = kMinIterations;
        opt.ransac.max_iterations = kMaxIterations;
        opt.ransac.success_prob = kSuccessProb;
        opt.ransac.seed = kSeed;

        poselib::CameraPose pose;
        std::vector<char> inliers;
        loransac_app::M2MTimingStats timing;
        const auto t_start = std::chrono::steady_clock::now();
        const poselib::RansacStats stats = loransac_app::estimate_relative_pose_m2m(
            a.points1, a.points2, camera1, camera2, a.idx1, a.idx2, a.prob, kM2MDelta, mode, kTopNCandidates, opt,
            &pose, &inliers, &timing);
        const double elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

        std::cout << "\n=== " << loransac_app::ransac_mode_name(mode) << " result ===\n";
        std::cout << "iterations    : " << stats.iterations << "\n";
        std::cout << "refinements   : " << stats.refinements << "\n";
        std::cout << "edge inliers  : " << stats.num_inliers << " / " << a.points1.size() << "\n";
        std::cout << "inlier ratio  : " << stats.inlier_ratio << "\n";
        std::cout << "model score   : " << stats.model_score << "\n";
        std::cout << "rotation q (w,x,y,z): " << pose.q(0) << ", " << pose.q(1) << ", " << pose.q(2) << ", "
                  << pose.q(3) << "\n";
        std::cout << "translation t (x,y,z): " << pose.t(0) << ", " << pose.t(1) << ", " << pose.t(2) << "\n";

        if (kHasGroundTruth) {
            const Eigen::Quaterniond q_gt = Eigen::Quaterniond(kGtQw, kGtQx, kGtQy, kGtQz).normalized();
            const Eigen::Vector3d t_gt(kGtTx, kGtTy, kGtTz);
            const PoseError err = pose_error(pose, q_gt, t_gt);
            std::cout << "rotation error    : " << err.rotation_deg << " deg\n";
            std::cout << "translation error : " << err.translation_deg << " deg\n";
        }

        std::cout << "total time        : " << elapsed_s * 1e3 << " ms\n";
        const double per_iter_us = stats.iterations > 0 ? elapsed_s / stats.iterations * 1e6 : 0.0;
        std::cout << "avg time / iter   : " << per_iter_us << " us\n";

        const auto ms = [](uint64_t ns) { return static_cast<double>(ns) / 1e6; };
        std::cout << "--- timing decomposition (ransac) ---\n";
        std::cout << "solve      : " << ms(timing.solve_ns) << " ms\n";
        std::cout << "score      : " << ms(timing.score_ns) << " ms over " << timing.score_calls << " evals ("
                  << (timing.score_calls > 0
                          ? static_cast<double>(timing.score_ns) / 1e3 / static_cast<double>(timing.score_calls)
                          : 0.0)
                  << " us/eval)\n";
        std::cout << "refine     : " << ms(timing.refine_ns) << " ms over " << timing.refine_calls << " calls\n";
        std::cout << "rank (MCM) : " << ms(timing.rank_score_ns) << " ms over " << timing.rank_score_calls
                  << " evals\n";
        std::cout << "ransac total: " << ms(timing.total_ns) << " ms\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
