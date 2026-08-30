#pragma once

#include "ransac_m2m_impl.h"
#include "ransac_mode.h"
#include "PoseLib/camera_pose.h"
#include "PoseLib/misc/camera_models.h"
#include "PoseLib/types.h"
#include <vector>

namespace loransac_app {

// Optional diagnostics for the bounded raw-seed pool used by HCM_MC. Scores
// follow the implementation convention (smaller is better). The polished pose
// applies the same nominal-threshold final refit as the returned model, making
// every candidate evaluable under alternative offline tie-breaking rules.
struct M2MCandidateTrace {
    size_t stored_hcm_rank = 0;
    poselib::CameraPose seed_pose;
    poselib::CameraPose refined_pose;
    poselib::CameraPose polished_pose;
    double stored_hcm_score = 0.0;
    double seed_hcm_left = 0.0;
    double seed_hcm_right = 0.0;
    double refined_hcm_score = 0.0;
    double refined_hcm_left = 0.0;
    double refined_hcm_right = 0.0;
    double refined_msac_cost = 0.0;
    size_t edge_inliers = 0;
    size_t cardinality = 0;
    uint64_t stage2_refine_ns = 0;
    uint64_t mcm_score_ns = 0;
};

// Optional, read-only diagnostic for asking whether the benchmark reference
// pose would enter the frozen Stage-1 HCM raw-seed pool if it were scored as
// one additional hypothesis.  Smaller HCM scores are better and ties do not
// enter, matching CandidatePool::would_accept().  The reference pose is scored
// only after the normal RANSAC, Stage-2 selection, and final polish complete.
struct M2MGroundTruthHypothesisProbe {
    bool evaluated = false;
    size_t pool_capacity = 0;
    size_t pool_size = 0;
    bool pool_full = false;
    double gt_hcm_score = 0.0;
    double pool_cutoff_hcm_score = 0.0;
    size_t gt_edge_inliers = 0;
    bool would_enter_pool = false;
};

// proposal_indices optionally names full-array edges eligible for minimal
// five-point sampling. When supplied, min_iterations must equal max_iterations.
// HCM/MCM scoring, LO/refinement, HCM_MC re-ranking, final polish, and the
// returned inlier mask always use the complete input arrays.
poselib::RansacStats estimate_relative_pose_m2m(const std::vector<poselib::Point2D> &points2D_1,
                                                const std::vector<poselib::Point2D> &points2D_2,
                                                const poselib::Camera &camera1,
                                                const poselib::Camera &camera2,
                                                const std::vector<int> &idx1_vec,
                                                const std::vector<int> &idx2_vec,
                                                const std::vector<double> &prob_vec, double delta,
                                                RansacMode mode, size_t top_n_candidates,
                                                const poselib::RelativePoseOptions &opt,
                                                poselib::CameraPose *relative_pose, std::vector<char> *inliers,
                                                M2MTimingStats *timing = nullptr, double pool_dedup_deg = 0.0,
                                                std::vector<M2MCandidateTrace> *candidate_traces = nullptr,
                                                const std::vector<size_t> *proposal_indices = nullptr,
                                                const poselib::CameraPose *ground_truth_probe_pose = nullptr,
                                                M2MGroundTruthHypothesisProbe *ground_truth_probe = nullptr);

} // namespace loransac_app
