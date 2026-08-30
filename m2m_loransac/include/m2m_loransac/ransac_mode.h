#pragma once

// Algorithm-layer header (project-agnostic).
//
// RansacMode selects which scoring formulation the many-to-many RANSAC engine
// uses. It lives here -- not in the project's config.h -- so the reusable
// algorithm files (ransac_m2m_impl.h, relative_pose_m2m.*) never have to pull in
// the experiment harness's Config struct.

namespace loransac_app {

// CM     : PoseLib one-to-one classic LO-RANSAC (reference; not handled here).
// HCM     : many-to-many log (Hidden Cardinality Maximization) score.
// MCM     : many-to-many maximum-cardinality matching score.
// HCM_MC : HCM exploration + maximum-cardinality rerank of a bounded pool.
enum class RansacMode { CM, HCM, MCM, HCM_MC };

inline bool is_m2m_ransac(RansacMode mode) {
    return mode == RansacMode::HCM || mode == RansacMode::MCM || mode == RansacMode::HCM_MC;
}

inline const char *ransac_mode_name(RansacMode mode) {
    switch (mode) {
    case RansacMode::CM:
        return "CM";
    case RansacMode::HCM:
        return "HCM";
    case RansacMode::MCM:
        return "MCM";
    case RansacMode::HCM_MC:
        return "HCM_MC";
    }
    return "CM";
}

} // namespace loransac_app
