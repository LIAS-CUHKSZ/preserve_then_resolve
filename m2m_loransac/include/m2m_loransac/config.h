#pragma once

#include "ransac_mode.h"

#include <filesystem>
#include <string>
#include <vector>

namespace loransac_app {

namespace fs = std::filesystem;

struct Config {
    fs::path matching_result_root = ".";
    std::vector<std::string> datasets;
    std::string method;
    fs::path output_csv = "loransac_results.csv";
    double max_error_px = 1.0;
    size_t min_iterations = 1000;
    size_t max_iterations = 100000;
    double similarity_threshold = 0.0;
    size_t max_matching_num = 0;
    // Zero keeps the historical full-graph sampler. A positive value restricts
    // only minimal proposals to CSV rows with k_first (or legacy k) <= this
    // value; scoring and refinement still use every retained association.
    size_t proposal_max_k = 0;
    double success_prob = 0.9999;
    unsigned long seed = 0;
    size_t ransac_times = 1;
    bool tangent_sampson = false;
    bool init_with_gt = false;
    bool skip_existing_pairs = true;
    bool allow_unbound_pose = false;
    RansacMode ransac_mode = RansacMode::CM;
    size_t top_n_candidates = 100;
    double pool_dedup_deg = 0.0; // >0 enables HCM_MC pool deduplication at this angle (degrees)
    bool write_candidate_traces = false; // HCM_MC only; writes one candidate CSV per image pair
    // HCM_MC only. Scores the stored benchmark pose after the normal run and
    // writes a separate read-only Stage-1 pool-admission sidecar.
    bool write_gt_hypothesis_probe = false;
    double m2m_delta = 0.01;
    double q_ub = 0.3;
};

std::string trim(const std::string &s);
fs::path resolve_path(const fs::path &base_dir, const fs::path &path);
Config load_config(const fs::path &config_path);

} // namespace loransac_app
