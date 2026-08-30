#pragma once

#include <PoseLib/poselib.h>

#include "config.h"

#include <filesystem>
#include <string>

namespace loransac_app {

namespace fs = std::filesystem;

struct DatasetStats {
    size_t processed = 0;
    size_t skipped = 0;
    size_t already_done = 0;
};

DatasetStats run_dataset(const fs::path &matching_result_root, const std::string &dataset, double q_ub,
                         const Config &cfg, const poselib::RelativePoseOptions &opt);
poselib::RelativePoseOptions make_relative_pose_options(const Config &cfg);

} // namespace loransac_app
