#include "m2m_loransac/config.h"
#include "m2m_loransac/runner.h"

#include <filesystem>
#include <iostream>

int main(int argc, char **argv) {
    try {
        namespace fs = std::filesystem;
        using namespace loransac_app;

        fs::path config_path = argc > 1 ? fs::path(argv[1]) : fs::path("config.txt");
        if (!fs::exists(config_path) && argc == 1 && fs::exists("../config.txt")) {
            config_path = "../config.txt";
        }
        config_path = fs::absolute(config_path);
        const fs::path config_dir = config_path.parent_path();
        Config cfg = load_config(config_path);

        const fs::path matching_result_root = resolve_path(config_dir, cfg.matching_result_root);
        const poselib::RelativePoseOptions opt = make_relative_pose_options(cfg);

        size_t total_processed = 0;
        size_t total_skipped = 0;
        size_t total_already_done = 0;
        std::cout << "Config: " << config_path << '\n';
        for (size_t dataset_idx = 0; dataset_idx < cfg.datasets.size(); ++dataset_idx) {
            const std::string &dataset = cfg.datasets[dataset_idx];
            std::cout << "Running dataset: " << dataset << '\n';
            if (is_m2m_ransac(cfg.ransac_mode)) {
                std::cout << "Running " << ransac_mode_name(cfg.ransac_mode) << " RANSAC with q_ub: " << cfg.q_ub
                          << '\n';
            } else {
                std::cout << "Running CM RANSAC (PoseLib one-to-one)\n";
            }
            const DatasetStats stats = run_dataset(matching_result_root, dataset, cfg.q_ub, cfg, opt);
            total_processed += stats.processed;
            total_skipped += stats.skipped;
            total_already_done += stats.already_done;
        }

        std::cout << "Total processed " << total_processed << " pair files";
        if (total_already_done > 0) {
            std::cout << " (" << total_already_done << " already done)";
        }
        if (total_skipped > 0) {
            std::cout << " (" << total_skipped << " skipped)";
        }
        std::cout << '\n';
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
