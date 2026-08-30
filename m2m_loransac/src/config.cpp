#include "m2m_loransac/config.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace loransac_app {
namespace {

std::vector<std::string> split_list(const std::string &value) {
    std::vector<std::string> items;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) {
            items.push_back(item);
        }
    }
    return items;
}

std::vector<double> split_double_list(const std::string &value) {
    std::vector<double> items;
    for (const std::string &item : split_list(value)) {
        items.push_back(std::stod(item));
    }
    return items;
}

bool parse_bool(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
    return value == "true" || value == "1" || value == "yes" || value == "on";
}

RansacMode parse_ransac_mode(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::toupper(c); });
    if (value == "CM") {
        return RansacMode::CM;
    }
    if (value == "HCM") {
        return RansacMode::HCM;
    }
    if (value == "MCM") {
        return RansacMode::MCM;
    }
    if (value == "HCM_MC") {
        return RansacMode::HCM_MC;
    }
    throw std::runtime_error("Invalid ransac_mode: expected CM, HCM, MCM, or HCM_MC");
}

} // namespace

std::string trim(const std::string &s) {
    const std::string whitespace = " \t\r\n";
    const auto begin = s.find_first_not_of(whitespace);
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = s.find_last_not_of(whitespace);
    return s.substr(begin, end - begin + 1);
}

fs::path resolve_path(const fs::path &base_dir, const fs::path &path) {
    return path.is_absolute() ? path : fs::weakly_canonical(base_dir / path);
}

Config load_config(const fs::path &config_path) {
    std::ifstream in(config_path);
    if (!in) {
        throw std::runtime_error("Could not open config: " + config_path.string());
    }

    std::unordered_map<std::string, std::string> values;
    std::string line;
    while (std::getline(in, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) {
            line = line.substr(0, comment);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error("Invalid config line, expected key=value: " + line);
        }
        values[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
    }

    Config cfg;
    if (values.count("matching_result_root")) {
        cfg.matching_result_root = values.at("matching_result_root");
    }
    if (values.count("datasets")) {
        cfg.datasets = split_list(values.at("datasets"));
    } else if (values.count("dataset")) {
        cfg.datasets = split_list(values.at("dataset"));
    }
    if (cfg.datasets.empty()) {
        throw std::runtime_error("Config must specify at least one dataset in dataset= or datasets=");
    }
    if (values.count("method")) {
        cfg.method = values.at("method");
    }
    if (cfg.method.empty()) {
        throw std::runtime_error("Config must specify a non-empty method=");
    }
    if (values.count("output_csv")) {
        cfg.output_csv = values.at("output_csv");
    }
    if (values.count("max_error_px")) {
        cfg.max_error_px = std::stod(values.at("max_error_px"));
    }
    if (values.count("min_iterations")) {
        cfg.min_iterations = static_cast<size_t>(std::stoull(values.at("min_iterations")));
    }
    if (values.count("max_iterations")) {
        cfg.max_iterations = static_cast<size_t>(std::stoull(values.at("max_iterations")));
    }
    if (values.count("similarity_threshold")) {
        cfg.similarity_threshold = std::stod(values.at("similarity_threshold"));
    }
    if (!std::isfinite(cfg.similarity_threshold)) {
        throw std::runtime_error("Config similarity_threshold must be finite");
    }
    if (values.count("max_matching_num")) {
        cfg.max_matching_num = static_cast<size_t>(std::stoull(values.at("max_matching_num")));
    }
    if (values.count("proposal_max_k")) {
        cfg.proposal_max_k = static_cast<size_t>(std::stoull(values.at("proposal_max_k")));
    }
    if (values.count("success_prob")) {
        cfg.success_prob = std::stod(values.at("success_prob"));
    }
    if (values.count("seed")) {
        cfg.seed = std::stoul(values.at("seed"));
    }
    if (values.count("ransac_times")) {
        cfg.ransac_times = static_cast<size_t>(std::stoull(values.at("ransac_times")));
    }
    if (values.count("tangent_sampson")) {
        cfg.tangent_sampson = parse_bool(values.at("tangent_sampson"));
    }
    if (values.count("init_with_gt")) {
        cfg.init_with_gt = parse_bool(values.at("init_with_gt"));
    }
    if (values.count("skip_existing_pairs")) {
        cfg.skip_existing_pairs = parse_bool(values.at("skip_existing_pairs"));
    }
    if (values.count("allow_unbound_pose")) {
        cfg.allow_unbound_pose = parse_bool(values.at("allow_unbound_pose"));
    }
    if (values.count("ransac_mode")) {
        cfg.ransac_mode = parse_ransac_mode(values.at("ransac_mode"));
    }
    if (values.count("top_n_candidates")) {
        cfg.top_n_candidates = static_cast<size_t>(std::stoull(values.at("top_n_candidates")));
    }
    if (values.count("pool_dedup_deg")) {
        cfg.pool_dedup_deg = std::stod(values.at("pool_dedup_deg"));
    }
    if (values.count("write_candidate_traces")) {
        cfg.write_candidate_traces = parse_bool(values.at("write_candidate_traces"));
    }
    if (values.count("write_gt_hypothesis_probe")) {
        cfg.write_gt_hypothesis_probe =
            parse_bool(values.at("write_gt_hypothesis_probe"));
    }
    if (cfg.ransac_mode == RansacMode::HCM_MC && cfg.top_n_candidates == 0) {
        throw std::runtime_error("HCM_MC requires top_n_candidates >= 1");
    }
    if (cfg.write_candidate_traces && cfg.ransac_mode != RansacMode::HCM_MC) {
        throw std::runtime_error("write_candidate_traces=true requires ransac_mode=HCM_MC");
    }
    if (cfg.write_gt_hypothesis_probe && cfg.ransac_mode != RansacMode::HCM_MC) {
        throw std::runtime_error(
            "write_gt_hypothesis_probe=true requires ransac_mode=HCM_MC");
    }
    if (cfg.write_gt_hypothesis_probe && cfg.init_with_gt) {
        throw std::runtime_error(
            "write_gt_hypothesis_probe=true requires init_with_gt=false");
    }
    if (cfg.write_gt_hypothesis_probe && cfg.ransac_times != 1) {
        throw std::runtime_error(
            "write_gt_hypothesis_probe=true requires ransac_times=1");
    }
    if (values.count("m2m_delta")) {
        cfg.m2m_delta = std::stod(values.at("m2m_delta"));
    } else if (values.count("delta")) {
        cfg.m2m_delta = std::stod(values.at("delta"));
    }
    if (!std::isfinite(cfg.m2m_delta) || cfg.m2m_delta <= 0.0) {
        throw std::runtime_error("M2M config m2m_delta must be finite and positive");
    }
    if (values.count("q_ub")) {
        cfg.q_ub = std::stod(values.at("q_ub"));
    }
    if (!std::isfinite(cfg.q_ub) || cfg.q_ub <= 0.0) {
        throw std::runtime_error("M2M config q_ub must be finite and positive");
    }
    if (cfg.min_iterations > cfg.max_iterations) {
        throw std::runtime_error("Config min_iterations must not exceed max_iterations");
    }
    if (cfg.proposal_max_k > 0) {
        if (!is_m2m_ransac(cfg.ransac_mode)) {
            throw std::runtime_error(
                "proposal_max_k requires ransac_mode=HCM, MCM, or HCM_MC");
        }
        if (cfg.min_iterations != cfg.max_iterations) {
            throw std::runtime_error(
                "proposal_max_k requires a fixed iteration budget (min_iterations=max_iterations)");
        }
        if (cfg.max_matching_num != 0 || cfg.similarity_threshold > 0.0) {
            throw std::runtime_error(
                "proposal_max_k requires the full scoring graph: set max_matching_num=0 "
                "and similarity_threshold=0");
        }
    }
    if (values.count("gt_mode")) {
        throw std::runtime_error(
            "Config key gt_mode has been removed; convert pose_intrinsics.csv ground truth to the stored "
            "image-1-to-image-2 convention and remove gt_mode from the config");
    }
    return cfg;
}

} // namespace loransac_app
