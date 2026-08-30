#include "m2m_loransac/runner.h"

#include "m2m_loransac/geometry.h"
#include "m2m_loransac/io.h"
#include "m2m_loransac/relative_pose_m2m.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace loransac_app {
namespace {

constexpr size_t kOutputColumnCount = 33;
constexpr size_t kRunningTimeColumn = 20;
constexpr const char kOutputHeader[] =
    "pair_idx,status,error_message,num_matches,num_inliers,inlier_ratio,iterations,refinements,model_score,"
    "ransac_times,best_run_idx,"
    "q_w,q_x,q_y,q_z,t_x,t_y,t_z,rotation_error_deg,translation_error_deg,running_time_s,"
    "solve_ms,score_ms,score_calls,score_us_per_eval,refine_ms,refine_calls,rank_score_ms,rank_score_calls,"
    "ransac_total_ms,pool_insert_attempts,pool_dup_hits,pool_unique_basins";
constexpr const char kGtHypothesisProbeHeader[] =
    "pair_idx,status,error_message,pool_size,pool_capacity,pool_full,"
    "gt_hcm_score,pool_cutoff_hcm_score,gt_edge_inliers,would_enter_top_n";

uint32_t rotate_right(uint32_t value, unsigned shift) {
    return (value >> shift) | (value << (32U - shift));
}

std::string sha256_file(const fs::path &path) {
    static constexpr std::array<uint32_t, 64> kRoundConstants = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
        0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
        0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
        0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
        0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
        0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    };
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open file for SHA-256: " + path.string());
    }
    std::vector<uint8_t> message;
    char byte = 0;
    while (input.get(byte)) {
        message.push_back(static_cast<uint8_t>(static_cast<unsigned char>(byte)));
    }
    if (!input.eof()) {
        throw std::runtime_error("Could not read file for SHA-256: " + path.string());
    }
    const uint64_t bit_length = static_cast<uint64_t>(message.size()) * 8U;
    message.push_back(0x80U);
    while (message.size() % 64U != 56U) {
        message.push_back(0U);
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
        message.push_back(static_cast<uint8_t>((bit_length >> shift) & 0xffU));
    }

    std::array<uint32_t, 8> state = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    for (size_t offset = 0; offset < message.size(); offset += 64U) {
        std::array<uint32_t, 64> words{};
        for (size_t i = 0; i < 16U; ++i) {
            const size_t base = offset + i * 4U;
            words[i] = (static_cast<uint32_t>(message[base]) << 24U) |
                       (static_cast<uint32_t>(message[base + 1U]) << 16U) |
                       (static_cast<uint32_t>(message[base + 2U]) << 8U) |
                       static_cast<uint32_t>(message[base + 3U]);
        }
        for (size_t i = 16U; i < words.size(); ++i) {
            const uint32_t s0 = rotate_right(words[i - 15U], 7U) ^
                                rotate_right(words[i - 15U], 18U) ^
                                (words[i - 15U] >> 3U);
            const uint32_t s1 = rotate_right(words[i - 2U], 17U) ^
                                rotate_right(words[i - 2U], 19U) ^
                                (words[i - 2U] >> 10U);
            words[i] = words[i - 16U] + s0 + words[i - 7U] + s1;
        }

        uint32_t a = state[0];
        uint32_t b = state[1];
        uint32_t c = state[2];
        uint32_t d = state[3];
        uint32_t e = state[4];
        uint32_t f = state[5];
        uint32_t g = state[6];
        uint32_t h = state[7];
        for (size_t i = 0; i < words.size(); ++i) {
            const uint32_t sum1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
            const uint32_t choice = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + sum1 + choice + kRoundConstants[i] + words[i];
            const uint32_t sum0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

    std::ostringstream digest;
    digest << std::hex << std::setfill('0');
    for (const uint32_t word : state) {
        digest << std::setw(8) << word;
    }
    return digest.str();
}

std::string json_string_field(const fs::path &path, const std::string &key) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open provenance manifest: " + path.string());
    }
    const std::string payload(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
    std::smatch match;
    if (!std::regex_search(payload, match, pattern)) {
        throw std::runtime_error(
            "Missing JSON string field '" + key + "' in " + path.string());
    }
    return match[1].str();
}

size_t json_size_field(const fs::path &path, const std::string &key) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open provenance manifest: " + path.string());
    }
    const std::string payload(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(payload, match, pattern)) {
        throw std::runtime_error(
            "Missing JSON integer field '" + key + "' in " + path.string());
    }
    const unsigned long long value = std::stoull(match[1].str());
    if (value > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("JSON integer field '" + key + "' is too large in " + path.string());
    }
    return static_cast<size_t>(value);
}

void validate_pair_provenance(
    const fs::path &dataset_dir,
    const fs::path &method_dir,
    const fs::path &pose_csv,
    const std::map<int, PoseIntrinsics> &pose_rows,
    const std::vector<fs::path> &pair_files,
    bool allow_unbound_pose) {
    const fs::path pose_manifest = dataset_dir / "pose_intrinsics_manifest.json";
    if (!fs::is_regular_file(pose_manifest)) {
        if (allow_unbound_pose) {
            return;
        }
        throw std::runtime_error(
            "Pair-bound pose manifest is required: " + pose_manifest.string() +
            ". Set allow_unbound_pose=true only for reviewed historical artifacts");
    }
    const fs::path association_manifest = method_dir / "association_manifest.json";
    if (!fs::is_regular_file(association_manifest)) {
        throw std::runtime_error(
            "Pair-bound pose metadata requires association_manifest.json in " +
            method_dir.string());
    }
    for (const std::string &key : {"pair_file_sha256", "pair_identity_sha256"}) {
        const std::string pose_value = json_string_field(pose_manifest, key);
        const std::string association_value = json_string_field(association_manifest, key);
        if (pose_value != association_value) {
            throw std::runtime_error(
                "Pair provenance mismatch for '" + key + "' between " +
                pose_manifest.string() + " and " + association_manifest.string());
        }
    }
    const std::string expected_pose_hash = json_string_field(
        pose_manifest, "pose_intrinsics_sha256");
    if (expected_pose_hash != sha256_file(pose_csv)) {
        throw std::runtime_error(
            "pose_intrinsics_sha256 mismatch for " + pose_csv.string());
    }
    const size_t pose_pair_count = json_size_field(pose_manifest, "pair_count");
    const size_t association_pair_count = json_size_field(association_manifest, "pair_count");
    if (pose_pair_count != association_pair_count) {
        throw std::runtime_error(
            "pair_count mismatch between " + pose_manifest.string() + " and " +
            association_manifest.string());
    }
    if (pose_pair_count != pose_rows.size()) {
        throw std::runtime_error(
            "pose manifest pair_count does not match pose_intrinsics.csv rows");
    }

    std::set<int> pose_ids;
    for (const auto &[pair_idx, unused] : pose_rows) {
        (void)unused;
        pose_ids.insert(pair_idx);
    }
    std::set<int> association_ids;
    for (const fs::path &pair_file : pair_files) {
        const int pair_idx = pair_index_from_path(pair_file);
        if (!association_ids.insert(pair_idx).second) {
            throw std::runtime_error(
                "Duplicate matching-file pair_idx in " + method_dir.string());
        }
    }
    if (association_pair_count != association_ids.size()) {
        throw std::runtime_error(
            "association manifest pair_count does not match matching_*.csv files");
    }
    if (association_ids != pose_ids) {
        throw std::runtime_error(
            "matching-file pair IDs do not match pose_intrinsics.csv pair IDs");
    }
}

std::string escape_csv_field(const std::string &field) {
    if (field.find_first_of(",\"\n\r") == std::string::npos) {
        return field;
    }
    std::string escaped = "\"";
    for (const char ch : field) {
        escaped += ch == '\"' ? "\"\"" : std::string(1, ch);
    }
    return escaped + '\"';
}

void write_result_row(std::ostream &out, const std::vector<std::string> &fields) {
    if (fields.size() != kOutputColumnCount) {
        throw std::logic_error("internal result row has the wrong number of columns");
    }
    for (size_t i = 0; i < fields.size(); ++i) {
        if (i > 0) {
            out << ',';
        }
        out << escape_csv_field(fields[i]);
    }
    out << '\n';
}

void write_skipped_result(std::ostream &out, int pair_idx, const std::string &reason,
                          size_t num_matches, double running_time_s) {
    std::vector<std::string> fields(kOutputColumnCount);
    fields[0] = std::to_string(pair_idx);
    fields[1] = "skipped";
    fields[2] = reason;
    fields[3] = std::to_string(num_matches);
    fields[kRunningTimeColumn] = std::to_string(running_time_s);
    write_result_row(out, fields);
}

void write_ransac_failure_result(std::ostream &out, int pair_idx,
                                 const std::string &reason, size_t num_matches,
                                 size_t iterations, size_t ransac_times,
                                 double running_time_s) {
    std::vector<std::string> fields(kOutputColumnCount);
    fields[0] = std::to_string(pair_idx);
    fields[1] = "skipped";
    fields[2] = reason;
    fields[3] = std::to_string(num_matches);
    fields[6] = std::to_string(iterations);
    fields[9] = std::to_string(ransac_times);
    fields[kRunningTimeColumn] = std::to_string(running_time_s);
    write_result_row(out, fields);
}

/** If path exists, use stem(1).ext, stem(2).ext, ... until a free name is found. */
fs::path unique_csv_path(fs::path path) {
    if (!fs::exists(path)) {
        return path;
    }
    const fs::path parent = path.parent_path();
    const std::string stem = path.stem().string();
    const std::string ext = path.extension().string();
    for (unsigned n = 1;; ++n) {
        fs::path candidate = parent / (stem + '(' + std::to_string(n) + ')' + ext);
        if (!fs::exists(candidate)) {
            return candidate;
        }
    }
}

fs::path output_csv_for_dataset(const Config &cfg, double q_ub) {
    fs::path output_csv = cfg.output_csv;
    std::string filename = output_csv.stem().string();

    for (const std::string &prefix :
         {std::string("HCM_MC_"), std::string("HCM_"), std::string("MCM_"), std::string("CM_")}) {
        if (filename.rfind(prefix, 0) == 0) {
            filename = filename.substr(prefix.size());
            break;
        }
    }

    const std::string mode_prefix = std::string(ransac_mode_name(cfg.ransac_mode)) + "_";
    if (filename.rfind(mode_prefix, 0) != 0) {
        filename = mode_prefix + filename;
    }

    if (cfg.init_with_gt && filename.rfind("wtgt_", 0) != 0) {
        filename = "wtgt_" + filename;
    }

    if (cfg.proposal_max_k > 0) {
        filename += "_proposal_k_" + std::to_string(cfg.proposal_max_k);
    }

    // q_ub is the Assumption-1 prior used only by HCM/HCM_MC scoring. MCM scores
    // purely by maximum-cardinality matching and is independent of q_ub, so its
    // output filename omits the q_ub tag.
    if (is_m2m_ransac(cfg.ransac_mode) && cfg.ransac_mode != RansacMode::MCM) {
        std::ostringstream suffix;
        suffix << "_q_ub_" << std::fixed << std::setprecision(2) << q_ub;
        filename += suffix.str();
    }
    filename += output_csv.extension().string();
    output_csv.replace_filename(filename);
    return output_csv;
}

fs::path candidate_trace_path(const fs::path &output_csv, int pair_idx) {
    std::ostringstream filename;
    filename << "candidate_" << std::setw(6) << std::setfill('0') << pair_idx << ".csv";
    return output_csv.parent_path() / "candidate_traces" / output_csv.stem() / filename.str();
}

fs::path gt_hypothesis_probe_path(const fs::path &output_csv) {
    return output_csv.parent_path() /
           (output_csv.stem().string() + "_gt_hypothesis_probe.csv");
}

void write_pose_fields(std::ostream &out, const poselib::CameraPose &pose) {
    out << pose.q(0) << ',' << pose.q(1) << ',' << pose.q(2) << ',' << pose.q(3) << ','
        << pose.t(0) << ',' << pose.t(1) << ',' << pose.t(2);
}

void write_candidate_trace(const fs::path &path, int pair_idx,
                           const std::vector<M2MCandidateTrace> &traces,
                           const PoseIntrinsics &intr) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Could not open candidate trace: " + path.string());
    }
    out << std::setprecision(17);
    out << "pair_idx,stored_hcm_rank,stored_hcm_score,seed_hcm_left,seed_hcm_right,"
           "refined_hcm_score,refined_hcm_left,refined_hcm_right,refined_msac_cost,"
           "edge_inliers,mcm_cardinality,stage2_refine_ms,mcm_score_ms,"
           "seed_qw,seed_qx,seed_qy,seed_qz,seed_tx,seed_ty,seed_tz,"
           "refined_qw,refined_qx,refined_qy,refined_qz,refined_tx,refined_ty,refined_tz,"
           "polished_qw,polished_qx,polished_qy,polished_qz,polished_tx,polished_ty,polished_tz,"
           "seed_rotation_error_deg,seed_translation_error_deg,seed_pose_error_deg,"
           "refined_rotation_error_deg,refined_translation_error_deg,refined_pose_error_deg,"
           "polished_rotation_error_deg,polished_translation_error_deg,polished_pose_error_deg\n";
    for (const M2MCandidateTrace &trace : traces) {
        const GtError seed_error = compute_gt_error(trace.seed_pose, intr);
        const GtError refined_error = compute_gt_error(trace.refined_pose, intr);
        const GtError polished_error = compute_gt_error(trace.polished_pose, intr);
        out << pair_idx << ',' << trace.stored_hcm_rank << ',' << trace.stored_hcm_score << ','
            << trace.seed_hcm_left << ',' << trace.seed_hcm_right << ',' << trace.refined_hcm_score << ','
            << trace.refined_hcm_left << ',' << trace.refined_hcm_right << ',' << trace.refined_msac_cost << ','
            << trace.edge_inliers << ',' << trace.cardinality << ','
            << static_cast<double>(trace.stage2_refine_ns) / 1e6 << ','
            << static_cast<double>(trace.mcm_score_ns) / 1e6 << ',';
        write_pose_fields(out, trace.seed_pose);
        out << ',';
        write_pose_fields(out, trace.refined_pose);
        out << ',';
        write_pose_fields(out, trace.polished_pose);
        out << ',' << seed_error.rotation_deg << ',' << seed_error.translation_deg << ','
            << std::max(seed_error.rotation_deg, seed_error.translation_deg) << ','
            << refined_error.rotation_deg << ',' << refined_error.translation_deg << ','
            << std::max(refined_error.rotation_deg, refined_error.translation_deg) << ','
            << polished_error.rotation_deg << ',' << polished_error.translation_deg << ','
            << std::max(polished_error.rotation_deg, polished_error.translation_deg) << '\n';
    }
    if (!out) {
        throw std::runtime_error("Failed while writing candidate trace: " + path.string());
    }
}

void write_gt_hypothesis_probe_status(std::ostream &out, int pair_idx,
                                      const std::string &status,
                                      const std::string &reason = "") {
    std::vector<std::string> fields(10);
    fields[0] = std::to_string(pair_idx);
    fields[1] = status;
    fields[2] = reason;
    for (size_t i = 0; i < fields.size(); ++i) {
        if (i > 0) {
            out << ',';
        }
        out << escape_csv_field(fields[i]);
    }
    out << '\n';
}

void write_gt_hypothesis_probe_result(
    std::ostream &out, int pair_idx,
    const M2MGroundTruthHypothesisProbe &probe) {
    if (!probe.evaluated) {
        throw std::runtime_error(
            "Attempted to write an unevaluated ground-truth hypothesis probe");
    }
    out << pair_idx << ",success,," << probe.pool_size << ','
        << probe.pool_capacity << ',' << (probe.pool_full ? 1 : 0) << ','
        << probe.gt_hcm_score << ',' << probe.pool_cutoff_hcm_score << ','
        << probe.gt_edge_inliers << ',' << (probe.would_enter_pool ? 1 : 0)
        << '\n';
}

std::unordered_set<int> load_processed_pair_indices(const fs::path &output_csv) {
    std::unordered_set<int> processed;
    std::ifstream in(output_csv);
    if (!in) {
        return processed;
    }

    std::string line;
    if (!std::getline(in, line)) {
        return processed;
    }

    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto comma = line.find(',');
        if (comma == std::string::npos) {
            continue;
        }
        try {
            processed.insert(std::stoi(line.substr(0, comma)));
        } catch (const std::exception &) {
            continue;
        }
    }
    return processed;
}

void validate_existing_result_schema(const fs::path &output_csv) {
    std::ifstream in(output_csv);
    std::string header;
    if (!in || !std::getline(in, header)) {
        return; // An absent or empty file will be initialized below.
    }
    if (!header.empty() && header.back() == '\r') {
        header.pop_back();
    }
    if (header == kOutputHeader) {
        return;
    }
    if (header.find("gt_mode_used") != std::string::npos) {
        throw std::runtime_error(
            "Existing result CSV uses the legacy gt_mode_used schema: " + output_csv.string() +
            ". Migrate it to the standardized ground-truth schema or choose a new output "
            "(for example, set skip_existing_pairs=false)");
    }
    throw std::runtime_error("Existing result CSV has an incompatible header: " + output_csv.string() +
                             ". Migrate it or choose a new output before resuming");
}

void validate_existing_gt_hypothesis_probe_schema(const fs::path &probe_csv) {
    std::ifstream in(probe_csv);
    std::string header;
    if (!in || !std::getline(in, header)) {
        return;
    }
    if (!header.empty() && header.back() == '\r') {
        header.pop_back();
    }
    if (header != kGtHypothesisProbeHeader) {
        throw std::runtime_error(
            "Existing ground-truth hypothesis probe CSV has an incompatible header: " +
            probe_csv.string());
    }
}

void filter_by_similarity(MatchData *matches, double threshold) {
    if (threshold <= 0.0) {
        return;
    }
    if (!matches->has_similarity || matches->similarity.size() != matches->points1.size()) {
        throw std::runtime_error("similarity_threshold requires a similarity column in the match CSV");
    }

    MatchData filtered;
    filtered.has_m2m_metadata = matches->has_m2m_metadata;
    filtered.has_similarity = true;
    filtered.has_first_k = matches->has_first_k;
    filtered.points1.reserve(matches->points1.size());
    filtered.points2.reserve(matches->points2.size());
    filtered.similarity.reserve(matches->similarity.size());
    if (matches->has_m2m_metadata) {
        filtered.idx1.reserve(matches->idx1.size());
        filtered.idx2.reserve(matches->idx2.size());
    }
    if (matches->has_first_k) {
        filtered.first_k.reserve(matches->first_k.size());
    }

    for (size_t i = 0; i < matches->similarity.size(); ++i) {
        if (matches->similarity[i] < threshold) {
            continue;
        }
        filtered.points1.push_back(matches->points1[i]);
        filtered.points2.push_back(matches->points2[i]);
        filtered.similarity.push_back(matches->similarity[i]);
        if (matches->has_m2m_metadata) {
            filtered.idx1.push_back(matches->idx1[i]);
            filtered.idx2.push_back(matches->idx2[i]);
        }
        if (matches->has_first_k) {
            filtered.first_k.push_back(matches->first_k[i]);
        }
    }

    *matches = std::move(filtered);
}

void keep_match_indices(MatchData *matches, const std::vector<size_t> &indices) {
    MatchData selected;
    selected.has_m2m_metadata = matches->has_m2m_metadata;
    selected.has_similarity = matches->has_similarity;
    selected.has_first_k = matches->has_first_k;
    selected.points1.reserve(indices.size());
    selected.points2.reserve(indices.size());
    if (matches->has_similarity) {
        selected.similarity.reserve(indices.size());
    }
    if (matches->has_m2m_metadata) {
        selected.idx1.reserve(indices.size());
        selected.idx2.reserve(indices.size());
    }
    if (matches->has_first_k) {
        selected.first_k.reserve(indices.size());
    }

    for (const size_t idx : indices) {
        selected.points1.push_back(matches->points1[idx]);
        selected.points2.push_back(matches->points2[idx]);
        if (matches->has_similarity) {
            selected.similarity.push_back(matches->similarity[idx]);
        }
        if (matches->has_m2m_metadata) {
            selected.idx1.push_back(matches->idx1[idx]);
            selected.idx2.push_back(matches->idx2[idx]);
        }
        if (matches->has_first_k) {
            selected.first_k.push_back(matches->first_k[idx]);
        }
    }

    *matches = std::move(selected);
}

void limit_matches_first_n(MatchData *matches, size_t max_count) {
    if (max_count == 0 || matches->points1.size() <= max_count) {
        return;
    }

    // Preserve the original behavior: keep the first max_count rows in CSV order.
    matches->points1.resize(max_count);
    matches->points2.resize(max_count);
    if (matches->prob.size() > 0) {
        matches->prob.resize(max_count);
    }
    if (matches->has_similarity) {
        matches->similarity.resize(max_count);
    }
    if (matches->has_m2m_metadata) {
        matches->idx1.resize(max_count);
        matches->idx2.resize(max_count);
    }
    if (matches->has_first_k) {
        matches->first_k.resize(max_count);
    }
}

void limit_matches_progressive_degree_cap(MatchData *matches, size_t max_count) {
    std::cout << "select the first " << max_count << " matches by progressive degree cap" << std::endl;
    int association_number = matches->points1.size();
    if (max_count == 0 || association_number <= max_count) {
        return;
    }
    std::vector<size_t> selected_indices;
    selected_indices.reserve(max_count);
    std::vector<char> selected(association_number, 0);
    std::unordered_map<int, size_t> degree_left;
    std::unordered_map<int, size_t> degree_right;
    int degree_cap;
    for (degree_cap = 1;selected_indices.size() < max_count && selected_indices.size() < association_number; ++degree_cap) {
        bool added_this_pass = false;
        for (size_t i = 0; i < association_number && selected_indices.size() < max_count; ++i) {
            // selected in previous passes
            if (selected[i]) {
                continue;
            }
            const int left_idx = matches->idx1[i];
            const int right_idx = matches->idx2[i];
            if (degree_left[left_idx] >= degree_cap || degree_right[right_idx] >= degree_cap) {
                continue;
            }
            selected[i] = 1;
            selected_indices.push_back(i);
            ++degree_left[left_idx];
            ++degree_right[right_idx];
            added_this_pass = true;
        }

        if (!added_this_pass) {
            break;
        }
    }
    keep_match_indices(matches, selected_indices);
    std::cout << "selected " << selected_indices.size() << " with degree cap " << degree_cap << std::endl;
}

} // namespace

poselib::RelativePoseOptions make_relative_pose_options(const Config &cfg) {
    poselib::RelativePoseOptions opt;
    opt.max_error = cfg.max_error_px;
    opt.tangent_sampson = cfg.tangent_sampson;
    opt.ransac.min_iterations = cfg.min_iterations;
    opt.ransac.max_iterations = cfg.max_iterations;
    opt.ransac.success_prob = cfg.success_prob;
    opt.ransac.seed = cfg.seed;
    opt.bundle.loss_scale = cfg.max_error_px;
    return opt;
}

DatasetStats run_dataset(const fs::path &matching_result_root, const std::string &dataset, double q_ub,
                         const Config &cfg, const poselib::RelativePoseOptions &opt) {
    const fs::path dataset_dir = matching_result_root / dataset;
    const fs::path method_dir = dataset_dir / cfg.method;
    const fs::path pose_csv = dataset_dir / "pose_intrinsics.csv";
    fs::path output_csv = method_dir / output_csv_for_dataset(cfg, q_ub);

    const auto pose_rows = load_pose_intrinsics(pose_csv);

    std::vector<fs::path> pair_files;
    for (const auto &entry : fs::directory_iterator(method_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto filename = entry.path().filename().string();
        if (filename.rfind("matching_", 0) == 0 && entry.path().extension() == ".csv") {
            pair_files.push_back(entry.path());
        }
    }
    std::sort(pair_files.begin(), pair_files.end(), [](const fs::path &a, const fs::path &b) {
        return pair_index_from_path(a) < pair_index_from_path(b);
    });

    validate_pair_provenance(
        dataset_dir,
        method_dir,
        pose_csv,
        pose_rows,
        pair_files,
        cfg.allow_unbound_pose);

    std::unordered_set<int> processed_pairs;
    const bool resume_existing = cfg.skip_existing_pairs && fs::exists(output_csv);
    if (resume_existing) {
        validate_existing_result_schema(output_csv);
        processed_pairs = load_processed_pair_indices(output_csv);
    }
    if (!cfg.skip_existing_pairs) {
        output_csv = unique_csv_path(std::move(output_csv));
    }

    fs::path gt_probe_csv;
    std::unordered_set<int> processed_probe_pairs;
    if (cfg.write_gt_hypothesis_probe) {
        gt_probe_csv = gt_hypothesis_probe_path(output_csv);
        if (resume_existing) {
            if (!fs::is_regular_file(gt_probe_csv)) {
                throw std::runtime_error(
                    "Retained pose rows are missing the ground-truth hypothesis probe sidecar: " +
                    gt_probe_csv.string());
            }
            validate_existing_gt_hypothesis_probe_schema(gt_probe_csv);
            processed_probe_pairs = load_processed_pair_indices(gt_probe_csv);
            if (processed_probe_pairs != processed_pairs) {
                throw std::runtime_error(
                    "Pose and ground-truth hypothesis probe sidecars cover different pair sets: " +
                    output_csv.string());
            }
        }
    }

    if (output_csv.has_parent_path()) {
        fs::create_directories(output_csv.parent_path());
    }
    const bool append_existing = resume_existing && !processed_pairs.empty();
    std::ofstream out;
    if (append_existing) {
        out.open(output_csv, std::ios::app);
    } else {
        out.open(output_csv);
    }
    if (!out) {
        throw std::runtime_error("Could not open output CSV: " + output_csv.string());
    }

    if (!append_existing) {
        out << kOutputHeader << '\n';
    }

    std::ofstream gt_probe_out;
    if (cfg.write_gt_hypothesis_probe) {
        if (append_existing) {
            gt_probe_out.open(gt_probe_csv, std::ios::app);
        } else {
            gt_probe_out.open(gt_probe_csv);
        }
        if (!gt_probe_out) {
            throw std::runtime_error(
                "Could not open ground-truth hypothesis probe CSV: " +
                gt_probe_csv.string());
        }
        gt_probe_out << std::setprecision(17);
        if (!append_existing) {
            gt_probe_out << kGtHypothesisProbeHeader << '\n';
        }
    }

    DatasetStats dataset_stats;
    for (const fs::path &pair_file : pair_files) {
        const int pair_idx = pair_index_from_path(pair_file);
        if (processed_pairs.count(pair_idx) > 0) {
            if (cfg.write_candidate_traces && !fs::is_regular_file(candidate_trace_path(output_csv, pair_idx))) {
                throw std::runtime_error("Retained pose row is missing candidate trace: " +
                                         candidate_trace_path(output_csv, pair_idx).string());
            }
            ++dataset_stats.already_done;
            continue;
        }
        const auto pair_start = std::chrono::steady_clock::now();
        const auto pair_running_time_s = [&pair_start]() {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - pair_start).count();
        };
        const auto pose_it = pose_rows.find(pair_idx);
        // missing pose intrinsics
        if (pair_idx < 0 || pose_it == pose_rows.end()) {
            write_skipped_result(out, pair_idx, "missing_pose_intrinsics", 0, pair_running_time_s());
            if (cfg.write_gt_hypothesis_probe) {
                write_gt_hypothesis_probe_status(
                    gt_probe_out, pair_idx, "skipped", "missing_pose_intrinsics");
            }
            ++dataset_stats.skipped;
            continue;
        }

        MatchData matches;
        try {
            matches = load_match_data(pair_file, cfg.proposal_max_k > 0);
        } catch (const std::exception &error) {
            write_skipped_result(out, pair_idx, std::string("invalid_match_csv: ") + error.what(), 0,
                                 pair_running_time_s());
            if (cfg.write_gt_hypothesis_probe) {
                write_gt_hypothesis_probe_status(
                    gt_probe_out, pair_idx, "skipped",
                    std::string("invalid_match_csv: ") + error.what());
            }
            ++dataset_stats.skipped;
            continue;
        }
        if (!matches.has_m2m_metadata) {
            write_skipped_result(out, pair_idx, "missing_m2m_metadata", matches.points1.size(),
                                 pair_running_time_s());
            if (cfg.write_gt_hypothesis_probe) {
                write_gt_hypothesis_probe_status(
                    gt_probe_out, pair_idx, "skipped", "missing_m2m_metadata");
            }
            ++dataset_stats.skipped;
            continue;
        }
        if (std::any_of(matches.idx1.begin(), matches.idx1.end(), [](int id) { return id < 0; }) ||
            std::any_of(matches.idx2.begin(), matches.idx2.end(), [](int id) { return id < 0; })) {
            write_skipped_result(out, pair_idx, "invalid_feature_ids", matches.points1.size(),
                                 pair_running_time_s());
            if (cfg.write_gt_hypothesis_probe) {
                write_gt_hypothesis_probe_status(
                    gt_probe_out, pair_idx, "skipped", "invalid_feature_ids");
            }
            ++dataset_stats.skipped;
            continue;
        }
        if (cfg.similarity_threshold > 0.0) {
            if (!matches.has_similarity || matches.similarity.size() != matches.points1.size()) {
                write_skipped_result(out, pair_idx, "similarity_threshold_requires_similarity",
                                     matches.points1.size(), pair_running_time_s());
                if (cfg.write_gt_hypothesis_probe) {
                    write_gt_hypothesis_probe_status(
                        gt_probe_out, pair_idx, "skipped",
                        "similarity_threshold_requires_similarity");
                }
                ++dataset_stats.skipped;
                continue;
            }
            filter_by_similarity(&matches, cfg.similarity_threshold);
        }
        if (matches.idx1.size() != matches.points1.size() || matches.idx2.size() != matches.points1.size()) {
            write_skipped_result(out, pair_idx, "column_size_mismatch", matches.points1.size(),
                                 pair_running_time_s());
            if (cfg.write_gt_hypothesis_probe) {
                write_gt_hypothesis_probe_status(
                    gt_probe_out, pair_idx, "skipped", "column_size_mismatch");
            }
            ++dataset_stats.skipped;
            continue;
        }
        if (cfg.proposal_max_k > 0 &&
            (!matches.has_first_k || matches.first_k.size() != matches.points1.size())) {
            write_skipped_result(out, pair_idx, "proposal_rank_size_mismatch", matches.points1.size(),
                                 pair_running_time_s());
            if (cfg.write_gt_hypothesis_probe) {
                write_gt_hypothesis_probe_status(
                    gt_probe_out, pair_idx, "skipped", "proposal_rank_size_mismatch");
            }
            ++dataset_stats.skipped;
            continue;
        }
        // limit_matches_progressive_degree_cap(&matches, cfg.max_matching_num);
        limit_matches_first_n(&matches, cfg.max_matching_num);
        // max idx1
        int association_number = matches.points1.size();
        if (association_number < 5) {
            write_skipped_result(out, pair_idx, "not_enough_matches", matches.points1.size(),
                                 pair_running_time_s());
            if (cfg.write_gt_hypothesis_probe) {
                write_gt_hypothesis_probe_status(
                    gt_probe_out, pair_idx, "skipped", "not_enough_matches");
            }
            ++dataset_stats.skipped;
            continue;
        }
        if (is_m2m_ransac(cfg.ransac_mode) && !matches.has_m2m_metadata) {
            write_skipped_result(out, pair_idx, "missing_m2m_metadata", matches.points1.size(),
                                 pair_running_time_s());
            if (cfg.write_gt_hypothesis_probe) {
                write_gt_hypothesis_probe_status(
                    gt_probe_out, pair_idx, "skipped", "missing_m2m_metadata");
            }
            ++dataset_stats.skipped;
            continue;
        }
        if (is_m2m_ransac(cfg.ransac_mode)) {
            matches.prob = assign_prob(matches.idx1, matches.idx2, q_ub, q_ub);
        }

        const PoseIntrinsics &intr = pose_it->second;
        const poselib::CameraPose ground_truth_pose = make_gt_pose(intr);
        const auto cameras = make_cameras_from_pose_intrinsics(intr);
        const poselib::Camera &camera1 = cameras.first;
        const poselib::Camera &camera2 = cameras.second;

        const size_t ransac_times = std::max<size_t>(cfg.ransac_times, 1);
        std::vector<int> indices(matches.points1.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(static_cast<uint32_t>(cfg.seed + static_cast<unsigned long>(pair_idx)));

        bool has_result = false;
        size_t best_run_idx = 0;
        poselib::CameraPose best_pose;
        poselib::RansacStats best_stats;
        M2MTimingStats best_timing; // stays zero for the CM (plain PoseLib) path
        std::vector<M2MCandidateTrace> best_candidate_traces;
        M2MGroundTruthHypothesisProbe best_gt_probe;
        size_t failed_run_iterations = 0;
        bool observed_failed_run = false;
        bool failed_run_iterations_are_uniform = true;
        // Outer loop: repeated RANSAC runs
        for (size_t run_idx = 0; run_idx < ransac_times; ++run_idx) {
            if (run_idx > 0) {
                std::shuffle(indices.begin(), indices.end(), rng);
            }

            std::vector<poselib::Point2D> shuffled_points1;
            std::vector<poselib::Point2D> shuffled_points2;
            std::vector<int> shuffled_idx1;
            std::vector<int> shuffled_idx2;
            std::vector<double> shuffled_prob;
            std::vector<size_t> shuffled_first_k;
            shuffled_points1.reserve(matches.points1.size());
            shuffled_points2.reserve(matches.points2.size());
            if (is_m2m_ransac(cfg.ransac_mode)) {
                shuffled_idx1.reserve(matches.idx1.size());
                shuffled_idx2.reserve(matches.idx2.size());
                shuffled_prob.reserve(matches.prob.size());
            }
            if (cfg.proposal_max_k > 0) {
                shuffled_first_k.reserve(matches.first_k.size());
            }
            for (const int idx : indices) {
                const size_t src_idx = static_cast<size_t>(idx);
                shuffled_points1.push_back(matches.points1[src_idx]);
                shuffled_points2.push_back(matches.points2[src_idx]);
                if (is_m2m_ransac(cfg.ransac_mode)) {
                    shuffled_idx1.push_back(matches.idx1[src_idx]);
                    shuffled_idx2.push_back(matches.idx2[src_idx]);
                    shuffled_prob.push_back(matches.prob[src_idx]);
                }
                if (cfg.proposal_max_k > 0) {
                    shuffled_first_k.push_back(matches.first_k[src_idx]);
                }
            }

            std::vector<size_t> proposal_indices;
            if (cfg.proposal_max_k > 0) {
                proposal_indices.reserve(shuffled_first_k.size());
                for (size_t edge_idx = 0; edge_idx < shuffled_first_k.size(); ++edge_idx) {
                    if (shuffled_first_k[edge_idx] <= cfg.proposal_max_k) {
                        proposal_indices.push_back(edge_idx);
                    }
                }
            }

            poselib::CameraPose run_pose;
            std::vector<char> run_inliers;
            poselib::RelativePoseOptions run_opt = opt;
            run_opt.ransac.seed = cfg.seed + static_cast<unsigned long>(pair_idx) + static_cast<unsigned long>(run_idx);
            if (cfg.init_with_gt) {
                run_pose = make_gt_pose(intr);
                run_opt.ransac.score_initial_model = true;
            }
            poselib::RansacStats run_stats;
            M2MTimingStats run_timing;
            std::vector<M2MCandidateTrace> run_candidate_traces;
            M2MGroundTruthHypothesisProbe run_gt_probe;
            if (is_m2m_ransac(cfg.ransac_mode)) {
                run_stats = estimate_relative_pose_m2m(shuffled_points1, shuffled_points2, camera1, camera2,
                                                       shuffled_idx1, shuffled_idx2, shuffled_prob, cfg.m2m_delta,
                                                       cfg.ransac_mode, cfg.top_n_candidates, run_opt, &run_pose,
                                                       &run_inliers, &run_timing, cfg.pool_dedup_deg,
                                                       cfg.write_candidate_traces ? &run_candidate_traces : nullptr,
                                                       cfg.proposal_max_k > 0 ? &proposal_indices : nullptr,
                                                       cfg.write_gt_hypothesis_probe ? &ground_truth_pose : nullptr,
                                                       cfg.write_gt_hypothesis_probe ? &run_gt_probe : nullptr);
            } else {
                run_stats = poselib::estimate_relative_pose(shuffled_points1, shuffled_points2, camera1, camera2,
                                                            run_opt, &run_pose, &run_inliers);
            }
            // The default RansacStats sentinel means that no model was generated
            // (for example, when the maximum one-to-one cardinality is below 5).
            const bool run_has_model = std::isfinite(run_stats.model_score) &&
                                       run_stats.model_score < std::numeric_limits<double>::max();
            if (!run_has_model) {
                if (observed_failed_run && run_stats.iterations != failed_run_iterations) {
                    failed_run_iterations_are_uniform = false;
                }
                failed_run_iterations = run_stats.iterations;
                observed_failed_run = true;
            }
            // Select the best valid model among repeated RANSAC runs.
            if (run_has_model && (!has_result || run_stats.model_score < best_stats.model_score)) {
                has_result = true;
                best_run_idx = run_idx;
                best_pose = run_pose;
                best_stats = run_stats;
                best_timing = run_timing;
                best_candidate_traces = std::move(run_candidate_traces);
                best_gt_probe = run_gt_probe;
            }
        }

        if (!has_result) {
            if (!observed_failed_run || !failed_run_iterations_are_uniform) {
                throw std::runtime_error(
                    "Failed RANSAC repetitions have no uniform iteration evidence for pair " +
                    std::to_string(pair_idx));
            }
            const std::string reason =
                failed_run_iterations == 0 ? "proposal_graph_not_enough_o2o" : "ransac_failed";
            write_ransac_failure_result(out, pair_idx, reason, matches.points1.size(),
                                        failed_run_iterations, ransac_times,
                                        pair_running_time_s());
            if (cfg.write_gt_hypothesis_probe) {
                write_gt_hypothesis_probe_status(
                    gt_probe_out, pair_idx, "skipped", reason);
            }
            ++dataset_stats.skipped;
            continue;
        }

        const GtError gt_error = compute_gt_error(best_pose, intr);

        if (cfg.write_candidate_traces) {
            if (best_candidate_traces.empty()) {
                throw std::runtime_error("Successful HCM_MC run produced no candidate trace for pair " +
                                         std::to_string(pair_idx));
            }
            write_candidate_trace(candidate_trace_path(output_csv, pair_idx), pair_idx,
                                  best_candidate_traces, intr);
        }
        if (cfg.write_gt_hypothesis_probe) {
            if (!best_gt_probe.evaluated) {
                throw std::runtime_error(
                    "Successful HCM_MC run produced no ground-truth hypothesis probe for pair " +
                    std::to_string(pair_idx));
            }
            write_gt_hypothesis_probe_result(gt_probe_out, pair_idx, best_gt_probe);
        }

        const auto ns_to_ms = [](uint64_t ns) { return static_cast<double>(ns) / 1e6; };
        const double score_us_per_eval =
            best_timing.score_calls > 0
                ? static_cast<double>(best_timing.score_ns) / 1e3 / static_cast<double>(best_timing.score_calls)
                : 0.0;
        // Sample the pair timer before any part of the CSV row is emitted so
        // running_time_s has one consistent, serialization-free boundary.
        const double running_time_s = pair_running_time_s();
        out << pair_idx << ",success,," << matches.points1.size() << ',' << best_stats.num_inliers << ','
            << best_stats.inlier_ratio << ',' << best_stats.iterations << ',' << best_stats.refinements << ','
            << best_stats.model_score << ',' << ransac_times << ',' << best_run_idx << ',' << best_pose.q(0) << ','
            << best_pose.q(1) << ',' << best_pose.q(2) << ',' << best_pose.q(3) << ',' << best_pose.t(0) << ','
            << best_pose.t(1) << ',' << best_pose.t(2) << ',' << gt_error.rotation_deg << ','
            << gt_error.translation_deg << ',' << running_time_s << ',' << ns_to_ms(best_timing.solve_ns)
            << ',' << ns_to_ms(best_timing.score_ns) << ',' << best_timing.score_calls << ',' << score_us_per_eval
            << ',' << ns_to_ms(best_timing.refine_ns) << ',' << best_timing.refine_calls << ','
            << ns_to_ms(best_timing.rank_score_ns) << ',' << best_timing.rank_score_calls << ','
            << ns_to_ms(best_timing.total_ns) << ',' << best_timing.pool_insert_attempts << ','
            << best_timing.pool_dup_hits << ',' << best_timing.pool_unique_basins << '\n';
        ++dataset_stats.processed;
        if ((dataset_stats.processed + dataset_stats.skipped) % 5 == 0) {
            std::cout << "Processed " << dataset_stats.processed + dataset_stats.skipped << " pair files\n";
        }
    }

    std::cout << "Dataset: " << dataset << ", method: " << cfg.method << '\n';
    std::cout << "Processed " << dataset_stats.processed << " pair files";
    if (dataset_stats.already_done > 0) {
        std::cout << " (" << dataset_stats.already_done << " already done)";
    }
    if (dataset_stats.skipped > 0) {
        std::cout << " (" << dataset_stats.skipped << " skipped)";
    }
    std::cout << "\nWrote: " << output_csv << "\n\n";
    return dataset_stats;
}

} // namespace loransac_app
