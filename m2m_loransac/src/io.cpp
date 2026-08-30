#include "m2m_loransac/config.h"
#include "m2m_loransac/io.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace loransac_app {
namespace {

std::vector<std::string> split_csv_line(const std::string &line) {
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        cols.push_back(trim(item));
    }
    return cols;
}

std::vector<std::string> split_csv_line_quoted(const std::string &line) {
    std::vector<std::string> cols;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '"') {
            in_quotes = !in_quotes;
            continue;
        }
        if (ch == ',' && !in_quotes) {
            cols.push_back(trim(cur));
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }
    cols.push_back(trim(cur));
    return cols;
}

std::vector<double> parse_numeric_list(const std::string &cell) {
    std::vector<double> out;
    const std::string t = trim(cell);
    if (t.empty()) {
        return out;
    }
    std::stringstream ss(t);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (item.empty()) {
            continue;
        }
        out.push_back(std::stod(item));
    }
    return out;
}

std::unordered_map<std::string, size_t> build_header_index(const std::vector<std::string> &header) {
    std::unordered_map<std::string, size_t> columns;
    for (size_t i = 0; i < header.size(); ++i) {
        columns[header[i]] = i;
    }
    return columns;
}

int find_column(const std::unordered_map<std::string, size_t> &columns, std::initializer_list<const char *> names) {
    for (const char *name : names) {
        const auto it = columns.find(name);
        if (it != columns.end()) {
            return static_cast<int>(it->second);
        }
    }
    return -1;
}

double at_col(const std::vector<std::string> &c, int col, const char *name) {
    if (col < 0 || static_cast<size_t>(col) >= c.size()) {
        throw std::runtime_error(std::string("Missing or out-of-range column: ") + name);
    }
    return std::stod(c[static_cast<size_t>(col)]);
}

int parse_feature_id(const std::string &cell, const char *name, size_t line_number, const fs::path &path) {
    try {
        size_t parsed = 0;
        const int value = std::stoi(cell, &parsed);
        if (parsed != cell.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    } catch (const std::exception &) {
        throw std::runtime_error("Invalid feature ID " + std::string(name) + " on line " +
                                 std::to_string(line_number) + " of " + path.string() +
                                 "; expected an integer");
    }
}

size_t parse_positive_rank(const std::string &cell, const char *name, size_t line_number, const fs::path &path) {
    try {
        if (cell.empty() || cell.front() == '-') {
            throw std::invalid_argument("not positive");
        }
        size_t parsed = 0;
        const unsigned long long value = std::stoull(cell, &parsed);
        if (parsed != cell.size() || value == 0 || value > std::numeric_limits<size_t>::max()) {
            throw std::invalid_argument("not a positive size_t");
        }
        return static_cast<size_t>(value);
    } catch (const std::exception &) {
        throw std::runtime_error("Invalid proposal rank " + std::string(name) + " on line " +
                                 std::to_string(line_number) + " of " + path.string() +
                                 "; expected a positive integer");
    }
}

double parse_finite_double(const std::string &cell, const char *name, size_t line_number,
                           const fs::path &path) {
    try {
        if (cell.empty()) {
            throw std::invalid_argument("empty value");
        }
        size_t parsed = 0;
        const double value = std::stod(cell, &parsed);
        if (parsed != cell.size() || !std::isfinite(value)) {
            throw std::invalid_argument("not a finite floating-point value");
        }
        return value;
    } catch (const std::exception &) {
        throw std::runtime_error("Invalid " + std::string(name) + " on line " +
                                 std::to_string(line_number) + " of " + path.string() +
                                 ": expected a finite floating-point value");
    }
}

} // namespace

std::map<int, PoseIntrinsics> load_pose_intrinsics(const fs::path &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open pose/intrinsics CSV: " + path.string());
    }

    std::string header_line;
    if (!std::getline(in, header_line)) {
        return {};
    }
    const auto header = split_csv_line_quoted(header_line);
    const auto columns = build_header_index(header);

    const int pair_col = find_column(columns, {"pair_idx"});
    const int qw_col = find_column(columns, {"qw"});
    const int qx_col = find_column(columns, {"qx"});
    const int qy_col = find_column(columns, {"qy"});
    const int qz_col = find_column(columns, {"qz"});
    const int tx_col = find_column(columns, {"tx"});
    const int ty_col = find_column(columns, {"ty"});
    const int tz_col = find_column(columns, {"tz"});

    const int fx1_col = find_column(columns, {"fx1"});
    const int fy1_col = find_column(columns, {"fy1"});
    const int cx1_col = find_column(columns, {"cx1"});
    const int cy1_col = find_column(columns, {"cy1"});
    const int fx2_col = find_column(columns, {"fx2"});
    const int fy2_col = find_column(columns, {"fy2"});
    const int cx2_col = find_column(columns, {"cx2"});
    const int cy2_col = find_column(columns, {"cy2"});

    const int f1_col = find_column(columns, {"f1"});
    const int f2_col = find_column(columns, {"f2"});

    const int dist0_col = find_column(columns, {"dist0_coeffs", "distortion_cam1"});
    const int dist1_col = find_column(columns, {"dist1_coeffs", "distortion_cam2"});

    const bool full_pinhole = fx1_col >= 0 && fy1_col >= 0 && cx1_col >= 0 && cy1_col >= 0 && fx2_col >= 0 &&
                              fy2_col >= 0 && cx2_col >= 0 && cy2_col >= 0;
    const bool legacy_shared_f = f1_col >= 0 && f2_col >= 0 && cx1_col >= 0 && cy1_col >= 0 && cx2_col >= 0 &&
                                  cy2_col >= 0 && !full_pinhole;

    if (!full_pinhole && !legacy_shared_f) {
        throw std::runtime_error("pose_intrinsics CSV must have either (fx1,fy1,cx1,cy1,fx2,fy2,cx2,cy2) or legacy "
                                 "(f1,cx1,cy1,f2,cx2,cy2): " +
                                 path.string());
    }
    if (pair_col < 0 || qw_col < 0 || qx_col < 0 || qy_col < 0 || qz_col < 0 || tx_col < 0 || ty_col < 0 ||
        tz_col < 0) {
        throw std::runtime_error("pose_intrinsics CSV missing pair_idx or pose columns (qw..tz): " + path.string());
    }

    std::map<int, PoseIntrinsics> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (trim(line).empty()) {
            continue;
        }
        const auto c = split_csv_line_quoted(line);

        PoseIntrinsics row;
        if (static_cast<size_t>(pair_col) >= c.size()) {
            throw std::runtime_error("Missing pair_idx column in line of " + path.string());
        }
        row.pair_idx = std::stoi(c[static_cast<size_t>(pair_col)]);

        if (full_pinhole) {
            row.fx1 = at_col(c, fx1_col, "fx1");
            row.fy1 = at_col(c, fy1_col, "fy1");
            row.cx1 = at_col(c, cx1_col, "cx1");
            row.cy1 = at_col(c, cy1_col, "cy1");
            row.fx2 = at_col(c, fx2_col, "fx2");
            row.fy2 = at_col(c, fy2_col, "fy2");
            row.cx2 = at_col(c, cx2_col, "cx2");
            row.cy2 = at_col(c, cy2_col, "cy2");
        } else {
            const double f1 = at_col(c, f1_col, "f1");
            const double f2 = at_col(c, f2_col, "f2");
            row.fx1 = row.fy1 = f1;
            row.fx2 = row.fy2 = f2;
            row.cx1 = at_col(c, cx1_col, "cx1");
            row.cy1 = at_col(c, cy1_col, "cy1");
            row.cx2 = at_col(c, cx2_col, "cx2");
            row.cy2 = at_col(c, cy2_col, "cy2");
        }

        row.q_gt =
            Eigen::Quaterniond(at_col(c, qw_col, "qw"), at_col(c, qx_col, "qx"), at_col(c, qy_col, "qy"),
                               at_col(c, qz_col, "qz"))
                .normalized();
        row.t_gt = Eigen::Vector3d(at_col(c, tx_col, "tx"), at_col(c, ty_col, "ty"), at_col(c, tz_col, "tz"));

        if (dist0_col >= 0 && static_cast<size_t>(dist0_col) < c.size()) {
            row.distortion_cam1 = parse_numeric_list(c[static_cast<size_t>(dist0_col)]);
        }
        if (dist1_col >= 0 && static_cast<size_t>(dist1_col) < c.size()) {
            row.distortion_cam2 = parse_numeric_list(c[static_cast<size_t>(dist1_col)]);
        }

        auto check_dist = [&](const std::vector<double> &d, const char *col_name) {
            if (d.empty()) {
                return;
            }
            if (d.size() != 4 && d.size() != 8) {
                throw std::runtime_error(std::string("Column ") + col_name +
                                         " must have 4 or 8 coefficients (COLMAP OPENCV / FULL_OPENCV): " +
                                         path.string());
            }
        };
        check_dist(row.distortion_cam1, "dist0_coeffs/distortion_cam1");
        check_dist(row.distortion_cam2, "dist1_coeffs/distortion_cam2");

        if (!rows.emplace(row.pair_idx, std::move(row)).second) {
            throw std::runtime_error("Duplicate pair_idx in pose/intrinsics CSV: " + path.string());
        }
    }
    return rows;
}

void load_matches(const fs::path &path, std::vector<poselib::Point2D> *points1,
                  std::vector<poselib::Point2D> *points2) {
    const MatchData data = load_match_data(path);
    *points1 = data.points1;
    *points2 = data.points2;
}

MatchData load_match_data(const fs::path &path, bool require_first_k) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open matches CSV: " + path.string());
    }

    MatchData data;
    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("Empty matches CSV: " + path.string());
    }

    const auto header = split_csv_line(line);
    const auto columns = build_header_index(header);
    const int idx1_col = find_column(columns, {"idx1", "left_idx", "feature_idx1", "feature1"});
    const int idx2_col = find_column(columns, {"idx2", "right_idx", "feature_idx2", "feature2"});
    const int x1_col = find_column(columns, {"x1"});
    const int y1_col = find_column(columns, {"y1"});
    const int x2_col = find_column(columns, {"x2"});
    const int y2_col = find_column(columns, {"y2"});
    const int similarity_col = find_column(columns, {"similarity", "sim", "score"});
    const int first_k_col = find_column(columns, {"k_first", "k"});
    if (x1_col < 0 || y1_col < 0 || x2_col < 0 || y2_col < 0) {
        throw std::runtime_error("matches CSV must contain x1,y1,x2,y2 columns: " + path.string());
    }
    if ((idx1_col < 0) != (idx2_col < 0)) {
        throw std::runtime_error("matches CSV must contain both left_idx and right_idx (or neither): " +
                                 path.string());
    }
    data.has_m2m_metadata = idx1_col >= 0 && idx2_col >= 0;
    data.has_similarity = similarity_col >= 0;
    if (require_first_k && first_k_col < 0) {
        throw std::runtime_error(
            "proposal_max_k requires a k_first or k column in the match CSV: " + path.string());
    }
    data.has_first_k = require_first_k;

    size_t line_number = 1;
    while (std::getline(in, line)) {
        ++line_number;
        if (trim(line).empty()) {
            continue;
        }
        const auto c = split_csv_line(line);
        if (c.size() != header.size()) {
            throw std::runtime_error("Invalid column count on line " + std::to_string(line_number) +
                                     " of " + path.string() + ": expected " +
                                     std::to_string(header.size()) + " fields but found " +
                                     std::to_string(c.size()));
        }
        int feature_idx1 = 0;
        int feature_idx2 = 0;
        if (data.has_m2m_metadata) {
            const size_t max_col = std::max(static_cast<size_t>(idx1_col), static_cast<size_t>(idx2_col));
            if (c.size() <= max_col) {
                throw std::runtime_error("Missing feature ID on line " + std::to_string(line_number) +
                                         " of " + path.string());
            }
            feature_idx1 = parse_feature_id(c[static_cast<size_t>(idx1_col)], "left_idx", line_number, path);
            feature_idx2 = parse_feature_id(c[static_cast<size_t>(idx2_col)], "right_idx", line_number, path);
        }

        size_t first_k = 0;
        if (data.has_first_k) {
            const size_t rank_col = static_cast<size_t>(first_k_col);
            if (c.size() <= rank_col) {
                throw std::runtime_error("Missing proposal rank on line " + std::to_string(line_number) +
                                         " of " + path.string());
            }
            first_k = parse_positive_rank(c[rank_col], "k_first/k", line_number, path);
        }

        const double x1 = parse_finite_double(c[static_cast<size_t>(x1_col)], "x1", line_number, path);
        const double y1 = parse_finite_double(c[static_cast<size_t>(y1_col)], "y1", line_number, path);
        const double x2 = parse_finite_double(c[static_cast<size_t>(x2_col)], "x2", line_number, path);
        const double y2 = parse_finite_double(c[static_cast<size_t>(y2_col)], "y2", line_number, path);

        double similarity = 0.0;
        if (data.has_similarity) {
            const size_t sim_col = static_cast<size_t>(similarity_col);
            similarity = parse_finite_double(c[sim_col], "similarity", line_number, path);
        }

        data.points1.emplace_back(x1, y1);
        data.points2.emplace_back(x2, y2);
        if (data.has_similarity) {
            data.similarity.push_back(similarity);
        }
        if (data.has_m2m_metadata) {
            data.idx1.push_back(feature_idx1);
            data.idx2.push_back(feature_idx2);
        }
        if (data.has_first_k) {
            data.first_k.push_back(first_k);
        }
    }
    return data;
}

std::vector<double> assign_prob(const std::vector<int> &idx1, const std::vector<int> &idx2, double qx_ub,double qy_ub) {
    //
    if (idx1.size() != idx2.size()) {
        throw std::invalid_argument("Cannot assign M2M probabilities with mismatched index vector sizes");
    }
    if (!std::isfinite(qx_ub) || qx_ub <= 0.0 || !std::isfinite(qy_ub) || qy_ub <= 0.0) {
        throw std::invalid_argument("Cannot assign M2M probabilities with invalid q upper bounds");
    }
    if (idx1.empty()) {
        return {};
    }
    // count number of associations for each feature in image 1 and image 2
    std::unordered_map<int, int> count_idx1;
    std::unordered_map<int, int> count_idx2;
    for (size_t i = 0; i < idx1.size(); ++i) {
        count_idx1[idx1[i]]++;
        count_idx2[idx2[i]]++;
    }
    // assign probability for each association
    std::vector<double> prob(idx1.size());
    for (size_t i = 0; i < idx1.size(); ++i) {
        // average of average probability from two directions
        double temp_prob_1 = qx_ub / static_cast<double>(count_idx1[idx1[i]]);
        double temp_prob_2 = qy_ub / static_cast<double>(count_idx2[idx2[i]]);
        prob[i] = (temp_prob_1 + temp_prob_2) / 2.0;
    }
    return prob;
}

int pair_index_from_path(const fs::path &path) {
    const std::string stem = path.stem().string();
    std::smatch match;
    if (!std::regex_search(stem, match, std::regex("(\\d+)$"))) {
        return -1;
    }
    return std::stoi(match[1].str());
}

} // namespace loransac_app
