#include "gms_matcher_m2m.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    fs::path root;
    bool with_scale = false;
    bool with_rotation = false;
    bool auto_mask = false;
    double threshold_factor = DEFAULT_THRESH_FACTOR_M2M;
    int grid_size = 20;
};

struct ImageSize {
    int width = 0;
    int height = 0;
};

struct CsvTable {
    std::string header_line;
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> fields;
};

struct PairColumns {
    int left_idx = -1;
    int right_idx = -1;
    int x1 = -1;
    int y1 = -1;
    int x2 = -1;
    int y2 = -1;
};

struct ImgPairSizes {
    ImageSize left;
    ImageSize right;
};

struct PreparedMatches {
    std::vector<KeyPoint> left_keypoints;
    std::vector<KeyPoint> right_keypoints;
    std::vector<DMatch> matches;
    std::vector<size_t> source_row_indices;
    ImageSize left_size;
    ImageSize right_size;
    int skipped_rows = 0;
};

struct ValidRow {
    size_t source_row = 0;
    long long left_idx = 0;
    long long right_idx = 0;
    double x1 = 0.0;
    double y1 = 0.0;
    double x2 = 0.0;
    double y2 = 0.0;
};

struct KeypointBounds {
    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    void add(double x, double y) {
        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
    }

    ImageSize extentSize() const {
        ImageSize size;
        if (std::isfinite(max_x)) {
            size.width = std::max(1, static_cast<int>(std::floor(max_x)) + 1);
        }
        if (std::isfinite(max_y)) {
            size.height = std::max(1, static_cast<int>(std::floor(max_y)) + 1);
        }
        return size;
    }

    ImageSize maskSize(double margin) const {
        ImageSize size;
        if (std::isfinite(min_x) && std::isfinite(max_x)) {
            size.width = std::max(1, static_cast<int>(std::floor((max_x + margin) - (min_x - margin))) + 1);
        }
        if (std::isfinite(min_y) && std::isfinite(max_y)) {
            size.height = std::max(1, static_cast<int>(std::floor((max_y + margin) - (min_y - margin))) + 1);
        }
        return size;
    }
};

std::vector<std::string> splitCsvLine(const std::string &line) {
    std::vector<std::string> out;
    std::string cell;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                cell.push_back('"');
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            out.push_back(cell);
            cell.clear();
        } else {
            cell.push_back(c);
        }
    }
    out.push_back(cell);
    return out;
}

void writeCsvRow(std::ostream &output, const std::vector<std::string> &fields) {
    const auto escape = [](const std::string &field) {
        if (field.find_first_of(",\"\n\r") == std::string::npos) {
            return field;
        }

        std::string escaped = "\"";
        for (const char c : field) {
            if (c == '"') {
                escaped += "\"\"";
            } else {
                escaped.push_back(c);
            }
        }
        escaped.push_back('"');
        return escaped;
    };

    for (size_t i = 0; i < fields.size(); ++i) {
        if (i > 0) {
            output << ',';
        }
        output << escape(fields[i]);
    }
    output << '\n';
}

int columnIndex(const std::vector<std::string> &header, const std::string &name) {
    const auto it = std::find(header.begin(), header.end(), name);
    return it == header.end() ? -1 : static_cast<int>(std::distance(header.begin(), it));
}

PairColumns getPairColumns(const CsvTable &table, bool need_coordinates = true) {
    PairColumns columns{
        columnIndex(table.header, "left_idx"),
        columnIndex(table.header, "right_idx"),
        columnIndex(table.header, "x1"),
        columnIndex(table.header, "y1"),
        columnIndex(table.header, "x2"),
        columnIndex(table.header, "y2"),
    };
    if (columns.left_idx < 0 || columns.right_idx < 0) {
        throw std::runtime_error("CSV must contain left_idx and right_idx columns");
    }
    if (need_coordinates && (columns.x1 < 0 || columns.y1 < 0 || columns.x2 < 0 || columns.y2 < 0)) {
        throw std::runtime_error("CSV must contain x1,y1,x2,y2 columns");
    }
    return columns;
}

double parseDouble(const std::vector<std::string> &fields, int index) {
    if (index < 0 || static_cast<size_t>(index) >= fields.size()) {
        throw std::runtime_error("missing numeric field");
    }
    return std::stod(fields[static_cast<size_t>(index)]);
}

long long parseInteger(const std::vector<std::string> &fields, int index) {
    if (index < 0 || static_cast<size_t>(index) >= fields.size()) {
        throw std::runtime_error("missing integer field");
    }
    return std::stoll(fields[static_cast<size_t>(index)]);
}

CsvTable readCsv(const fs::path &path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open " + path.string());
    }

    CsvTable table;
    if (!std::getline(input, table.header_line)) {
        throw std::runtime_error("empty CSV: " + path.string());
    }
    if (!table.header_line.empty() && table.header_line.back() == '\r') {
        table.header_line.pop_back();
    }
    table.header = splitCsvLine(table.header_line);

    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        table.fields.push_back(splitCsvLine(line));
    }
    return table;
}

void writeFilteredCsv(
    const fs::path &path,
    const CsvTable &table,
    const std::vector<bool> &keep_mask,
    const std::vector<size_t> &source_row_indices) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to write " + path.string());
    }

    const PairColumns columns = getPairColumns(table, false);
    const auto compactId = [](std::unordered_map<long long, int> &index_map, long long external_index) {
        const auto it = index_map.find(external_index);
        if (it != index_map.end()) {
            return it->second;
        }
        const int compact = static_cast<int>(index_map.size());
        index_map.emplace(external_index, compact);
        return compact;
    };

    std::unordered_map<long long, int> left_indices;
    std::unordered_map<long long, int> right_indices;
    for (size_t i = 0; i < keep_mask.size(); ++i) {
        if (keep_mask[i]) {
            const std::vector<std::string> &row = table.fields[source_row_indices[i]];
            compactId(left_indices, parseInteger(row, columns.left_idx));
            compactId(right_indices, parseInteger(row, columns.right_idx));
        }
    }

    output << table.header_line << '\n';
    for (size_t i = 0; i < keep_mask.size(); ++i) {
        if (keep_mask[i]) {
            std::vector<std::string> row = table.fields[source_row_indices[i]];
            row[static_cast<size_t>(columns.left_idx)] = std::to_string(
                left_indices.at(parseInteger(row, columns.left_idx)));
            row[static_cast<size_t>(columns.right_idx)] = std::to_string(
                right_indices.at(parseInteger(row, columns.right_idx)));
            // Only compact feature IDs; pixel coordinate fields remain in the original image frame.
            writeCsvRow(output, row);
        }
    }
}

int compactIndex(
    std::unordered_map<long long, int> &index_map,
    std::vector<KeyPoint> &keypoints,
    long long external_index,
    double x,
    double y) {
    const auto it = index_map.find(external_index);
    if (it != index_map.end()) {
        return it->second;
    }
    const int compact = static_cast<int>(keypoints.size());
    index_map.emplace(external_index, compact);
    keypoints.emplace_back(Point2f(static_cast<float>(x), static_cast<float>(y)), 1.0f);
    return compact;
}

int pairIndexFromMatchingFilename(const fs::path &path) {
    const std::regex pattern(R"(^matching_(\d+)\.csv$)");
    std::smatch match;
    const std::string name = path.filename().string();
    return std::regex_match(name, match, pattern) ? std::stoi(match[1].str()) : -1;
}

std::unordered_map<int, ImgPairSizes> loadImgPairSizes(const fs::path &pose_path) {
    std::unordered_map<int, ImgPairSizes> sizes;
    if (!fs::is_regular_file(pose_path)) {
        return sizes;
    }

    CsvTable table = readCsv(pose_path);
    const int pair_col = columnIndex(table.header, "pair_idx");
    const int cx1_col = columnIndex(table.header, "cx1");
    const int cy1_col = columnIndex(table.header, "cy1");
    const int cx2_col = columnIndex(table.header, "cx2");
    const int cy2_col = columnIndex(table.header, "cy2");
    if (pair_col < 0 || cx1_col < 0 || cy1_col < 0 || cx2_col < 0 || cy2_col < 0) {
        return sizes;
    }

    for (const auto &row : table.fields) {
        try {
            ImgPairSizes pose_sizes;
            const int pair_idx = static_cast<int>(parseInteger(row, pair_col));
            pose_sizes.left.width = std::max(1, static_cast<int>(std::lround(2.0 * parseDouble(row, cx1_col))));
            pose_sizes.left.height = std::max(1, static_cast<int>(std::lround(2.0 * parseDouble(row, cy1_col))));
            pose_sizes.right.width = std::max(1, static_cast<int>(std::lround(2.0 * parseDouble(row, cx2_col))));
            pose_sizes.right.height = std::max(1, static_cast<int>(std::lround(2.0 * parseDouble(row, cy2_col))));
            sizes[pair_idx] = pose_sizes;
        } catch (const std::exception &) {
            continue;
        }
    }
    return sizes;
}

PreparedMatches prepareMatches(const CsvTable &table, const ImgPairSizes &pose_sizes, bool auto_mask, int grid_size) {
    const PairColumns columns = getPairColumns(table);
    PreparedMatches prepared;
    std::vector<ValidRow> valid_rows;
    KeypointBounds left_bounds;
    KeypointBounds right_bounds;

    for (size_t row_index = 0; row_index < table.fields.size(); ++row_index) {
        const auto &row = table.fields[row_index];
        try {
            ValidRow valid;
            valid.source_row = row_index;
            valid.left_idx = parseInteger(row, columns.left_idx);
            valid.right_idx = parseInteger(row, columns.right_idx);
            valid.x1 = parseDouble(row, columns.x1);
            valid.y1 = parseDouble(row, columns.y1);
            valid.x2 = parseDouble(row, columns.x2);
            valid.y2 = parseDouble(row, columns.y2);
            if (!std::isfinite(valid.x1) || !std::isfinite(valid.y1) ||
                !std::isfinite(valid.x2) || !std::isfinite(valid.y2) ||
                valid.x1 < 0.0 || valid.y1 < 0.0 || valid.x2 < 0.0 || valid.y2 < 0.0) {
                ++prepared.skipped_rows;
                continue;
            }

            left_bounds.add(valid.x1, valid.y1);
            right_bounds.add(valid.x2, valid.y2);
            valid_rows.push_back(valid);
        } catch (const std::exception &) {
            ++prepared.skipped_rows;
        }
    }

    if (valid_rows.empty()) {
        return prepared;
    }

    // Adaptive margin: large enough so that the half-cell shift in GetGridIndexLeft
    // (types 2/3/4 add 0.5 in grid units) never pushes the nearest data point
    // outside the valid range.  Derived from: margin >= span / (2*(W-1)).
    const double max_span = std::max({
        left_bounds.max_x  - left_bounds.min_x,
        left_bounds.max_y  - left_bounds.min_y,
        right_bounds.max_x - right_bounds.min_x,
        right_bounds.max_y - right_bounds.min_y
    });
    const double margin = std::ceil(max_span / (2.0 * (grid_size - 1)));
    const double left_origin_x = auto_mask ? left_bounds.min_x - margin : 0.0;
    const double left_origin_y = auto_mask ? left_bounds.min_y - margin : 0.0;
    const double right_origin_x = auto_mask ? right_bounds.min_x - margin : 0.0;
    const double right_origin_y = auto_mask ? right_bounds.min_y - margin : 0.0;

    const ImageSize left_extent = left_bounds.extentSize();
    const ImageSize right_extent = right_bounds.extentSize();
    prepared.left_size = auto_mask
        ? left_bounds.maskSize(margin)
        : ImageSize{
            std::max(pose_sizes.left.width, left_extent.width),
            std::max(pose_sizes.left.height, left_extent.height)};
    prepared.right_size = auto_mask
        ? right_bounds.maskSize(margin)
        : ImageSize{
            std::max(pose_sizes.right.width, right_extent.width),
            std::max(pose_sizes.right.height, right_extent.height)};

    std::unordered_map<long long, int> left_indices;
    std::unordered_map<long long, int> right_indices;

    for (const ValidRow &row : valid_rows) {
        const int left_compact = compactIndex(
            left_indices,
            prepared.left_keypoints,
            row.left_idx,
            row.x1 - left_origin_x,
            row.y1 - left_origin_y);

        const int right_compact = compactIndex(
            right_indices,
            prepared.right_keypoints,
            row.right_idx,
            row.x2 - right_origin_x,
            row.y2 - right_origin_y);

        prepared.matches.emplace_back(left_compact, right_compact, 0.0f);
        prepared.source_row_indices.push_back(row.source_row);
    }

    return prepared;
}

std::vector<fs::path> findMatchingFiles(const fs::path &input_dir) {
    std::vector<fs::path> files;
    for (const fs::directory_entry &entry : fs::directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.rfind("matching_", 0) == 0 && entry.path().extension() == ".csv") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::string thresholdTag(double threshold_factor) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(6) << threshold_factor;

    std::string value = stream.str();
    while (value.find('.') != std::string::npos && value.back() == '0' && value[value.size() - 2] != '.') {
        value.pop_back();
    }
    return "ThrFact_" + value;
}

Options parseArgs(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--root" && i + 1 < argc) {
            options.root = fs::path(argv[++i]);
        } else if (arg == "--threshold-factor" && i + 1 < argc) {
            char *end = nullptr;
            const char *value = argv[++i];
            options.threshold_factor = std::strtod(value, &end);
            if (end == value || *end != '\0') {
                throw std::runtime_error("--threshold-factor must be a double number");
            }
        } else if ((arg == "--auto-mask" || arg == "--auto_mask") && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "0") {
                options.auto_mask = false;
            } else if (value == "1") {
                options.auto_mask = true;
            } else {
                throw std::runtime_error("--auto-mask must be 0 or 1");
            }
        } else if (arg == "--grid-size" && i + 1 < argc) {
            char *end = nullptr;
            const char *value = argv[++i];
            options.grid_size = static_cast<int>(std::strtol(value, &end, 10));
            if (end == value || *end != '\0' || options.grid_size < 2) {
                throw std::runtime_error("--grid-size must be >1");
            }
        } else if (arg == "--with-scale") {
            options.with_scale = true;
        } else if (arg == "--with-rotation") {
            options.with_rotation = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: gms_filter_csv_m2m --root INPUT_DIR [--threshold-factor 6.0] [--grid-size 20] "
                   "[--auto-mask 0|1] [--with-scale] [--with-rotation]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    if (options.root.empty()) {
        throw std::runtime_error("--root INPUT_DIR is required");
    }
    if (!(options.threshold_factor > 0.0) || !std::isfinite(options.threshold_factor)) {
        throw std::runtime_error("--threshold-factor must be a positive finite number");
    }
    return options;
}

int filterOneFile(
    const fs::path &csv_path,
    const fs::path &output_path,
    const ImgPairSizes &pose_sizes,
    const Options &options) {

    CsvTable table = readCsv(csv_path);
    PreparedMatches prepared = prepareMatches(table, pose_sizes, options.auto_mask, options.grid_size);

    std::vector<bool> inliers;
    int num_inliers = 0;
    if (!prepared.matches.empty()) {
        gms_matcher_m2m gms(
            prepared.left_keypoints,
            Size(prepared.left_size.width, prepared.left_size.height),
            prepared.right_keypoints,
            Size(prepared.right_size.width, prepared.right_size.height),
            prepared.matches,
            options.threshold_factor,
            options.grid_size);
        num_inliers = gms.GetInlierMask(inliers, options.with_scale, options.with_rotation);
    }

    fs::create_directories(output_path.parent_path());
    writeFilteredCsv(output_path, table, inliers, prepared.source_row_indices);

    std::cout << "[GMS(m2m)] "
              << " valid=" << prepared.matches.size()
              << " skipped=" << prepared.skipped_rows
              << " inliers=" << num_inliers << '\n';
            //   << " auto_mask=" << std::boolalpha << options.auto_mask
            //   << " threshold_factor=" << options.threshold_factor << '\n';
    return 1;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        Options options = parseArgs(argc, argv);
        const fs::path input_dir = fs::absolute(options.root);
        if (!fs::is_directory(input_dir)) {
            throw std::runtime_error("input folder does not exist: " + input_dir.string());
        }

        const fs::path dataset_dir = input_dir.parent_path().parent_path();
        const auto img_pair_size = options.auto_mask
            ? std::unordered_map<int, ImgPairSizes>{}
            : loadImgPairSizes(dataset_dir / "pose_intrinsics.csv");
        const std::string auto_mask_tag = options.auto_mask ? "_auto_mask" : "";
        const fs::path output_dir = input_dir.parent_path() / (
            input_dir.filename().string() + "_GMSm2m_" + thresholdTag(options.threshold_factor) + "_Gridsz_" + std::to_string(options.grid_size) + auto_mask_tag
        );

        int processed = 0;
        for (const fs::path &csv_path : findMatchingFiles(input_dir)) {
            ImgPairSizes pose_sizes;
            const int pair_idx = pairIndexFromMatchingFilename(csv_path);
            const auto temp = img_pair_size.find(pair_idx);
            // if image pair size is missing, use:
            // width  = floor(max_x) + 1, height = floor(max_y) + 1
            if (temp != img_pair_size.end()) {
                pose_sizes = temp->second;
            }
            processed += filterOneFile(
                csv_path,
                output_dir / csv_path.filename(),
                pose_sizes,
                options);
        }

        const fs::path association_manifest = input_dir / "association_manifest.json";
        if (fs::is_regular_file(association_manifest)) {
            fs::create_directories(output_dir);
            fs::copy_file(
                association_manifest,
                output_dir / association_manifest.filename(),
                fs::copy_options::overwrite_existing);
        }

        std::cout << "[SUMMARY] input_dir=" << input_dir
                  << " output_dir=" << output_dir
                  << " files=" << processed
                  << " threshold_factor=" << options.threshold_factor
                  << " grid_size=" << options.grid_size
                  << " auto_mask=" << std::boolalpha << options.auto_mask
                  << " with_scale=" << std::boolalpha << options.with_scale
                  << " with_rotation=" << options.with_rotation << '\n';
    } catch (const std::exception &exc) {
        std::cerr << "[ERROR] " << exc.what() << '\n';
        return 1;
    }
    return 0;
}
