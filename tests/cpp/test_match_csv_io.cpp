#include "m2m_loransac/io.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

void write_text(const fs::path &path, const std::string &payload) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("could not create test input " + path.string());
    }
    output << payload;
    if (!output) {
        throw std::runtime_error("could not write test input " + path.string());
    }
}

void expect_invalid(const fs::path &root, const std::string &name,
                    const std::string &payload, const std::string &needle) {
    const fs::path path = root / (name + ".csv");
    write_text(path, payload);
    try {
        (void)loransac_app::load_match_data(path, true);
    } catch (const std::runtime_error &error) {
        const std::string message = error.what();
        if (message.find(needle) == std::string::npos ||
            message.find(path.string()) == std::string::npos) {
            throw std::runtime_error(name + " produced an uninformative error: " + message);
        }
        return;
    }
    throw std::runtime_error(name + " was silently accepted");
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("expected one test-root argument");
        }
        const fs::path root = argv[1];
        fs::remove_all(root);
        fs::create_directories(root);

        const std::string header = "left_idx,right_idx,x1,y1,x2,y2,similarity,k_first\n";
        const fs::path valid_path = root / "valid.csv";
        write_text(valid_path, header +
                                   "0,1, 1.25 ,-0,2e1,+3.5,0.75,1\n"
                                   "\n"
                                   "2,3,4,5,6,7,0,5\n");
        const loransac_app::MatchData valid =
            loransac_app::load_match_data(valid_path, true);
        if (valid.points1.size() != 2 || valid.points2.size() != 2 ||
            valid.idx1.size() != 2 || valid.idx2.size() != 2 ||
            valid.similarity.size() != 2 || valid.first_k.size() != 2 ||
            !valid.has_m2m_metadata || !valid.has_similarity || !valid.has_first_k ||
            std::abs(valid.points1[0](0) - 1.25) > 1e-12 ||
            std::abs(valid.points2[0](1) - 3.5) > 1e-12 ||
            std::abs(valid.similarity[0] - 0.75) > 1e-12 ||
            valid.first_k[1] != 5) {
            throw std::runtime_error("valid CSV behavior changed");
        }

        expect_invalid(root, "short_row", header + "0,1,2\n",
                       "Invalid column count on line 2");
        expect_invalid(root, "extra_column", header + "0,1,2,3,4,5,0.9,1,extra\n",
                       "Invalid column count on line 2");
        expect_invalid(root, "non_numeric_coordinate", header + "0,1,oops,3,4,5,0.9,1\n",
                       "Invalid x1 on line 2");
        expect_invalid(root, "trailing_coordinate_junk", header + "0,1,2junk,3,4,5,0.9,1\n",
                       "Invalid x1 on line 2");
        expect_invalid(root, "nan_coordinate", header + "0,1,2,nan,4,5,0.9,1\n",
                       "Invalid y1 on line 2");
        expect_invalid(root, "infinite_coordinate", header + "0,1,2,3,inf,5,0.9,1\n",
                       "Invalid x2 on line 2");
        expect_invalid(root, "empty_coordinate", header + "0,1,2,3,4,,0.9,1\n",
                       "Invalid y2 on line 2");
        expect_invalid(root, "non_numeric_similarity", header + "0,1,2,3,4,5,bad,1\n",
                       "Invalid similarity on line 2");
        expect_invalid(root, "nan_similarity", header + "0,1,2,3,4,5,nan,1\n",
                       "Invalid similarity on line 2");
        expect_invalid(root, "infinite_similarity", header + "0,1,2,3,4,5,-inf,1\n",
                       "Invalid similarity on line 2");
        expect_invalid(root, "late_bad_row",
                       header + "0,1,2,3,4,5,0.9,1\n1,2,3,4,5,6,bad,2\n",
                       "Invalid similarity on line 3");

        const fs::path no_similarity_path = root / "no_similarity.csv";
        write_text(no_similarity_path,
                   "left_idx,right_idx,x1,y1,x2,y2\n0,1,2,3,4,5\n");
        const loransac_app::MatchData no_similarity =
            loransac_app::load_match_data(no_similarity_path, false);
        if (no_similarity.points1.size() != 1 || no_similarity.has_similarity ||
            no_similarity.has_first_k) {
            throw std::runtime_error("legal CSV without optional columns changed");
        }
    } catch (const std::exception &error) {
        std::cerr << "test_match_csv_io: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
