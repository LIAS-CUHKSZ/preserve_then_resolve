#include "m2m_loransac/hopcroft_karp.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <stdexcept>

namespace dino_m2m {
namespace {

constexpr std::ptrdiff_t kUnmatched = -1;
constexpr std::size_t kInfiniteDistance = std::numeric_limits<std::size_t>::max();

}  // namespace

HopcroftKarp::HopcroftKarp(std::size_t left_vertex_count, std::size_t right_vertex_count)
    : left_vertex_count_(left_vertex_count), right_vertex_count_(right_vertex_count),
      adjacency_(left_vertex_count), left_match_(left_vertex_count, kUnmatched),
      right_match_(right_vertex_count, kUnmatched), distance_(left_vertex_count, kInfiniteDistance) {}

void HopcroftKarp::add_edge(std::size_t left, std::size_t right) {
    if (left >= left_vertex_count_ || right >= right_vertex_count_) {
        throw std::out_of_range("HopcroftKarp edge endpoint is outside the graph");
    }
    adjacency_[left].push_back(right);
}

std::size_t HopcroftKarp::maximum_matching() {
    std::fill(left_match_.begin(), left_match_.end(), kUnmatched);
    std::fill(right_match_.begin(), right_match_.end(), kUnmatched);

    std::size_t cardinality = 0;
    while (build_layers()) {
        for (std::size_t left = 0; left < left_vertex_count_; ++left) {
            if (left_match_[left] == kUnmatched && find_augmenting_path(left)) {
                ++cardinality;
            }
        }
    }
    return cardinality;
}

std::vector<std::pair<std::size_t, std::size_t>> HopcroftKarp::matched_pairs() const {
    std::vector<std::pair<std::size_t, std::size_t>> result;
    result.reserve(left_vertex_count_);
    for (std::size_t left = 0; left < left_vertex_count_; ++left) {
        if (left_match_[left] != kUnmatched) {
            result.emplace_back(left, static_cast<std::size_t>(left_match_[left]));
        }
    }
    return result;
}

bool HopcroftKarp::build_layers() {
    std::queue<std::size_t> frontier;
    for (std::size_t left = 0; left < left_vertex_count_; ++left) {
        if (left_match_[left] == kUnmatched) {
            distance_[left] = 0;
            frontier.push(left);
        } else {
            distance_[left] = kInfiniteDistance;
        }
    }

    bool found_augmenting_path = false;
    while (!frontier.empty()) {
        const std::size_t left = frontier.front();
        frontier.pop();
        for (const std::size_t right : adjacency_[left]) {
            const std::ptrdiff_t next_left = right_match_[right];
            if (next_left == kUnmatched) {
                found_augmenting_path = true;
                continue;
            }
            const std::size_t next = static_cast<std::size_t>(next_left);
            if (distance_[next] == kInfiniteDistance) {
                distance_[next] = distance_[left] + 1;
                frontier.push(next);
            }
        }
    }
    return found_augmenting_path;
}

bool HopcroftKarp::find_augmenting_path(std::size_t left) {
    for (const std::size_t right : adjacency_[left]) {
        const std::ptrdiff_t next_left = right_match_[right];
        if (next_left == kUnmatched ||
            (distance_[static_cast<std::size_t>(next_left)] == distance_[left] + 1 &&
             find_augmenting_path(static_cast<std::size_t>(next_left)))) {
            left_match_[left] = static_cast<std::ptrdiff_t>(right);
            right_match_[right] = static_cast<std::ptrdiff_t>(left);
            return true;
        }
    }
    distance_[left] = kInfiniteDistance;
    return false;
}

}  // namespace dino_m2m
