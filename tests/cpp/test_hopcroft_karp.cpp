#include "m2m_loransac/hopcroft_karp.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool augment(std::size_t left, const std::vector<std::vector<std::size_t>> &adjacency,
             std::vector<int> *right_match, std::vector<char> *seen) {
    for (const std::size_t right : adjacency[left]) {
        if ((*seen)[right]) {
            continue;
        }
        (*seen)[right] = 1;
        if ((*right_match)[right] < 0 ||
            augment(static_cast<std::size_t>((*right_match)[right]), adjacency, right_match, seen)) {
            (*right_match)[right] = static_cast<int>(left);
            return true;
        }
    }
    return false;
}

std::size_t reference_cardinality(const std::vector<std::vector<std::size_t>> &adjacency,
                                  std::size_t right_vertex_count) {
    std::vector<int> right_match(right_vertex_count, -1);
    std::size_t cardinality = 0;
    for (std::size_t left = 0; left < adjacency.size(); ++left) {
        std::vector<char> seen(right_vertex_count, 0);
        cardinality += augment(left, adjacency, &right_match, &seen) ? 1 : 0;
    }
    return cardinality;
}

void add_edges(dino_m2m::HopcroftKarp *graph,
               const std::vector<std::vector<std::size_t>> &adjacency) {
    for (std::size_t left = 0; left < adjacency.size(); ++left) {
        for (const std::size_t right : adjacency[left]) {
            graph->add_edge(left, right);
        }
    }
}

}  // namespace

int main() {
    {
        dino_m2m::HopcroftKarp graph(0, 0);
        require(graph.maximum_matching() == 0, "empty graph cardinality");
        require(graph.matched_pairs().empty(), "empty graph pairs");
    }
    {
        // The third edge requires an alternating augmenting path.
        const std::vector<std::vector<std::size_t>> adjacency{{0, 1}, {0}, {1, 2}};
        dino_m2m::HopcroftKarp graph(3, 3);
        add_edges(&graph, adjacency);
        require(graph.maximum_matching() == 3, "augmenting-path cardinality");
        require(graph.matched_pairs() ==
                    (std::vector<std::pair<std::size_t, std::size_t>>{{0, 1}, {1, 0}, {2, 2}}),
                "deterministic matching");
        // Re-running must reset state and produce the same deterministic result.
        require(graph.maximum_matching() == 3, "repeat cardinality");
        require(graph.matched_pairs() ==
                    (std::vector<std::pair<std::size_t, std::size_t>>{{0, 1}, {1, 0}, {2, 2}}),
                "repeat deterministic matching");
    }
    {
        dino_m2m::HopcroftKarp graph(1, 1);
        graph.add_edge(0, 0);
        graph.add_edge(0, 0);
        require(graph.maximum_matching() == 1, "duplicate edge cardinality");
        bool threw = false;
        try {
            graph.add_edge(1, 0);
        } catch (const std::out_of_range &) {
            threw = true;
        }
        require(threw, "out-of-range edge guard");
    }

    std::mt19937 rng(20260801U);
    std::bernoulli_distribution has_edge(0.28);
    for (std::size_t trial = 0; trial < 500; ++trial) {
        const std::size_t left_count = trial % 11;
        const std::size_t right_count = (trial * 7) % 13;
        std::vector<std::vector<std::size_t>> adjacency(left_count);
        dino_m2m::HopcroftKarp graph(left_count, right_count);
        for (std::size_t left = 0; left < left_count; ++left) {
            for (std::size_t right = 0; right < right_count; ++right) {
                if (has_edge(rng)) {
                    adjacency[left].push_back(right);
                    graph.add_edge(left, right);
                }
            }
        }
        require(graph.maximum_matching() == reference_cardinality(adjacency, right_count),
                "random graph reference parity");
    }

    std::cout << "Hopcroft-Karp tests passed\n";
    return 0;
}
