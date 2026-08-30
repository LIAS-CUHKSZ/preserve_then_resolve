#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace dino_m2m {

// Maximum-cardinality matching for a zero-based bipartite graph.
//
// Vertices and adjacency lists are visited in insertion order. This keeps the
// selected matching deterministic while leaving the matching cardinality
// independent of edge order.
class HopcroftKarp {
  public:
    HopcroftKarp(std::size_t left_vertex_count, std::size_t right_vertex_count);

    void add_edge(std::size_t left, std::size_t right);
    std::size_t maximum_matching();

    // Pairs are zero-based and sorted by their left vertex.
    std::vector<std::pair<std::size_t, std::size_t>> matched_pairs() const;

  private:
    bool build_layers();
    bool find_augmenting_path(std::size_t left);

    std::size_t left_vertex_count_;
    std::size_t right_vertex_count_;
    std::vector<std::vector<std::size_t>> adjacency_;
    std::vector<std::ptrdiff_t> left_match_;
    std::vector<std::ptrdiff_t> right_match_;
    std::vector<std::size_t> distance_;
};

}  // namespace dino_m2m
