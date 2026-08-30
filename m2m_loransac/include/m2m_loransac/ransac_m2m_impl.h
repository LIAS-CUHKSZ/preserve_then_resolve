// Copyright (c) 2021, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "ransac_mode.h"
#include "PoseLib/camera_pose.h"
#include "PoseLib/types.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace loransac_app {

// Solver concept (M2M estimators must implement):
//   void generate_models(std::vector<Model> *models);
//   ModelScore score_model(const Model &model) const;
//   void refine_model(Model *model) const;
//   RansacMode ransac_mode() const;
//   size_t top_n_candidates() const;  // >0 enables HCM_MC candidate pool
//   double pool_dedup_deg() const;    // >0 enables pool deduplication (HCM_MC)
//   void finalize_candidate_pool(CandidatePool *pool, Model *best_model,
//                                poselib::RansacStats *stats,
//                                M2MTimingStats *timing) const;  // HCM_MC only
//   const size_t sample_sz;
//   const size_t num_data;
//   const size_t num_o2o_association_ub;

// Wall-clock decomposition of one ransac() call, for timing experiments that
// must separate the scoring-mechanism cost (HCM vs MCM per-evaluation) from the
// local-refinement policy (gated in-loop for CM/HCM/MCM vs fixed-N_c finalize
// for HCM_MC) and from the adaptive iteration count.
struct M2MTimingStats {
    uint64_t solve_ns = 0;        // sampling + minimal solver (generate_models)
    uint64_t score_ns = 0;        // mode's own score_model calls (incl. post-refine re-scores)
    uint64_t score_calls = 0;
    uint64_t refine_ns = 0;       // all refine_model calls (gated in-loop or pool finalize)
    uint64_t refine_calls = 0;
    uint64_t rank_score_ns = 0;   // HCM_MC only: MCM re-scoring of refined pool candidates
    uint64_t rank_score_calls = 0;
    uint64_t total_ns = 0;        // whole ransac() call
    // Pool-diversity diagnostics (HCM_MC only). Duplicates are counted at the
    // measurement threshold even when behavioral dedup is disabled.
    uint64_t pool_insert_attempts = 0; // try_insert calls (already passed the score gate)
    uint64_t pool_dup_hits = 0;        // attempts that matched an existing pool entry
    uint64_t pool_unique_basins = 0;   // distinct basins among refined candidates (finalize)
};

// Two relative poses describe the same model up to noise? Rotation via the
// absolute quaternion dot (handles the double cover), translation via the
// absolute direction dot (Sampson scoring depends only on E=[t]xR, which is
// invariant to t -> -t). Drop the abs on translation if the pool ever becomes
// cheirality-sensitive.
inline bool poses_similar(const poselib::CameraPose &a, const poselib::CameraPose &b, double cos_half_rot,
                          double cos_trans) {
    if (std::abs(a.q.dot(b.q)) < cos_half_rot) {
        return false;
    }
    const double denom = a.t.norm() * b.t.norm();
    if (denom < 1e-18) {
        return true; // both translations ~zero: rotation match decides
    }
    return std::abs(a.t.dot(b.t)) / denom >= cos_trans;
}

struct ModelScore {
    double score = std::numeric_limits<double>::max();
    size_t edge_inliers = 0;
    size_t cardinality = 0;
    std::vector<size_t> inlier_idx1;
    std::vector<size_t> inlier_idx2;
};

struct Candidate {
    poselib::CameraPose pose;
    double hcm_score = std::numeric_limits<double>::max();
    size_t edge_inliers = 0;
    size_t cardinality = 0;
    std::vector<size_t> inlier_idx1;
    std::vector<size_t> inlier_idx2;
};

// Read-only snapshot of the bounded HCM raw-seed pool.  The snapshot is
// captured after Stage 1 and before Stage-2 refinement mutates the retained
// candidates.  It deliberately contains no pose supplied by a diagnostic
// caller, so taking the snapshot cannot change pool admission or ranking.
struct HCMCandidatePoolSnapshot {
    bool captured = false;
    size_t capacity = 0;
    size_t size = 0;
    bool full = false;
    double admission_cutoff_score = std::numeric_limits<double>::max();
};

class CandidatePool {
  public:
    CandidatePool() = default;
    explicit CandidatePool(size_t capacity) : capacity_(capacity) {}

    // dedup_deg > 0 enables behavioral deduplication at that angular threshold
    // (rotation and translation direction). With dedup_deg == 0 the pool
    // behaves exactly as before, but duplicates are still COUNTED at a fixed
    // 1-degree measurement threshold for diagnostics.
    void reset(size_t capacity, double dedup_deg = 0.0) {
        capacity_ = capacity;
        candidates_.clear();
        dedup_enabled_ = dedup_deg > 0.0;
        const double measure_deg = dedup_enabled_ ? dedup_deg : 1.0;
        constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
        cos_half_rot_ = std::cos(0.5 * measure_deg * kDegToRad);
        cos_trans_ = std::cos(measure_deg * kDegToRad);
        insert_attempts_ = 0;
        dup_hits_ = 0;
    }

    bool enabled() const { return capacity_ > 0; }
    bool empty() const { return candidates_.empty(); }
    size_t size() const { return candidates_.size(); }
    size_t capacity() const { return capacity_; }
    std::vector<Candidate> &entries() { return candidates_; }
    const std::vector<Candidate> &entries() const { return candidates_; }
    bool dedup_enabled() const { return dedup_enabled_; }
    uint64_t insert_attempts() const { return insert_attempts_; }
    uint64_t dup_hits() const { return dup_hits_; }

    // Index of an entry whose pose matches `pose` at the measurement
    // threshold, or -1 if none.
    int find_similar(const poselib::CameraPose &pose) const {
        for (size_t i = 0; i < candidates_.size(); ++i) {
            if (poses_similar(pose, candidates_[i].pose, cos_half_rot_, cos_trans_)) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    // Greedy count of distinct basins among current entries (diagnostic; used
    // after stage-2 refinement has pulled every candidate to its basin).
    size_t count_unique_basins() const {
        size_t unique = 0;
        for (size_t i = 0; i < candidates_.size(); ++i) {
            bool dup = false;
            for (size_t k = 0; k < i && !dup; ++k) {
                dup = poses_similar(candidates_[i].pose, candidates_[k].pose, cos_half_rot_, cos_trans_);
            }
            unique += dup ? 0 : 1;
        }
        return unique;
    }

    // Would a candidate with this hcm_score earn a slot in the current top-N?
    // Mirrors try_insert's admission test without mutating, so pool admission
    // can be decided independently of the monotone local-optimization gate.
    bool would_accept(double hcm_score) const {
        if (!enabled()) {
            return false;
        }
        if (candidates_.size() < capacity_) {
            return true;
        }
        const auto worst_it = std::max_element(
            candidates_.begin(), candidates_.end(),
            [](const Candidate &a, const Candidate &b) { return a.hcm_score < b.hcm_score; });
        return hcm_score < worst_it->hcm_score;
    }

    // Return the strict admission boundary used by would_accept().  An
    // under-filled pool accepts every finite score, represented by +infinity.
    double admission_cutoff_score() const {
        if (!enabled() || candidates_.size() < capacity_) {
            return std::numeric_limits<double>::max();
        }
        const auto worst_it = std::max_element(
            candidates_.begin(), candidates_.end(),
            [](const Candidate &a, const Candidate &b) { return a.hcm_score < b.hcm_score; });
        return worst_it->hcm_score;
    }

    bool try_insert(Candidate candidate) {
        if (!enabled()) {
            return false;
        }
        insert_attempts_++;
        // Duplicate scan (O(N_c), a few ns per entry). Counted always; acted
        // on only when dedup is enabled: the basin keeps its better seed and
        // never holds two representatives.
        const int similar = find_similar(candidate.pose);
        if (similar >= 0) {
            dup_hits_++;
            if (dedup_enabled_) {
                Candidate &clone = candidates_[static_cast<size_t>(similar)];
                if (candidate.hcm_score < clone.hcm_score) {
                    clone = std::move(candidate);
                    return true;
                }
                return false;
            }
        }
        if (candidates_.size() < capacity_) {
            candidates_.push_back(std::move(candidate));
            return true;
        }
        const auto worst_it = std::max_element(
            candidates_.begin(), candidates_.end(),
            [](const Candidate &a, const Candidate &b) { return a.hcm_score < b.hcm_score; });
        if (candidate.hcm_score < worst_it->hcm_score) {
            *worst_it = std::move(candidate);
            return true;
        }
        return false;
    }

  private:
    size_t capacity_ = 0;
    std::vector<Candidate> candidates_;
    bool dedup_enabled_ = false;
    double cos_half_rot_ = 1.0;
    double cos_trans_ = 1.0;
    uint64_t insert_attempts_ = 0;
    uint64_t dup_hits_ = 0;
};

struct RansacState {
    size_t best_minimal_edge_inliers = 0;
    size_t best_minimal_cardinality = 0;
    double best_minimal_score = std::numeric_limits<double>::max();
    size_t dynamic_max_iter = 100000;
    double log_prob_missing_model = std::log(1.0 - 0.9999);
    CandidatePool candidate_pool;
    M2MTimingStats timing;
};

namespace detail {

inline uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

inline double all_inlier_sample_probability(size_t num_inliers, size_t num_data, size_t sample_sz) {
    if (sample_sz == 0) {
        return 1.0;
    }
    if (num_inliers < sample_sz || num_data < sample_sz) {
        return 0.0;
    }

    double prob_all_inliers = 1.0;
    for (size_t i = 0; i < sample_sz; ++i) {
        prob_all_inliers *= static_cast<double>(num_inliers - i) / static_cast<double>(num_data - i);
    }
    return prob_all_inliers;
}

inline size_t compute_dynamic_max_iter(size_t num_inliers, size_t num_data, size_t sample_sz,
                                       double log_prob_missing_model, double dyn_num_trials_mult, size_t min_iterations,
                                       size_t max_iterations) {
    const double prob_all_inliers = all_inlier_sample_probability(num_inliers, num_data, sample_sz);
    if (prob_all_inliers >= 0.9999999) {
        return min_iterations;
    }
    if (prob_all_inliers <= 0.0000001) {
        return max_iterations;
    }

    const double prob_outlier = 1.0 - prob_all_inliers;
    const size_t num_iters =
        static_cast<size_t>(std::ceil(log_prob_missing_model / std::log(prob_outlier) * dyn_num_trials_mult));
    return std::max(min_iterations, std::min(max_iterations, num_iters));
}

inline void update_dynamic_max_iter(const poselib::RansacOptions &opt, RansacState &state, poselib::RansacStats &stats,
                                    size_t num_data, size_t sample_sz) {
    stats.inlier_ratio = static_cast<double>(stats.num_inliers) / static_cast<double>(num_data);
    const auto temp = state.dynamic_max_iter;
    state.dynamic_max_iter = compute_dynamic_max_iter(stats.num_inliers, num_data, sample_sz, state.log_prob_missing_model,
                                                      opt.dyn_num_trials_mult, opt.min_iterations, opt.max_iterations);
#if M2M_VERBOSE
    if (temp != state.dynamic_max_iter) {
        std::cout << "dynamic_max_iter changed from " << temp << " to " << state.dynamic_max_iter << std::endl;
    }
#else
    (void)temp;
#endif
}

inline bool triggers_local_optimization(RansacMode mode, const ModelScore &cur, const RansacState &state) {
    const bool better_score = cur.score < state.best_minimal_score;
    const bool larger_cardinality = cur.cardinality >= state.best_minimal_cardinality;
    const bool more_edge_inliers = cur.edge_inliers >= state.best_minimal_edge_inliers;
    switch (mode) {
    case RansacMode::MCM:
        return larger_cardinality;
    case RansacMode::HCM:
    case RansacMode::HCM_MC:
        return better_score || more_edge_inliers;
    default: // CM
        return more_edge_inliers || better_score;
    }
}

inline void update_minimal_state(RansacMode mode, const ModelScore &cur, RansacState &state) {
    (void)mode;
    if (cur.edge_inliers > state.best_minimal_edge_inliers) {
        state.best_minimal_edge_inliers = cur.edge_inliers;
    }
    if (cur.cardinality > state.best_minimal_cardinality) {
        state.best_minimal_cardinality = cur.cardinality;
    }
    if (cur.score < state.best_minimal_score) {
        state.best_minimal_score = cur.score;
    }
}

inline bool is_better_batch_candidate(RansacMode mode, const ModelScore &cur, const ModelScore &best) {
    switch (mode) {
    case RansacMode::MCM:
        return cur.cardinality > best.cardinality;
    case RansacMode::HCM:
    case RansacMode::HCM_MC:
        return cur.score < best.score;
    default: // CM
        if (cur.edge_inliers != best.edge_inliers) {
            return cur.edge_inliers > best.edge_inliers;
        }
        return cur.score < best.score;
    }
}

inline Candidate candidate_from_score(const poselib::CameraPose &pose, const ModelScore &score) {
    Candidate candidate;
    candidate.pose = pose;
    candidate.hcm_score = score.score;
    candidate.edge_inliers = score.edge_inliers;
    candidate.inlier_idx1 = score.inlier_idx1;
    candidate.inlier_idx2 = score.inlier_idx2;
    return candidate;
}

} // namespace detail

template <typename Solver, typename Model = poselib::CameraPose>
void score_models(const Solver &estimator, const std::vector<Model> &models, const poselib::RansacOptions &opt,
                  RansacState &state, poselib::RansacStats &stats, Model *best_model) {
    if (models.empty()) {
        return;
    }

    int best_model_ind = -1;
    ModelScore best_batch_score;
    for (size_t i = 0; i < models.size(); ++i) {
        // Keep candidate comparison and ModelScore vector copies outside the
        // scorer timer so HCM/MCM per-evaluation costs remain comparable.
        const uint64_t score_start = detail::now_ns();
        const ModelScore m = estimator.score_model(models[i]);
        state.timing.score_ns += detail::now_ns() - score_start;
        state.timing.score_calls++;
        if (best_model_ind == -1 || detail::is_better_batch_candidate(estimator.ransac_mode(), m, best_batch_score)) {
            best_batch_score = m;
            best_model_ind = static_cast<int>(i);
        }
    }

    if (best_batch_score.score < stats.model_score) {
        stats.model_score = best_batch_score.score;
        *best_model = models[static_cast<size_t>(best_model_ind)];
        stats.num_inliers = best_batch_score.edge_inliers;
        detail::update_dynamic_max_iter(opt, state, stats, estimator.num_data, estimator.sample_sz);
    }

    // HCM_MC: admit the raw batch-best into the bounded candidate pool. Final
    // selection happens exclusively in finalize_candidate_pool (refine the N
    // survivors, rank by maximum-cardinality matching, break ties by the
    // stage-1 seed score).
    if (state.candidate_pool.enabled() && state.candidate_pool.would_accept(best_batch_score.score)) {
        state.candidate_pool.try_insert(
            detail::candidate_from_score(models[static_cast<size_t>(best_model_ind)], best_batch_score));
    }

    // Classic LO-RANSAC local optimization on the monotone gate (all modes).
    // In HCM_MC mode, the refined model is used only to update the adaptive
    // stopping bound.  It never enters the candidate pool: stage 2 therefore
    // receives exactly the top-N_c raw seeds and every retained HCM tie-break
    // score is independent of local refinement.
    if (!detail::triggers_local_optimization(estimator.ransac_mode(), best_batch_score, state)) {
        return;
    }
    detail::update_minimal_state(estimator.ransac_mode(), best_batch_score, state);
    Model refined_model = models[static_cast<size_t>(best_model_ind)];
    const uint64_t refine_start = detail::now_ns();
    estimator.refine_model(&refined_model);
    state.timing.refine_ns += detail::now_ns() - refine_start;
    state.timing.refine_calls++;
    stats.refinements++;
    const uint64_t rescore_start = detail::now_ns();
    const ModelScore refined = estimator.score_model(refined_model);
    state.timing.score_ns += detail::now_ns() - rescore_start;
    state.timing.score_calls++;
    if (refined.score < stats.model_score) {
        stats.model_score = refined.score;
        stats.num_inliers = refined.edge_inliers;
        *best_model = refined_model;
        detail::update_dynamic_max_iter(opt, state, stats, estimator.num_data, estimator.sample_sz);
    }
}

template <typename Solver, typename Model = poselib::CameraPose>
poselib::RansacStats ransac(Solver &estimator, const poselib::RansacOptions &opt, Model *best_model,
                            M2MTimingStats *timing = nullptr,
                            HCMCandidatePoolSnapshot *pool_snapshot = nullptr) {
    poselib::RansacStats stats;

    if (estimator.num_o2o_association_ub < estimator.sample_sz) {
        if (pool_snapshot != nullptr) {
            pool_snapshot->captured = true;
            pool_snapshot->capacity = estimator.top_n_candidates();
            pool_snapshot->size = 0;
            pool_snapshot->full = false;
            pool_snapshot->admission_cutoff_score =
                std::numeric_limits<double>::max();
        }
        return stats;
    }
    const uint64_t total_start = detail::now_ns();

    stats.num_inliers = 0;
    stats.model_score = std::numeric_limits<double>::max();
    RansacState state;
    state.dynamic_max_iter = opt.max_iterations;
    state.log_prob_missing_model = std::log(1.0 - opt.success_prob);
    if (estimator.top_n_candidates() > 0) {
        state.candidate_pool.reset(estimator.top_n_candidates(), estimator.pool_dedup_deg());
    }

    if (opt.score_initial_model) {
        score_models(estimator, {*best_model}, opt, state, stats, best_model);
    }

    std::vector<Model> models;
    for (stats.iterations = 0; stats.iterations < opt.max_iterations; stats.iterations++) {
        if (stats.iterations > opt.min_iterations && stats.iterations > state.dynamic_max_iter) {
            break;
        }
        models.clear();
        const uint64_t solve_start = detail::now_ns();
        estimator.generate_models(&models);
        state.timing.solve_ns += detail::now_ns() - solve_start;
        score_models(estimator, models, opt, state, stats, best_model);
    }
    if (pool_snapshot != nullptr) {
        pool_snapshot->captured = true;
        pool_snapshot->capacity = state.candidate_pool.capacity();
        pool_snapshot->size = state.candidate_pool.size();
        pool_snapshot->full = state.candidate_pool.enabled() &&
                              state.candidate_pool.size() == state.candidate_pool.capacity();
        pool_snapshot->admission_cutoff_score =
            state.candidate_pool.admission_cutoff_score();
    }
    // Under HCM_MC mode, select exclusively from the bounded raw-seed pool.
    if (state.candidate_pool.enabled() && !state.candidate_pool.empty()) {
        estimator.finalize_candidate_pool(&state.candidate_pool, best_model, &stats, &state.timing);
    }
    state.timing.pool_insert_attempts = state.candidate_pool.insert_attempts();
    state.timing.pool_dup_hits = state.candidate_pool.dup_hits();
    // The refinement below is unnecessary
    // else {
    //     Model refined_model = *best_model;
    //     estimator.refine_model(&refined_model);
    //     stats.refinements++;
    //     const ModelScore refined = estimator.score_model(refined_model);
    //     if (refined.score < stats.model_score) {
    //         *best_model = refined_model;
    //         stats.num_inliers = refined.edge_inliers;
    //         stats.model_score = refined.score;
    //     }
    // }
    stats.inlier_ratio = stats.num_inliers / static_cast<double>(estimator.num_data);
    state.timing.total_ns = detail::now_ns() - total_start;
    if (timing != nullptr) {
        *timing = state.timing;
    }
    return stats;
}

} // namespace loransac_app
