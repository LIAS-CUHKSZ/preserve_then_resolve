#include "m2m_loransac/relative_pose_m2m.h"

#include "PoseLib/misc/essential.h"
#include "PoseLib/robust/bundle.h"
#include "m2m_loransac/ransac_m2m_impl.h"
#include "PoseLib/robust/sampling.h"
#include "PoseLib/solvers/relpose_5pt.h"
#include "m2m_loransac/hopcroft_karp.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace loransac_app {
namespace {

constexpr size_t kSampleSize = 5;
constexpr double kProbabilityClamp = 1e-9;

struct M2MFeatureData {
    std::vector<size_t> idx1;
    std::vector<size_t> idx2;
    std::vector<double> prob;
    std::vector<double> q_x;
    std::vector<double> q_y;
    std::vector<double> C_x;
    std::vector<double> C_y;
    std::vector<std::vector<size_t>> edges_by_x;
    size_t num_features1 = 0;
    size_t num_features2 = 0;
    size_t num_o2o_association_ub = 0;
};

struct M2MProposalData {
    // Each entry stores an index into the full association arrays. The scorer,
    // all refinement stages, and final inlier extraction intentionally retain
    // M2MFeatureData and the full point arrays.
    std::vector<std::vector<size_t>> edges_by_x;
    size_t num_o2o_association_ub = 0;
};

size_t obtain_max_cardinality(const std::vector<size_t> &idx1, const std::vector<size_t> &idx2) {
    std::unordered_map<size_t, int> compact_x;
    std::unordered_map<size_t, int> compact_y;
    compact_x.reserve(idx1.size());
    compact_y.reserve(idx2.size());

    for (size_t k = 0; k < idx1.size(); ++k) {
        compact_x.emplace(idx1[k], static_cast<int>(compact_x.size()));
        compact_y.emplace(idx2[k], static_cast<int>(compact_y.size()));
    }

    if (compact_x.size() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        compact_y.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("M2M relative pose feature count exceeds Hopcroft-Karp index range");
    }

    dino_m2m::HopcroftKarp graph(compact_x.size(), compact_y.size());
    for (size_t k = 0; k < idx1.size(); ++k) {
        graph.add_edge(static_cast<size_t>(compact_x.at(idx1[k])),
                       static_cast<size_t>(compact_y.at(idx2[k])));
    }
    return graph.maximum_matching();
}

void shuffle_indices(std::vector<size_t> *indices, poselib::RNG_t &state) {
    for (size_t i = indices->size(); i > 1; --i) {
        const size_t j = static_cast<size_t>(static_cast<unsigned int>(poselib::random_int(state))) % i;
        std::swap((*indices)[i - 1], (*indices)[j]);
    }
}

class M2MSampler {
  public:
    M2MSampler(const M2MFeatureData &features, const std::vector<std::vector<size_t>> &proposal_edges_by_x,
               poselib::RNG_t seed)
        : idx2_(features.idx2), edges_by_x_(proposal_edges_by_x), num_features2_(features.num_features2),
          state_(seed) {
        active_features_x_.reserve(edges_by_x_.size());
        for (size_t fx = 0; fx < edges_by_x_.size(); ++fx) {
            if (!edges_by_x_[fx].empty()) {
                active_features_x_.push_back(fx);
            }
        }
    }

    bool generate_sample(std::vector<size_t> *sample) {
        sample->clear();
        if (active_features_x_.size() < kSampleSize || num_features2_ < kSampleSize) {
            return false;
        }

        std::vector<size_t> feature_order;
        feature_order.reserve(active_features_x_.size());
        std::vector<char> used_y(num_features2_, 0);
        const size_t max_attempts = std::max<size_t>(50, 10 * active_features_x_.size());

        for (size_t attempt = 0; attempt < max_attempts; ++attempt) {
            sample->clear();
            std::fill(used_y.begin(), used_y.end(), 0);
            feature_order = active_features_x_;
            shuffle_indices(&feature_order, state_);

            for (size_t fx : feature_order) {
                if (sample->size() == kSampleSize) {
                    break;
                }
                size_t chosen_k = 0;
                if (!pick_edge_for_feature(fx, used_y, &chosen_k)) {
                    continue;
                }
                used_y[idx2_[chosen_k]] = 1;
                sample->push_back(chosen_k);
            }

            if (sample->size() == kSampleSize) {
                return true;
            }
        }

        return false;
    }

  private:
    bool pick_edge_for_feature(size_t fx, const std::vector<char> &used_y, size_t *chosen_k) const {
        const std::vector<size_t> &edges = edges_by_x_[fx];
        if (edges.empty()) {
            return false;
        }

        const size_t start =
            static_cast<size_t>(static_cast<unsigned int>(poselib::random_int(state_))) % edges.size();
        for (size_t t = 0; t < edges.size(); ++t) {
            const size_t k = edges[(start + t) % edges.size()];
            if (!used_y[idx2_[k]]) {
                *chosen_k = k;
                return true;
            }
        }
        return false;
    }

    const std::vector<size_t> &idx2_;
    const std::vector<std::vector<size_t>> &edges_by_x_;
    size_t num_features2_;
    std::vector<size_t> active_features_x_;
    mutable poselib::RNG_t state_;
};

M2MFeatureData build_feature_data(const std::vector<int> &idx1_vec, const std::vector<int> &idx2_vec,
                                  const std::vector<double> &prob_vec, double delta, size_t expected_count) {
    if (idx1_vec.size() != expected_count || idx2_vec.size() != expected_count || prob_vec.size() != expected_count) {
        throw std::invalid_argument("M2M relative pose inputs must have the same association count");
    }
    if (!std::isfinite(delta) || delta <= 0.0) {
        throw std::invalid_argument("M2M relative pose delta must be positive");
    }

    M2MFeatureData data;
    data.idx1.resize(expected_count);
    data.idx2.resize(expected_count);
    data.prob = prob_vec;

    for (size_t k = 0; k < expected_count; ++k) {
        if (idx1_vec[k] < 0 || idx2_vec[k] < 0) {
            throw std::invalid_argument("M2M relative pose feature indices must be non-negative");
        }
        if (!std::isfinite(prob_vec[k]) || prob_vec[k] < 0.0) {
            throw std::invalid_argument("M2M relative pose probabilities must be finite and non-negative");
        }
        data.idx1[k] = static_cast<size_t>(idx1_vec[k]);
        data.idx2[k] = static_cast<size_t>(idx2_vec[k]);
        data.num_features1 = std::max(data.num_features1, data.idx1[k] + 1);
        data.num_features2 = std::max(data.num_features2, data.idx2[k] + 1);
    }
    // compute the maximum one-to-one association by finding the maximum matching in the bipartite graph
    data.num_o2o_association_ub = obtain_max_cardinality(data.idx1, data.idx2);
    data.edges_by_x.assign(data.num_features1, {});
    for (size_t k = 0; k < expected_count; ++k) {
        data.edges_by_x[data.idx1[k]].push_back(k);
    }
    data.q_x.assign(data.num_features1, 0.0);
    data.q_y.assign(data.num_features2, 0.0);
    for (size_t k = 0; k < expected_count; ++k) {
        data.q_x[data.idx1[k]] += data.prob[k];
        data.q_y[data.idx2[k]] += data.prob[k];
    }

    data.C_x.resize(data.num_features1);
    data.C_y.resize(data.num_features2);
    for (size_t fx = 0; fx < data.num_features1; ++fx) {
        data.C_x[fx] = 1.0 / (std::max(1.0 - data.q_x[fx], kProbabilityClamp) * delta);
    }
    for (size_t fy = 0; fy < data.num_features2; ++fy) {
        data.C_y[fy] = 1.0 / (std::max(1.0 - data.q_y[fy], kProbabilityClamp) * delta);
    }

    return data;
}

M2MProposalData build_proposal_data(const M2MFeatureData &features,
                                    const std::vector<size_t> &proposal_indices) {
    M2MProposalData proposal;
    proposal.edges_by_x.assign(features.num_features1, {});
    std::vector<char> seen(features.idx1.size(), 0);
    std::vector<size_t> proposal_idx1;
    std::vector<size_t> proposal_idx2;
    proposal_idx1.reserve(proposal_indices.size());
    proposal_idx2.reserve(proposal_indices.size());
    for (const size_t edge_idx : proposal_indices) {
        if (edge_idx >= features.idx1.size()) {
            throw std::invalid_argument("M2M proposal index is outside the full association graph");
        }
        if (seen[edge_idx]) {
            throw std::invalid_argument("M2M proposal indices must be unique");
        }
        seen[edge_idx] = 1;
        const size_t left_idx = features.idx1[edge_idx];
        const size_t right_idx = features.idx2[edge_idx];
        proposal.edges_by_x[left_idx].push_back(edge_idx);
        proposal_idx1.push_back(left_idx);
        proposal_idx2.push_back(right_idx);
    }
    proposal.num_o2o_association_ub = obtain_max_cardinality(proposal_idx1, proposal_idx2);
    return proposal;
}

double sampson_residual_sq(const Eigen::Matrix3d &E, const poselib::Point2D &x1, const poselib::Point2D &x2) {
    const double x1_0 = x1(0), x1_1 = x1(1);
    const double x2_0 = x2(0), x2_1 = x2(1);

    const double Ex1_0 = E(0, 0) * x1_0 + E(0, 1) * x1_1 + E(0, 2);
    const double Ex1_1 = E(1, 0) * x1_0 + E(1, 1) * x1_1 + E(1, 2);
    const double Ex1_2 = E(2, 0) * x1_0 + E(2, 1) * x1_1 + E(2, 2);

    const double Ex2_0 = E(0, 0) * x2_0 + E(1, 0) * x2_1 + E(2, 0);
    const double Ex2_1 = E(0, 1) * x2_0 + E(1, 1) * x2_1 + E(2, 1);

    const double C = x2_0 * Ex1_0 + x2_1 * Ex1_1 + Ex1_2;
    const double denom = Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1 + Ex2_0 * Ex2_0 + Ex2_1 * Ex2_1;
    if (denom <= std::numeric_limits<double>::epsilon()) {
        return std::numeric_limits<double>::infinity();
    }
    return C * C / denom;
}

bool is_sampson_inlier(const poselib::CameraPose &pose, const Eigen::Matrix3d &E,
                       const std::vector<poselib::Point2D> &x1, const std::vector<poselib::Point2D> &x2, size_t k,
                       double sq_threshold, double *r2) {
    *r2 = sampson_residual_sq(E, x1[k], x2[k]);
    if (!std::isfinite(*r2) || *r2 >= sq_threshold) {
        return false;
    }
    return poselib::check_cheirality(pose, x1[k].homogeneous().normalized(), x2[k].homogeneous().normalized(), 0.01);
}

ModelScore compute_mcm_score(const poselib::CameraPose &pose, const std::vector<poselib::Point2D> &x1,
                             const std::vector<poselib::Point2D> &x2, const M2MFeatureData &features,
                             double sq_threshold) {
    ModelScore result;
    Eigen::Matrix3d E;
    poselib::essential_from_motion(pose, &E);

    std::vector<size_t> inlier_idx1;
    std::vector<size_t> inlier_idx2;
    inlier_idx1.reserve(x1.size());
    inlier_idx2.reserve(x1.size());
    for (size_t k = 0; k < x1.size(); ++k) {
        double r2 = 0.0;
        if (!is_sampson_inlier(pose, E, x1, x2, k, sq_threshold, &r2)) {
            continue;
        }
        inlier_idx1.push_back(features.idx1[k]);
        inlier_idx2.push_back(features.idx2[k]);
        result.edge_inliers++;
    }

    result.cardinality = obtain_max_cardinality(inlier_idx1, inlier_idx2);
    result.score = -static_cast<double>(result.cardinality);
    return result;
}

ModelScore compute_hcm_score(const poselib::CameraPose &pose, const std::vector<poselib::Point2D> &x1,
                             const std::vector<poselib::Point2D> &x2, const M2MFeatureData &features,
                             double sq_threshold, double *left_score = nullptr, double *right_score = nullptr) {
    ModelScore result;
    Eigen::Matrix3d E;
    poselib::essential_from_motion(pose, &E);

    std::vector<size_t> inlier_idx1;
    std::vector<size_t> inlier_idx2;
    inlier_idx1.reserve(x1.size());
    inlier_idx2.reserve(x1.size());
    std::vector<double> weight_x(features.num_features1, 0.0);
    std::vector<double> weight_y(features.num_features2, 0.0);
    for (size_t k = 0; k < x1.size(); ++k) {
        double r2 = 0.0;
        if (!is_sampson_inlier(pose, E, x1, x2, k, sq_threshold, &r2)) {
            continue;
        }

        const size_t fx = features.idx1[k];
        const size_t fy = features.idx2[k];
        // const double residual_weight = 1.0 - std::sqrt(r2 / sq_threshold);
        // const double residual_weight = 1.0 - r2 / sq_threshold;  // an imperfect alignment with msac score in PoseLib implementation
        const double residual_weight = 1.0;
        weight_x[fx] += features.C_x[fx] * features.prob[k] * residual_weight;
        weight_y[fy] += features.C_y[fy] * features.prob[k] * residual_weight;
        inlier_idx1.push_back(fx);
        inlier_idx2.push_back(fy);
        result.edge_inliers++;
    }
    double left_log_score = 0.0;
    for (double weight : weight_x) {
        if (weight > 0) {
            left_log_score += std::log1p(weight);
        }
    }
    double right_log_score = 0.0;
    for (double weight : weight_y) {
        if (weight > 0) {
            right_log_score += std::log1p(weight);
        }
    }
    result.score = -(left_log_score + right_log_score);
    if (left_score != nullptr) {
        *left_score = -left_log_score;
    }
    if (right_score != nullptr) {
        *right_score = -right_log_score;
    }
    result.inlier_idx1 = std::move(inlier_idx1);
    result.inlier_idx2 = std::move(inlier_idx2);
    return result;
}

double compute_msac_cost(const poselib::CameraPose &pose, const std::vector<poselib::Point2D> &x1,
                         const std::vector<poselib::Point2D> &x2, double sq_threshold) {
    Eigen::Matrix3d E;
    poselib::essential_from_motion(pose, &E);
    double cost = 0.0;
    for (size_t k = 0; k < x1.size(); ++k) {
        double r2 = 0.0;
        if (!is_sampson_inlier(pose, E, x1, x2, k, sq_threshold, &r2)) {
            cost += 1.0;
        } else {
            cost += r2 / sq_threshold;
        }
    }
    return x1.empty() ? 1.0 : cost / static_cast<double>(x1.size());
}

void get_m2m_inliers(const poselib::CameraPose &pose, const std::vector<poselib::Point2D> &x1,
                     const std::vector<poselib::Point2D> &x2, double sq_threshold, std::vector<char> *inliers) {
    Eigen::Matrix3d E;
    poselib::essential_from_motion(pose, &E);
    inliers->assign(x1.size(), 0);
    for (size_t k = 0; k < x1.size(); ++k) {
        double r2 = 0.0;
        (*inliers)[k] = is_sampson_inlier(pose, E, x1, x2, k, sq_threshold, &r2);
    }
}

void collect_weighted_m2m_inliers(const std::vector<poselib::Point2D> &x1, const std::vector<poselib::Point2D> &x2,
                                  const M2MFeatureData &features, const std::vector<char> &inlier_mask,
                                  std::vector<poselib::Point2D> *x1_inlier,
                                  std::vector<poselib::Point2D> *x2_inlier, std::vector<double> *weights) {
    x1_inlier->clear();
    x2_inlier->clear();
    weights->clear();

    std::vector<int> inlier_count_x(features.num_features1, 0);
    std::vector<int> inlier_count_y(features.num_features2, 0);
    for (size_t k = 0; k < x1.size(); ++k) {
        if (!inlier_mask[k]) {
            continue;
        }
        inlier_count_x[features.idx1[k]] += 1;
        inlier_count_y[features.idx2[k]] += 1;
    }

    x1_inlier->reserve(x1.size());
    x2_inlier->reserve(x1.size());
    weights->reserve(x1.size());
    for (size_t k = 0; k < x1.size(); ++k) {
        if (!inlier_mask[k]) {
            continue;
        }
        const size_t idx1_k = features.idx1[k];
        const size_t idx2_k = features.idx2[k];
        x1_inlier->push_back(x1[k]);
        x2_inlier->push_back(x2[k]);
        weights->push_back(1.0 / inlier_count_x[idx1_k] + 1.0 / inlier_count_y[idx2_k]);
    }
}

bool polish_m2m_pose(poselib::CameraPose *pose, const std::vector<poselib::Point2D> &x1,
                     const std::vector<poselib::Point2D> &x2, const M2MFeatureData &features,
                     const poselib::RelativePoseOptions &opt) {
    std::vector<char> inlier_mask;
    get_m2m_inliers(*pose, x1, x2, opt.max_error * opt.max_error, &inlier_mask);
    std::vector<poselib::Point2D> x1_inlier;
    std::vector<poselib::Point2D> x2_inlier;
    std::vector<double> weights;
    collect_weighted_m2m_inliers(x1, x2, features, inlier_mask, &x1_inlier, &x2_inlier, &weights);
    if (x1_inlier.size() <= kSampleSize) {
        return false;
    }
    poselib::refine_relpose(x1_inlier, x2_inlier, pose, opt.bundle, weights);
    return true;
}

class RelativePoseM2MEstimator {
  public:
    RelativePoseM2MEstimator(const poselib::RelativePoseOptions &opt, const std::vector<poselib::Point2D> &x1,
                             const std::vector<poselib::Point2D> &x2, const M2MFeatureData &features,
                             const std::vector<std::vector<size_t>> &proposal_edges_by_x,
                             size_t proposal_o2o_association_ub,
                             RansacMode mode, size_t top_n_candidates = 0, double pool_dedup_deg = 0.0,
                             std::vector<M2MCandidateTrace> *candidate_traces = nullptr)
        : num_data(x1.size()), opt_(opt), x1_(x1), x2_(x2), features_(features), mode_(mode),
          top_n_candidates_(mode == RansacMode::HCM_MC ? top_n_candidates : 0), pool_dedup_deg_(pool_dedup_deg),
          candidate_traces_(candidate_traces),
          num_o2o_association_ub(proposal_o2o_association_ub),
          sampler_(features, proposal_edges_by_x, opt.ransac.seed),
          x1s_(kSampleSize), x2s_(kSampleSize) {}

    void generate_models(std::vector<poselib::CameraPose> *models) {
        models->clear();
        sample_.clear();
        if (num_o2o_association_ub < kSampleSize ||
            !sampler_.generate_sample(&sample_)) {
            return;
        }
        for (size_t k = 0; k < kSampleSize; ++k) {
            x1s_[k] = x1_[sample_[k]].homogeneous().normalized();
            x2s_[k] = x2_[sample_[k]].homogeneous().normalized();
        }
        poselib::relpose_5pt(x1s_, x2s_, models);
    }

    RansacMode ransac_mode() const { return mode_; }
    size_t top_n_candidates() const { return top_n_candidates_; }
    double pool_dedup_deg() const { return pool_dedup_deg_; }

    ModelScore score_model(const poselib::CameraPose &pose) const {
        const double sq_threshold = opt_.max_error * opt_.max_error;
        if (mode_ == RansacMode::MCM) {
            return compute_mcm_score(pose, x1_, x2_, features_, sq_threshold);
        }
        return compute_hcm_score(pose, x1_, x2_, features_, sq_threshold);
    }

    void finalize_candidate_pool(CandidatePool *pool, poselib::CameraPose *best_model,
                                 poselib::RansacStats *stats, M2MTimingStats *timing) const {
        if (pool == nullptr || pool->empty()) {
            return;
        }

        const auto is_better = [](const Candidate &cur, const Candidate &best) {
            if (cur.cardinality != best.cardinality) {
                return cur.cardinality > best.cardinality;
            }
            return cur.hcm_score < best.hcm_score;
        };

        // The pool holds raw minimal poses (explored under HCM). Refine each of
        // the N survivors once here, then score it by maximum-cardinality
        // matching. The cardinality tie-break below deliberately keeps the
        // stage-1 HCM score of the UNREFINED seed: being computed before local
        // optimization, it is independent of the refinement objective and thus
        // discounts hypotheses whose consensus is manufactured by the optimizer
        // itself. (A/B on MegaDepth/METU-CS/METU-CC showed re-evaluating HCM at
        // the refined pose degrades AUC.) Refinement and MCM ranking are timed
        // into separate buckets so stage-2 cost decomposes as
        // N_c x (refine + MCM eval).
        if (candidate_traces_ != nullptr) {
            candidate_traces_->clear();
            candidate_traces_->reserve(pool->entries().size());
        }
        for (Candidate &candidate : pool->entries()) {
            M2MCandidateTrace trace;
            if (candidate_traces_ != nullptr) {
                trace.seed_pose = candidate.pose;
                trace.stored_hcm_score = candidate.hcm_score;
                compute_hcm_score(candidate.pose, x1_, x2_, features_, opt_.max_error * opt_.max_error,
                                  &trace.seed_hcm_left, &trace.seed_hcm_right);
            }
            const uint64_t refine_start = detail::now_ns();
            refine_model(&candidate.pose);
            const uint64_t rank_start = detail::now_ns();
            const ModelScore mcm =
                compute_mcm_score(candidate.pose, x1_, x2_, features_, opt_.max_error * opt_.max_error);
            const uint64_t rank_end = detail::now_ns();
            if (timing != nullptr) {
                timing->refine_ns += rank_start - refine_start;
                timing->refine_calls++;
                timing->rank_score_ns += rank_end - rank_start;
                timing->rank_score_calls++;
            }
            candidate.cardinality = mcm.cardinality;
            candidate.edge_inliers = mcm.edge_inliers;
            stats->refinements++;
            if (candidate_traces_ != nullptr) {
                trace.stage2_refine_ns = rank_start - refine_start;
                trace.mcm_score_ns = rank_end - rank_start;
                trace.refined_pose = candidate.pose;
                const ModelScore refined_hcm =
                    compute_hcm_score(candidate.pose, x1_, x2_, features_, opt_.max_error * opt_.max_error,
                                      &trace.refined_hcm_left, &trace.refined_hcm_right);
                trace.refined_hcm_score = refined_hcm.score;
                trace.refined_msac_cost =
                    compute_msac_cost(candidate.pose, x1_, x2_, opt_.max_error * opt_.max_error);
                trace.edge_inliers = mcm.edge_inliers;
                trace.cardinality = mcm.cardinality;
                trace.polished_pose = candidate.pose;
                polish_m2m_pose(&trace.polished_pose, x1_, x2_, features_, opt_);
                candidate_traces_->push_back(std::move(trace));
            }
        }

        if (candidate_traces_ != nullptr) {
            std::stable_sort(candidate_traces_->begin(), candidate_traces_->end(),
                             [](const M2MCandidateTrace &a, const M2MCandidateTrace &b) {
                                 return a.stored_hcm_score < b.stored_hcm_score;
                             });
            for (size_t k = 0; k < candidate_traces_->size(); ++k) {
                (*candidate_traces_)[k].stored_hcm_rank = k + 1;
            }
        }

        // Diversity diagnostic: distinct basins among the refined candidates.
        if (timing != nullptr) {
            timing->pool_unique_basins = pool->count_unique_basins();
        }

        const Candidate *best_candidate = &pool->entries().front();
        for (const Candidate &candidate : pool->entries()) {
            if (is_better(candidate, *best_candidate)) {
                best_candidate = &candidate;
            }
        }

        *best_model = best_candidate->pose;
        stats->num_inliers = best_candidate->edge_inliers;
        stats->model_score = -static_cast<double>(best_candidate->cardinality);
    }

    void refine_model(poselib::CameraPose *pose) const {
        poselib::BundleOptions bundle_opt;
        bundle_opt.loss_type = poselib::BundleOptions::LossType::TRUNCATED;
        bundle_opt.loss_scale = opt_.max_error;
        bundle_opt.max_iterations = 25;

        const double sq_threshold = 5.0 * opt_.max_error * opt_.max_error;
        std::vector<char> inlier_mask;
        get_m2m_inliers(*pose, x1_, x2_, sq_threshold, &inlier_mask);

        std::vector<poselib::Point2D> x1_inlier;
        std::vector<poselib::Point2D> x2_inlier;
        std::vector<double> weights;
        collect_weighted_m2m_inliers(x1_, x2_, features_, inlier_mask, &x1_inlier, &x2_inlier, &weights);
        if (x1_inlier.size() <= kSampleSize) {
            return;
        }
        poselib::refine_relpose(x1_inlier, x2_inlier, pose, bundle_opt, weights);
    }

    const size_t sample_sz = kSampleSize;
    const size_t num_data;
    const size_t num_o2o_association_ub;

  private:
    const poselib::RelativePoseOptions &opt_;
    const std::vector<poselib::Point2D> &x1_;
    const std::vector<poselib::Point2D> &x2_;
    const M2MFeatureData &features_;
    RansacMode mode_;
    size_t top_n_candidates_;
    double pool_dedup_deg_ = 0.0;
    std::vector<M2MCandidateTrace> *candidate_traces_ = nullptr;
    M2MSampler sampler_;
    std::vector<Eigen::Vector3d> x1s_;
    std::vector<Eigen::Vector3d> x2s_;
    std::vector<size_t> sample_;
};

poselib::RansacStats ransac_relpose_m2m(const std::vector<poselib::Point2D> &x1,
                                        const std::vector<poselib::Point2D> &x2,
                                        const poselib::RelativePoseOptions &opt, const M2MFeatureData &features,
                                        const std::vector<std::vector<size_t>> &proposal_edges_by_x,
                                        size_t proposal_o2o_association_ub,
                                        RansacMode mode, size_t top_n_candidates, poselib::CameraPose *best_model,
                                        std::vector<char> *best_inliers, M2MTimingStats *timing,
                                        double pool_dedup_deg = 0.0,
                                        std::vector<M2MCandidateTrace> *candidate_traces = nullptr,
                                        HCMCandidatePoolSnapshot *pool_snapshot = nullptr) {
    if (!opt.ransac.score_initial_model) {
        best_model->q << 1.0, 0.0, 0.0, 0.0;
        best_model->t.setZero();
    }
    RelativePoseM2MEstimator estimator(opt, x1, x2, features, proposal_edges_by_x,
                                      proposal_o2o_association_ub, mode, top_n_candidates,
                                      pool_dedup_deg, candidate_traces);
    poselib::RansacStats stats =
        loransac_app::ransac<RelativePoseM2MEstimator>(estimator, opt.ransac, best_model, timing,
                                                       pool_snapshot);
    get_m2m_inliers(*best_model, x1, x2, opt.max_error * opt.max_error, best_inliers);
    return stats;
}

} // namespace

poselib::RansacStats estimate_relative_pose_m2m(const std::vector<poselib::Point2D> &points2D_1,
                                                const std::vector<poselib::Point2D> &points2D_2,
                                                const poselib::Camera &camera1,
                                                const poselib::Camera &camera2,
                                                const std::vector<int> &idx1_vec,
                                                const std::vector<int> &idx2_vec,
                                                const std::vector<double> &prob_vec, double delta,
                                                RansacMode mode, size_t top_n_candidates,
                                                const poselib::RelativePoseOptions &opt,
                                                poselib::CameraPose *relative_pose, std::vector<char> *inliers,
                                                M2MTimingStats *timing, double pool_dedup_deg,
                                                std::vector<M2MCandidateTrace> *candidate_traces,
                                                const std::vector<size_t> *proposal_indices,
                                                const poselib::CameraPose *ground_truth_probe_pose,
                                                M2MGroundTruthHypothesisProbe *ground_truth_probe) {
    if (relative_pose == nullptr || inliers == nullptr) {
        throw std::invalid_argument("M2M relative pose output pointers must not be null");
    }
    if (points2D_1.size() != points2D_2.size()) {
        throw std::invalid_argument("M2M relative pose point vectors must have the same length");
    }
    if (mode != RansacMode::HCM && mode != RansacMode::MCM && mode != RansacMode::HCM_MC) {
        throw std::invalid_argument("M2M relative pose requires ransac_mode=HCM, MCM, or HCM_MC");
    }
    if (mode == RansacMode::HCM_MC && top_n_candidates == 0) {
        throw std::invalid_argument("HCM_MC requires top_n_candidates >= 1");
    }
    if (candidate_traces != nullptr && mode != RansacMode::HCM_MC) {
        throw std::invalid_argument("candidate traces require HCM_MC mode");
    }
    if ((ground_truth_probe_pose == nullptr) != (ground_truth_probe == nullptr)) {
        throw std::invalid_argument(
            "ground-truth hypothesis probe requires both pose and result pointers");
    }
    if (ground_truth_probe != nullptr && mode != RansacMode::HCM_MC) {
        throw std::invalid_argument("ground-truth hypothesis probe requires HCM_MC mode");
    }
    if (opt.tangent_sampson) {
        throw std::invalid_argument("M2M relative pose currently supports only non-tangent Sampson scoring");
    }
    if (proposal_indices != nullptr && opt.ransac.min_iterations != opt.ransac.max_iterations) {
        throw std::invalid_argument(
            "M2M proposal sampling requires a fixed iteration budget");
    }

    const size_t num_pts = points2D_1.size();
    M2MFeatureData features = build_feature_data(idx1_vec, idx2_vec, prob_vec, delta, num_pts);
    M2MProposalData proposal;
    const std::vector<std::vector<size_t>> *proposal_edges_by_x = &features.edges_by_x;
    size_t proposal_o2o_association_ub = features.num_o2o_association_ub;
    if (proposal_indices != nullptr) {
        proposal = build_proposal_data(features, *proposal_indices);
        proposal_edges_by_x = &proposal.edges_by_x;
        proposal_o2o_association_ub = proposal.num_o2o_association_ub;
    }

    double scale = 0.5 * (1.0 / camera1.focal() + 1.0 / camera2.focal());
    poselib::RelativePoseOptions opt_scaled = opt;
    opt_scaled.max_error *= scale;
    opt_scaled.bundle.loss_scale *= scale;

    std::vector<poselib::Point2D> x1_calib(num_pts);
    std::vector<poselib::Point2D> x2_calib(num_pts);
    for (size_t k = 0; k < num_pts; ++k) {
        camera1.unproject(points2D_1[k], &x1_calib[k]);
        camera2.unproject(points2D_2[k], &x2_calib[k]);
    }

    HCMCandidatePoolSnapshot pool_snapshot;
    poselib::RansacStats stats = ransac_relpose_m2m(
        x1_calib, x2_calib, opt_scaled, features, *proposal_edges_by_x,
        proposal_o2o_association_ub, mode, top_n_candidates, relative_pose, inliers,
        timing, pool_dedup_deg, candidate_traces,
        ground_truth_probe != nullptr ? &pool_snapshot : nullptr);
    // refine the best model returned by RANSAC (common polish step for all modes;
    // timed into the refine bucket, outside ransac() total_ns)
    if (stats.num_inliers > kSampleSize) {
        const uint64_t polish_start = detail::now_ns();
        const bool polished = polish_m2m_pose(relative_pose, x1_calib, x2_calib, features, opt_scaled);
        if (timing != nullptr && polished) {
            timing->refine_ns += detail::now_ns() - polish_start;
            timing->refine_calls++;
        }
    }

    // This score is intentionally computed after every behavior-affecting
    // operation above.  It is not inserted into the candidate pool and is not
    // included in stopping, LO, Stage-2 selection, final polish, or timing.
    if (ground_truth_probe != nullptr) {
        if (!pool_snapshot.captured) {
            throw std::runtime_error("ground-truth hypothesis probe did not capture Stage-1 pool");
        }
        const ModelScore gt_score = compute_hcm_score(
            *ground_truth_probe_pose, x1_calib, x2_calib, features,
            opt_scaled.max_error * opt_scaled.max_error);
        ground_truth_probe->evaluated = true;
        ground_truth_probe->pool_capacity = pool_snapshot.capacity;
        ground_truth_probe->pool_size = pool_snapshot.size;
        ground_truth_probe->pool_full = pool_snapshot.full;
        ground_truth_probe->gt_hcm_score = gt_score.score;
        ground_truth_probe->pool_cutoff_hcm_score = pool_snapshot.admission_cutoff_score;
        ground_truth_probe->gt_edge_inliers = gt_score.edge_inliers;
        ground_truth_probe->would_enter_pool =
            pool_snapshot.size < pool_snapshot.capacity ||
            gt_score.score < pool_snapshot.admission_cutoff_score;
    }

    return stats;
}

} // namespace loransac_app
