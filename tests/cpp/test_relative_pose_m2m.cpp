#include "m2m_loransac/relative_pose_m2m.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

int fail(const char *message) {
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    using loransac_app::RansacMode;

    const std::vector<poselib::Point2D> points1 = {
        {100.0, 100.0}, {180.0, 140.0}, {260.0, 200.0}, {340.0, 260.0}, {420.0, 320.0}};
    const std::vector<poselib::Point2D> points2 = {
        {110.0, 100.0}, {195.0, 140.0}, {280.0, 200.0}, {365.0, 260.0}, {450.0, 320.0}};
    const poselib::Camera camera("PINHOLE", {800.0, 800.0, 320.0, 240.0}, 640, 480);
    const std::vector<int> right_ids = {0, 1, 2, 3, 4};
    const std::vector<double> probabilities(5, 0.3);

    poselib::RelativePoseOptions options;
    options.max_error = 1.0;
    options.bundle.loss_scale = 1.0;
    options.ransac.min_iterations = 0;
    options.ransac.max_iterations = 0;

    // Five CSV rows are not sufficient when their maximum one-to-one
    // association cardinality is only one.
    const std::vector<int> repeated_left_ids(5, 0);
    poselib::CameraPose pose;
    std::vector<char> inliers;
    const poselib::RansacStats no_model = loransac_app::estimate_relative_pose_m2m(
        points1, points2, camera, camera, repeated_left_ids, right_ids, probabilities, 0.01,
        RansacMode::HCM, 0, options, &pose, &inliers);
    if (no_model.iterations != 0 || no_model.model_score != std::numeric_limits<double>::max()) {
        return fail("low-cardinality input did not retain the no-model RANSAC sentinel");
    }

    // A standardized as-stored initial pose is a valid scored model even when
    // the random-sampling loop is disabled.
    options.ransac.score_initial_model = true;
    const Eigen::Matrix3d identity_rotation = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d x_translation(1.0, 0.0, 0.0);
    pose = poselib::CameraPose(identity_rotation, x_translation);
    loransac_app::M2MTimingStats hcm_timing;
    const poselib::RansacStats initial_model = loransac_app::estimate_relative_pose_m2m(
        points1, points2, camera, camera, right_ids, right_ids, probabilities, 0.01,
        RansacMode::HCM, 0, options, &pose, &inliers, &hcm_timing);
    if (!std::isfinite(initial_model.model_score) ||
        initial_model.model_score == std::numeric_limits<double>::max()) {
        return fail("a scored initial pose was not reported as a valid model");
    }
    if (hcm_timing.score_calls != 2 || hcm_timing.score_ns == 0 || hcm_timing.rank_score_calls != 0) {
        return fail("HCM initial-pose timing did not isolate the two score_model calls");
    }

    // The zero-iteration HCM and MCM runs replay the same initial pose and the
    // same deterministic refinement. This is the paired scorer fixture used by
    // the paired scorer check; only the scoring mechanism differs.
    pose = poselib::CameraPose(identity_rotation, x_translation);
    loransac_app::M2MTimingStats mcm_timing;
    const poselib::RansacStats mcm_initial_model = loransac_app::estimate_relative_pose_m2m(
        points1, points2, camera, camera, right_ids, right_ids, probabilities, 0.01,
        RansacMode::MCM, 0, options, &pose, &inliers, &mcm_timing);
    if (!std::isfinite(mcm_initial_model.model_score) || mcm_timing.score_calls != 2 ||
        mcm_timing.score_ns == 0 || mcm_timing.rank_score_calls != 0) {
        return fail("MCM paired initial-pose timing did not isolate the two score_model calls");
    }

    // HCM_MC stage 2 must contain only the bounded raw-seed pool.  The scored
    // initial pose triggers local optimization in this zero-iteration fixture;
    // if LO-best were appended as a separate candidate, a capacity-one trace
    // would incorrectly contain two entries.
    pose = poselib::CameraPose(identity_rotation, x_translation);
    std::vector<loransac_app::M2MCandidateTrace> candidate_traces;
    loransac_app::M2MTimingStats staged_timing;
    const poselib::RansacStats staged_model = loransac_app::estimate_relative_pose_m2m(
        points1, points2, camera, camera, right_ids, right_ids, probabilities, 0.01,
        RansacMode::HCM_MC, 1, options, &pose, &inliers, &staged_timing, 0.0,
        &candidate_traces);
    if (!std::isfinite(staged_model.model_score) || candidate_traces.size() != 1) {
        return fail("HCM_MC stage 2 did not retain exactly the capacity-one raw-seed pool");
    }
    if (staged_timing.score_calls != 2 || staged_timing.score_ns == 0 ||
        staged_timing.rank_score_calls != 1 || staged_timing.rank_score_ns == 0) {
        return fail("HCM_MC paired timing did not separate stage-1 HCM from stage-2 MCM");
    }

    // The optional reference-pose probe observes the frozen raw-seed pool but
    // must not enter it or alter any behavior-affecting output/counter.  Replay
    // the exact fixture with the probe enabled and compare against the run
    // above; timing durations may vary, while call counts and candidates may
    // not.
    const poselib::CameraPose staged_pose_without_probe = pose;
    const poselib::CameraPose reference_pose(
        identity_rotation, Eigen::Vector3d(0.0, 1.0, 0.0));
    pose = poselib::CameraPose(identity_rotation, x_translation);
    std::vector<loransac_app::M2MCandidateTrace> probed_candidate_traces;
    loransac_app::M2MTimingStats probed_timing;
    loransac_app::M2MGroundTruthHypothesisProbe ground_truth_probe;
    const poselib::RansacStats probed_model = loransac_app::estimate_relative_pose_m2m(
        points1, points2, camera, camera, right_ids, right_ids, probabilities, 0.01,
        RansacMode::HCM_MC, 1, options, &pose, &inliers, &probed_timing, 0.0,
        &probed_candidate_traces, nullptr, &reference_pose, &ground_truth_probe);
    if (!ground_truth_probe.evaluated || ground_truth_probe.pool_capacity != 1 ||
        ground_truth_probe.pool_size != 1 || !ground_truth_probe.pool_full ||
        !std::isfinite(ground_truth_probe.gt_hcm_score) ||
        !std::isfinite(ground_truth_probe.pool_cutoff_hcm_score) ||
        ground_truth_probe.would_enter_pool !=
            (ground_truth_probe.gt_hcm_score <
             ground_truth_probe.pool_cutoff_hcm_score)) {
        return fail("ground-truth hypothesis probe did not report strict pool admission");
    }
    if (probed_model.iterations != staged_model.iterations ||
        probed_model.refinements != staged_model.refinements ||
        probed_model.num_inliers != staged_model.num_inliers ||
        probed_model.model_score != staged_model.model_score ||
        !pose.q.isApprox(staged_pose_without_probe.q, 0.0) ||
        !pose.t.isApprox(staged_pose_without_probe.t, 0.0) ||
        probed_timing.score_calls != staged_timing.score_calls ||
        probed_timing.refine_calls != staged_timing.refine_calls ||
        probed_timing.rank_score_calls != staged_timing.rank_score_calls ||
        probed_timing.pool_insert_attempts != staged_timing.pool_insert_attempts ||
        probed_timing.pool_dup_hits != staged_timing.pool_dup_hits ||
        probed_timing.pool_unique_basins != staged_timing.pool_unique_basins ||
        probed_candidate_traces.size() != candidate_traces.size() ||
        probed_candidate_traces.front().stored_hcm_score !=
            candidate_traces.front().stored_hcm_score ||
        !probed_candidate_traces.front().seed_pose.q.isApprox(
            candidate_traces.front().seed_pose.q, 0.0) ||
        !probed_candidate_traces.front().seed_pose.t.isApprox(
            candidate_traces.front().seed_pose.t, 0.0)) {
        return fail("ground-truth hypothesis probe changed the normal HCM_MC run");
    }

    // A controlled proposal source changes only minimal sampling. The full
    // graph below has ten one-to-one edges, while the first proposal has only
    // four. The proposal-cardinality guard must reject it even though the full
    // graph itself easily satisfies the five-point requirement.
    std::vector<poselib::Point2D> full_points1 = points1;
    std::vector<poselib::Point2D> full_points2 = points2;
    full_points1.insert(full_points1.end(),
                        {{120.0, 360.0}, {200.0, 380.0}, {280.0, 400.0},
                         {360.0, 420.0}, {440.0, 440.0}});
    full_points2.insert(full_points2.end(),
                        {{135.0, 360.0}, {220.0, 380.0}, {305.0, 400.0},
                         {390.0, 420.0}, {475.0, 440.0}});
    const std::vector<int> full_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const std::vector<double> full_probabilities(10, 0.3);
    const std::vector<size_t> low_cardinality_proposal = {0, 1, 2, 3};
    pose = poselib::CameraPose(identity_rotation, x_translation);
    const poselib::RansacStats proposal_rejected = loransac_app::estimate_relative_pose_m2m(
        full_points1, full_points2, camera, camera, full_ids, full_ids,
        full_probabilities, 0.01, RansacMode::HCM, 0, options, &pose, &inliers,
        nullptr, 0.0, nullptr, &low_cardinality_proposal);
    if (proposal_rejected.model_score != std::numeric_limits<double>::max()) {
        return fail("minimal sampling used the full graph instead of the proposal graph");
    }

    // With a valid five-edge E1 proposal, both HCM and MCM initial-pose scores
    // must still see all ten E_K edges. The returned inlier mask is likewise a
    // full-graph object, which also covers the data scope used by LO/refinement.
    const std::vector<size_t> e1_proposal = {0, 1, 2, 3, 4};
    for (const RansacMode mode : {RansacMode::HCM, RansacMode::MCM}) {
        pose = poselib::CameraPose(identity_rotation, x_translation);
        const poselib::RansacStats full_score = loransac_app::estimate_relative_pose_m2m(
            full_points1, full_points2, camera, camera, full_ids, full_ids,
            full_probabilities, 0.01, mode, 0, options, &pose, &inliers,
            nullptr, 0.0, nullptr, &e1_proposal);
        if (!std::isfinite(full_score.model_score) ||
            full_score.num_inliers != full_points1.size() ||
            inliers.size() != full_points1.size() ||
            !std::all_of(inliers.begin(), inliers.end(), [](char value) { return value != 0; })) {
            return fail("proposal sampling leaked into full-graph scoring or inlier extraction");
        }
    }

    // Proposal-mode adaptive stopping would mix proposal trials with a
    // full-graph inlier count. The public API therefore enforces the same fixed
    // budget contract as the runner.
    poselib::RelativePoseOptions adaptive_options = options;
    adaptive_options.ransac.max_iterations = 1;
    bool rejected_adaptive_proposal = false;
    try {
        loransac_app::estimate_relative_pose_m2m(
            full_points1, full_points2, camera, camera, full_ids, full_ids,
            full_probabilities, 0.01, RansacMode::HCM, 0, adaptive_options,
            &pose, &inliers, nullptr, 0.0, nullptr, &e1_proposal);
    } catch (const std::invalid_argument &) {
        rejected_adaptive_proposal = true;
    }
    if (!rejected_adaptive_proposal) {
        return fail("proposal sampling accepted a non-fixed iteration budget");
    }

    std::vector<int> negative_ids = right_ids;
    negative_ids.front() = -1;
    bool rejected_negative_id = false;
    try {
        loransac_app::estimate_relative_pose_m2m(
            points1, points2, camera, camera, negative_ids, right_ids, probabilities, 0.01,
            RansacMode::HCM, 0, options, &pose, &inliers);
    } catch (const std::invalid_argument &) {
        rejected_negative_id = true;
    }
    if (!rejected_negative_id) {
        return fail("negative feature IDs were accepted by the public estimator");
    }

    return 0;
}
