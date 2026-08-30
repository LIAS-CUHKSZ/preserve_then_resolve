# Evaluation entry points

The public release separates correspondence diagnostics from downstream pose
estimation. Neither workflow downloads data or model weights.

| Scope | Entry point | Main output |
| --- | --- | --- |
| Section 2 | `evaluation.dense_correspondence.evaluate` | one resumable NPZ shard per pair/layer |
| Figure 2 | `evaluation.dense_correspondence.plot_extended_rank_layer_curves` | PDF/PNG plus compact CSV/JSON reports |
| Q1 | `evaluation.estimation.run_q1_nested_k` | five-seed nested-K AUC and edge-count CSVs |
| Q1 probe | `evaluation.estimation.ground_truth_hypothesis_probe` | read-only top-100 hypothesis-pool inclusion rates |
| Q2/Q3 | `evaluation.estimation.run_pose_gms_selection` | per-seed/five-seed Pose-AUC and runtime CSVs |
| Figure 5 | `evaluation.visualization.plot_q1_nested_k` | fixed-proposal K sweep |
| Figure 6 | `evaluation.visualization.plot_q2_estimator_comparison` | four-estimator comparison |
| Figure 7 | `evaluation.visualization.plot_q3_dino` | corrected-DINOv3 and raw-DINOv2 panels |

Exact commands and parameters are in
[`experiments/README.md`](../experiments/README.md).

## Correspondence metric

The dense NAVI evaluator ranks the top eight cosine candidates against the
complete target patch grid in both directions. The main metric is the equal
arithmetic mean of strict 1, 2, and 5 cm object-macro correspondence recalls.
The Figure 2 horizontal coordinate is the directly observed, uncapped number
of mutual associations at each K; curve points are not interpolated.

The correspondence pair lists are image-disjoint from all three canonical
NAVI-Wild estimation bins. The evaluator checks this before writing shards.

## Pose metric

For each pair, pose error is
`max(rotation_error_deg, translation_error_deg)`. Pose AUC is normalized at 5,
10, and 20 degrees. Expected pair IDs come from `pose_intrinsics.csv`; duplicate
or unexpected IDs are rejected, and missing/non-finite outcomes are scored as
180-degree failures.

All stochastic comparisons use pair-specific seeds 0--4. Q3 deltas are paired
within representation, dataset, threshold, and seed before their mean and
sample standard deviation are plotted.

## Result-tree invariant

Raw progressive associations and GMS-filtered associations are estimator
inputs, not disposable caches. Each directory contains a provenance manifest
and exactly one `matching_<index>.csv` for every registered pair. Resume logic
validates the existing configuration and pair rectangle before reusing any
result.
