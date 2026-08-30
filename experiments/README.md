# Main-paper experiment recipes

The YAML files record the frozen contracts; the commands below form the
corresponding from-input execution chain.

Set local roots first:

```bash
export DATA_ROOT=/path/to/data
export DINO3_WEIGHTS=/path/to/dinov3_vitl16_pretrain_lvd1689m.pth
export DINO2_WEIGHTS=/path/to/dinov2_vitl14_reg4_pretrain.pth
export POSE_ROOT=/path/to/generated/matching_estimation_results
export WORK_ROOT=/path/to/generated/experiment_work
```

The expected checkpoint SHA-256 values are:

```text
DINOv3 ViT-L/16  8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035
DINOv2 ViT-L/14  36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51
```

Obtain the DINO checkpoints and datasets from their official sources before
running these scripts.
LightGlue can download its bundled SuperPoint checkpoint on the first
uncached keypoint extraction, as described in the root README. `POSE_ROOT`
and `WORK_ROOT` must be writable. Dataset layout, including the required
ScanNet/MegaDepth benchmark pair metadata, is documented in `data/README.md`.

Prepare the common six-dataset pose tree once:

```bash
python -m evaluation.estimation.prepare_pose_estimation_data \
  --rss-root . --data-root "$DATA_ROOT" --output-root "$POSE_ROOT"
```

## Section 2 / Figure 2

Extract the tracked NAVI-Wild correspondence split for both backbones and fit
the DINOv3 rank-600 bases:

```bash
for BIN in 0-40 40-80 80-120; do
  dino-m2m extract-dino \
    --input-root "$DATA_ROOT/navi" \
    --pair-file "data/splits/navi/correspondence/pairs_wildset_${BIN}.csv" \
    --output-root "$WORK_ROOT/dense_features/dinov3" \
    --source third_party/dinov3 --weights "$DINO3_WEIGHTS" \
    --model dinov3_vitl16 --layer 16 17 18 19 20 21 22 23 24

  dino-m2m extract-dino \
    --input-root "$DATA_ROOT/navi" \
    --pair-file "data/splits/navi/correspondence/pairs_wildset_${BIN}.csv" \
    --output-root "$WORK_ROOT/dense_features/dinov2" \
    --source third_party/dinov2 --weights "$DINO2_WEIGHTS" \
    --model dinov2_vitl14_reg --layer 16 17 18 19 20 21 22 23 24
done

dino-m2m fit-bias-for-pairs \
  --pairs-root data/splits/navi/correspondence \
  --pair-pattern 'pairs_wildset_*.csv' \
  --output-root "$WORK_ROOT/dense_basis/dinov3" \
  --source third_party/dinov3 --weights "$DINO3_WEIGHTS" \
  --model dinov3_vitl16 --layer 16 17 18 19 20 21 22 23 24 \
  --debias-ranks 600 --existing skip --save-json
```

Evaluate every bin, then render the body plots:

```bash
for BIN in 0-40 40-80 80-120; do
  python -m evaluation.dense_correspondence.evaluate \
    --bin "$BIN" --image-root "$DATA_ROOT/navi" \
    --dino-root "$WORK_ROOT/dense_features/dinov3" \
    --basis-root "$WORK_ROOT/dense_basis/dinov3" \
    --output-root "$WORK_ROOT/section2/dinov3" \
    --weights "$DINO3_WEIGHTS" --model dinov3_vitl16 \
    --layer 16 17 18 19 20 21 22 23 24 \
    --rank 0 100 200 300 400 500 600 --max-k 8

  python -m evaluation.dense_correspondence.evaluate \
    --bin "$BIN" --image-root "$DATA_ROOT/navi" \
    --dino-root "$WORK_ROOT/dense_features/dinov2" \
    --output-root "$WORK_ROOT/section2/dinov2" \
    --weights "$DINO2_WEIGHTS" --model dinov2_vitl14_reg \
    --layer 16 17 18 19 20 21 22 23 24 --rank 0 --max-k 8
done

python -m evaluation.dense_correspondence.plot_extended_rank_layer_curves \
  --evaluation-root "$WORK_ROOT/section2/dinov3" \
  --dinov2-evaluation-root "$WORK_ROOT/section2/dinov2" \
  --report-output "$WORK_ROOT/section2/reports" \
  --figure-output "$WORK_ROOT/section2/figures" \
  --fixed-rank 200 --layer-sweep-layers 16 17 18 19 20 21 22 23 24 \
  --highlight-layers 19 24
```

The horizontal coordinate is the uncapped mutual-association count. The
vertical coordinate is the equal mean of strict 1/2/5 cm correspondence
recalls.

## Q1 / Figure 5

Generate an isolated, uncapped DINOv3 layer-19/rank-200 K5 tree:

```bash
python -m evaluation.estimation.run_pose_matching all \
  --results-root "$POSE_ROOT" --input-manifest "$POSE_ROOT/experiment_inputs.json" \
  --basis-root "$WORK_ROOT/pose_basis" --cache-root "$WORK_ROOT/pose_cache" \
  --keypoint-cache-root "$WORK_ROOT/keypoint_cache" \
  --model-name dinov3_vitl16 --layer 19 --debias-ranks 200 \
  --source third_party/dinov3 --weights "$DINO3_WEIGHTS" \
  --method-prefix combined_interpolate_v3_q1 --max-ks 5 \
  --association-upperbound 0

python -m evaluation.estimation.run_q1_nested_k all \
  --results-root "$POSE_ROOT" --source-method combined_interpolate_v3_q1 \
  --work-root "$WORK_ROOT/q1" \
  --runner-binary build/m2m_loransac/m2m_loransac_runner

python -m evaluation.visualization.plot_q1_nested_k \
  --summary "$WORK_ROOT/q1/summary/q1_auc_summary.csv" \
  --edge-counts "$WORK_ROOT/q1/summary/q1_edge_counts.csv" \
  --output-stem "$WORK_ROOT/q1/figures/q1_nested_k"
```

The optional read-only reference-hypothesis probe uses the resulting run
index and never enters estimator decisions:

```bash
python -m evaluation.estimation.ground_truth_hypothesis_probe run \
  --run-index "$WORK_ROOT/q1/run_index.csv" \
  --runner build/m2m_loransac/m2m_loransac_runner \
  --output-root "$WORK_ROOT/q1/gt_probe"

python -m evaluation.estimation.ground_truth_hypothesis_probe summarize \
  --output-root "$WORK_ROOT/q1/gt_probe" \
  --summary-dir "$WORK_ROOT/q1/gt_probe/summary"
```

## Q2 / Figure 6 and Table 1

Generate the capped DINOv3 K1/K5 tree, apply the fixed GMS setting, and run
the four estimators:

```bash
python -m evaluation.estimation.run_pose_matching all \
  --results-root "$POSE_ROOT" --input-manifest "$POSE_ROOT/experiment_inputs.json" \
  --basis-root "$WORK_ROOT/pose_basis" --cache-root "$WORK_ROOT/pose_cache" \
  --keypoint-cache-root "$WORK_ROOT/keypoint_cache" \
  --model-name dinov3_vitl16 --layer 19 --debias-ranks 200 \
  --source third_party/dinov3 --weights "$DINO3_WEIGHTS" \
  --method-prefix combined_interpolate_v3 --max-ks 1 5 \
  --association-upperbound 2048

python -m evaluation.estimation.run_pose_gms_selection filter \
  --results-root "$POSE_ROOT" --input-manifest "$POSE_ROOT/experiment_inputs.json" \
  --work-root "$WORK_ROOT/q2" --model-name dinov3_vitl16 \
  --layer 19 --debias-rank 200 --method-prefix combined_interpolate_v3 \
  --candidates 4:20 --gms-binary build/gms_filter/gms_filter_csv_m2m

python -m evaluation.estimation.run_pose_gms_selection estimate \
  --results-root "$POSE_ROOT" --input-manifest "$POSE_ROOT/experiment_inputs.json" \
  --work-root "$WORK_ROOT/q2" --model-name dinov3_vitl16 \
  --layer 19 --debias-rank 200 --method-prefix combined_interpolate_v3 \
  --pipelines gms:4:20:CM gms:4:20:HCM gms:4:20:MCM gms:4:20:HCM_MC \
  --seeds 0 1 2 3 4 \
  --runner-binary build/m2m_loransac/m2m_loransac_runner

python -m evaluation.estimation.run_pose_gms_selection summarize \
  --results-root "$POSE_ROOT" --input-manifest "$POSE_ROOT/experiment_inputs.json" \
  --work-root "$WORK_ROOT/q2" --model-name dinov3_vitl16 \
  --layer 19 --debias-rank 200 --method-prefix combined_interpolate_v3 \
  --pipelines gms:4:20:CM gms:4:20:HCM gms:4:20:MCM gms:4:20:HCM_MC \
  --seeds 0 1 2 3 4 --output "$WORK_ROOT/q2/summary/pose_auc.json"

python -m evaluation.visualization.plot_q2_estimator_comparison \
  --input "$WORK_ROOT/q2/summary/auc_five_seed_summary.csv" \
  --figure-output "$WORK_ROOT/q2/figures"
```

The summarizer writes five-seed AUC summaries and five-seed mean runtimes in
the Table 1 layout to `table1_runtime.csv`.

## Q3 / first two panels of Figure 7

For corrected DINOv3, reuse Q2's proposed results/configuration and estimate
only the missing MNN+CM baseline in the same Q2 work tree:

```bash
python -m evaluation.estimation.run_pose_gms_selection estimate \
  --results-root "$POSE_ROOT" --input-manifest "$POSE_ROOT/experiment_inputs.json" \
  --work-root "$WORK_ROOT/q2" --model-name dinov3_vitl16 \
  --layer 19 --debias-rank 200 --method-prefix combined_interpolate_v3 \
  --pipelines mnn:CM --seeds 0 1 2 3 4 \
  --runner-binary build/m2m_loransac/m2m_loransac_runner

python -m evaluation.estimation.run_pose_gms_selection summarize \
  --results-root "$POSE_ROOT" --input-manifest "$POSE_ROOT/experiment_inputs.json" \
  --work-root "$WORK_ROOT/q2" --model-name dinov3_vitl16 \
  --layer 19 --debias-rank 200 --method-prefix combined_interpolate_v3 \
  --pipelines mnn:CM gms:4:20:HCM_MC --seeds 0 1 2 3 4 \
  --output "$WORK_ROOT/q3/dinov3/summary/pose_auc.json"
```

Generate and evaluate the raw DINOv2 layer-24/rank-0 path:

```bash
python -m evaluation.estimation.run_pose_matching all \
  --results-root "$POSE_ROOT" --input-manifest "$POSE_ROOT/experiment_inputs.json" \
  --basis-root "$WORK_ROOT/pose_basis" --cache-root "$WORK_ROOT/pose_cache" \
  --keypoint-cache-root "$WORK_ROOT/keypoint_cache" \
  --model-name dinov2_vitl14_reg --layer 24 --debias-ranks 0 \
  --correction none --source third_party/dinov2 --weights "$DINO2_WEIGHTS" \
  --method-prefix combined_interpolate_dinov2_vitl14_reg --max-ks 1 5 \
  --association-upperbound 2048

python -m evaluation.estimation.run_pose_gms_selection filter \
  --results-root "$POSE_ROOT" --input-manifest "$POSE_ROOT/experiment_inputs.json" \
  --work-root "$WORK_ROOT/q3/dinov2" --model-name dinov2_vitl14_reg \
  --layer 24 --debias-rank 0 \
  --method-prefix combined_interpolate_dinov2_vitl14_reg \
  --candidates 4:20 --gms-binary build/gms_filter/gms_filter_csv_m2m

python -m evaluation.estimation.run_pose_gms_selection estimate \
  --results-root "$POSE_ROOT" --input-manifest "$POSE_ROOT/experiment_inputs.json" \
  --work-root "$WORK_ROOT/q3/dinov2" --model-name dinov2_vitl14_reg \
  --layer 24 --debias-rank 0 \
  --method-prefix combined_interpolate_dinov2_vitl14_reg \
  --pipelines mnn:CM gms:4:20:HCM_MC --seeds 0 1 2 3 4 \
  --runner-binary build/m2m_loransac/m2m_loransac_runner

python -m evaluation.estimation.run_pose_gms_selection summarize \
  --results-root "$POSE_ROOT" --input-manifest "$POSE_ROOT/experiment_inputs.json" \
  --work-root "$WORK_ROOT/q3/dinov2" --model-name dinov2_vitl14_reg \
  --layer 24 --debias-rank 0 \
  --method-prefix combined_interpolate_dinov2_vitl14_reg \
  --pipelines mnn:CM gms:4:20:HCM_MC --seeds 0 1 2 3 4 \
  --output "$WORK_ROOT/q3/dinov2/summary/pose_auc.json"

python -m evaluation.visualization.plot_q3_dino \
  --dinov3 "$WORK_ROOT/q3/dinov3/summary/auc_per_seed.csv" \
  --dinov2 "$WORK_ROOT/q3/dinov2/summary/auc_per_seed.csv" \
  --output-stem "$WORK_ROOT/q3/figures/q3_dino_transfer"
```

Q1 and Q2 are DINOv3 mechanism studies. DINOv2 enters the main body in
Section 2 and Q3.
