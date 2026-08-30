# Pipeline guide

The main DINO path has five explicit stages:

```text
DINO layer maps
  -> optional DINOv3 positional-bias projection
  -> SuperPoint-location descriptor interpolation
  -> progressive mutual KNN association
  -> GMS and/or ambiguity-aware LO-RANSAC
```

## 1. Feature maps

`dino-m2m extract-dino` supports exactly the two backbones used by the main
paper release:

- `dinov3_vitl16`, patch 16, layers 1--24;
- `dinov2_vitl14_reg`, patch 14, layers 1--24.

Feature caches bind model name, layer, preprocessing, source revision, and
checkpoint SHA-256. DINOv2 is raw-only. DINOv3 can use the training-free
positional-bias projection fitted by `fit-bias` or `fit-bias-for-pairs`.

## 2. Local support and interpolation

SuperPoint supplies at most 2,048 keypoint locations per image. DINO patch
features are bilinearly interpolated at those locations and L2-normalized.
The keypoint and DINO caches must record the same processed image geometry.

## 3. Progressive MKNN

For each left descriptor, progressive MKNN preserves every mutual edge that
first appears by K. Association CSVs record that first K in the `k` column,
which makes E1 ⊆ E2 ⊆ ... directly recoverable without recomputing cosine
similarities.

Section 2 and Q1 use uncapped graphs. The practical Q2/Q3 system retains at
most 2,048 association rows before GMS and passes at most 1,024 rows to the
estimator. These are different experimental contracts and should use separate
output trees.

## 4. GMS

`gms_filter_csv_m2m` applies the paper's many-to-many adaptation of Grid-Based
Motion Statistics. The main configuration is threshold factor 4, grid size
20, automatic masking, five-scale search, and eight-rotation search. The
filter copies the raw association manifest and writes its own settings
manifest.

## 5. Robust estimation

`m2m_loransac_runner` provides CM, HCM, MCM, and the two-stage HCM→MCM mode
(`HCM_MC`). Q1 fixes proposals to E1 with `proposal_max_k=1` while using EK for
scoring, refinement, and reranking. Q2 compares estimator modes on one fixed
post-GMS graph. Q3 compares complete rank-one and preserve-then-resolve
pipelines within each representation.

See [`experiments/`](../experiments/) for fixed parameters and
[`data/schemas/`](../data/schemas/) for file contracts.
