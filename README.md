# Preserve, Then Resolve

Code for **“Preserve, Then Resolve: Many-to-Many Association and Robust
Estimation with General-Purpose Visual Features.”**

The paper is available on arXiv as
[“Preserve, Then Resolve: Many-to-Many Association and Robust Estimation with
General-Purpose Visual Features” (arXiv:2604.23670)](https://arxiv.org/abs/2604.23670).

The semantic transferability of general-purpose visual features
does not guarantee geometric consistency across images. Our paper
shows that geometrically correct correspondences often fall below
rank one in cosine similarity yet remain within  a small top-K
candidate set. Based on this observation, we tailor a preserve-then-resolve
framework for better exploiting general-purpose visual features
in geometric estimation.

This repository tracks the main-paper evidence chain:

| Paper component | Public experiment |
| --- | --- |
| Section 2 / Figure 2 | correspondence recall vs. association count across DINOv3\&DINOv2 layers and positional-bias ranks; |
| Q1 / Figure 5 | fixed-E1 proposals with nested E1..E5 scoring graphs, five seeds, and reference-hypothesis probe |
| Q2 / Figure 6 / Table 1 | CM, HCM, MCM, and HCM→MCM on the same post-GMS K5 graph |
| Q3 / Figure 7 (first two panels) | MNN+CM versus K5+GMS+HCM→MCM for corrected DINOv3 and raw DINOv2 |

## Repository layout

```text
src/dino_m2m/                 DINOv2/v3 extraction, debiasing, and MKNN
evaluation/dense_correspondence/  Section 2 evaluation and Figure 2
evaluation/estimation/        Q1--Q3 pose experiment drivers
evaluation/visualization/     main-paper Figure 5--7 renderers
gms_filter/                   many-to-many GMS executable
m2m_loransac/                 HCM/MCM LO-RANSAC executable
experiments/                  exact paper contracts and runnable recipes
data/schemas/                 pair, feature, association, and pose formats
```

The most direct starting point is
[experiments/README.md](experiments/README.md).

## Installation

Clone with the pinned DINOv2, DINOv3, and PoseLib submodules:

```bash
git clone --recurse-submodules https://github.com/LIAS-CUHKSZ/preserve_then_resolve.git
cd preserve_then_resolve
conda env create -f environment.yml
conda activate dino-m2m
```

SuperPoint keypoints are produced through the official LightGlue package,
which is not vendored:

```bash
git clone https://github.com/cvg/LightGlue.git ../LightGlue
git -C ../LightGlue checkout eb42fee2d71449efb0aa5c10549752b5d75384d8
python -m pip install -e ../LightGlue
python -m pip install -e .
```

Build the geometry stages with CMake, Eigen3, OpenCV, and the pinned PoseLib
checkout:

```bash
cmake -S . -B build -G Ninja -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The tested PoseLib revision is
`fa7280fee27f97aff31ae7f98bab7f583fac7d08`; PoseLib v2.0.5 has an incompatible
API. DINO checkpoints are never downloaded implicitly: place the official
DINOv3 ViT-L/16 and DINOv2 ViT-L/14-register checkpoints locally and pass
their paths explicitly. On its first uncached SuperPoint extraction,
LightGlue may download its bundled `superpoint_v1.pth`; later runs validate
the resulting state hash against the paper contract.

## Data and generated artifacts

Licensed datasets, model checkpoints, descriptor caches, and paper result
trees are not distributed. Copy `configs/paths.example.yaml` to the ignored
`configs/paths.local.yaml`, or pass paths on the command line. Large data and
artifact directories may be symlinked from another location; the code does
not require them to live inside the repository.

All pose inputs use one result-tree contract:

```text
RESULTS_ROOT/<root_name>/<subset_name>/
├── pairs.csv
├── pose_intrinsics.csv
└── <method>/layer<L>/debias_svd<R>/
    ├── progressive_k1/
    ├── progressive_k5/
    └── progressive_k5_GMSm2m_.../
```

Relative poses must already map camera-1 coordinates to camera-2 coordinates.
The evaluators do not guess or invert dataset-specific conventions at runtime.
See [data/README.md](data/README.md) and the schemas in `data/schemas/`.

## Checks

Dependency-light Python tests:

```bash
PYTHONPATH=src:. pytest -q
PYTHONPATH=src:. python -m unittest discover -s evaluation/tests -v
```

For a quick command-line audit without datasets:

```bash
python -m evaluation.estimation.run_q1_nested_k --help
python -m evaluation.estimation.run_pose_gms_selection --help
python -m evaluation.visualization.plot_q3_dino --help
```

Unit tests and toy C++ tests validate formats and algorithms; they do not
claim end-to-end reproduction of reported numbers without the original
licensed inputs. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## License and citation

Original project code is BSD-3-Clause. DINOv2, DINOv3, PoseLib, LightGlue,
model weights, and datasets retain their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Please cite the paper and software using [CITATION.cff](CITATION.cff).
