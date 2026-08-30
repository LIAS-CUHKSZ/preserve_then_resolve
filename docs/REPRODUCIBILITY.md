# Reproducibility status

This repository publishes the code paths and fixed contracts for Section 2
and Q1--Q3. It does not redistribute licensed datasets, DINO checkpoints,
LightGlue/SuperPoint weights, dense NAVI derivatives, descriptor caches, or
the paper's multi-hundred-gigabyte result tree.

## Checkable without paper artifacts

- Python unit tests for feature/association schemas, MKNN, correspondence
  geometry and aggregation, seeded pose summaries, and all three body figure
  renderers.
- C++ unit tests and the toy many-to-many example after initializing PoseLib.
- Integrity and image-disjointness checks for the tracked NAVI split lists.
- Command-line parsing and exact Q1/Q2/Q3 protocol generation.

## Needed for numeric reproduction

- the datasets listed in `data/README.md`, obtained under their original
  licenses;
- official DINOv3 ViT-L/16 and DINOv2 ViT-L/14-register checkpoints;
- the official DINOv2/DINOv3 source submodules and a compatible LightGlue
  installation;
- normalized pair/intrinsics/pose metadata following `data/schemas/`;
- the exact ScanNet/MegaDepth benchmark pair/GT lists and checksums documented
  in `data/README.md`;
- GPU/storage capacity for layers 16--24, DINOv3 rank bases through 600, and
  the six-dataset association/result trees.

Large inputs may be mounted or symlinked from outside the repository. All
commands accept explicit roots; no experiment requires copying those files
into the source checkout.

Successful unit tests establish implementation and format consistency, not
the paper's reported numeric values. A clean end-to-end reproduction requires
the original licensed inputs and substantial compute. The fixed experiment
contracts are recorded in `experiments/*.yaml`.
