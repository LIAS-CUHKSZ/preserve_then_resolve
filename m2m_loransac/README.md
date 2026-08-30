# Many-to-Many LO-RANSAC

This directory contains the reusable relative-pose estimator and a separate
CSV experiment harness. The estimator implements CM, HCM, MCM, and the
two-stage HCM_MC experiment used in the main paper.

## Build

Eigen 3.4 or newer is a required build and exported-package dependency. On
Debian/Ubuntu, install it with `sudo apt install libeigen3-dev`. Then initialize
the pinned PoseLib submodule before configuring:

```bash
git submodule update --init third_party/PoseLib
cmake -S m2m_loransac -B build/loransac -DCMAKE_BUILD_TYPE=Release
cmake --build build/loransac -j
```

To use an installed PoseLib instead, pass
`-DM2M_LORANSAC_USE_BUNDLED_POSELIB=OFF -DPoseLib_DIR=...`. When the bundled
copy is used, `cmake --install` installs that pinned PoseLib alongside this
package so downstream `find_package(DinoM2MLORansac)` calls can resolve the
public dependency. Per-pair debug output is opt-in through
`-DM2M_LORANSAC_VERBOSE=ON`.

The installed CMake package exports `DinoM2M::loransac` and
`DinoM2M::hopcroft_karp`. The public estimator is declared in
`m2m_loransac/relative_pose_m2m.h`:

```cpp
estimate_relative_pose_m2m(points1, points2, camera1, camera2,
                           left_ids, right_ids, probabilities, delta,
                           mode, top_n, options, &pose, &inliers);
```

All association vectors must have equal length, feature IDs must be
non-negative, and probabilities must be finite and non-negative.

## Examples and Dataset Runner

Run the synthetic many-to-many example from the repository root:

```bash
./build/loransac/m2m_loransac_toy m2m_loransac/examples/toy_matches.csv HCM_MC 0.3
```

For a result tree, copy `examples/config.example.cfg`, edit its paths, then run
`m2m_loransac_runner CONFIG`. Each method directory contains `matching_<id>.csv`
with `left_idx,right_idx,x1,y1,x2,y2`; each dataset directory contains
`pose_intrinsics.csv`. Ground-truth poses in that file must use the stored
image-1-to-image-2 convention; runtime convention switching is intentionally
unsupported. Relative paths are resolved from the config file.
Malformed pair files are recorded as padded `skipped` rows instead of changing
the result CSV width.

For a controlled proposal-source experiment, set `proposal_max_k=N`. The
runner then constructs the minimal five-point sampling graph only from rows
whose `k_first` value is at most `N`; the existing `k` column is accepted as a
backward-compatible alias. The CSV rank is required and must be a positive
integer. Missing or invalid metadata produces an explicit skipped row and
never falls back to full-graph sampling. The setting is available only for
HCM, MCM, and HCM_MC and requires
`min_iterations=max_iterations`, `max_matching_num=0`, and
`similarity_threshold=0`. Probabilities, HCM/MCM scoring, all LO and refinement
steps, HCM_MC Stage-2 ranking, final polish, and output inliers continue to use
the complete association graph. Active output filenames include
`_proposal_k_<N>` to prevent accidental resume into a run with different
proposal semantics. `proposal_max_k=0` preserves the prior behavior and output
names.

For a read-only reference-hypothesis diagnostic on an HCM_MC run, set
`write_gt_hypothesis_probe=true`, `init_with_gt=false`, and `ransac_times=1`.
After Stage 1 freezes its bounded raw-seed pool, the runner snapshots the
strict HCM admission cutoff; only after Stage 2 and final polish complete does
it score the stored benchmark pose once. The score is never inserted into the
pool and is excluded from stopping, LO, MCM ranking, output-pose selection, and
the estimator timing counters. A separate
`*_gt_hypothesis_probe.csv` sidecar records the pool size, cutoff, reference
score, reference-consistent edge count, and whether the reference would enter
as one additional raw hypothesis. This diagnostic does not report oracle pose
accuracy, and a low-residual edge is not thereby certified as a true match.

The runner requires `pose_intrinsics_manifest.json` at the dataset root and an
`association_manifest.json` in the method directory by default. Before reading
or resuming results, it compares their pair-file and ordered-pair hashes,
verifies the pose CSV checksum and pair count, and requires the pose and match
file ID sets to agree. For a reviewed historical tree that predates sidecars,
set `allow_unbound_pose=true` explicitly in its copied config; never use that
escape hatch for current split versions.

The repository-wide convention is
`artifacts/matching_estimation_results/<root_name>/<subset_name>/`: set
`matching_result_root` through `<root_name>`, use `<subset_name>` in
`datasets`, and make `method` the path below the subset directory.
