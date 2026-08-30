# Many-to-Many GMS Filter

This component applies the paper's many-to-many adaptation of Grid-based
Motion Statistics to association CSV files. It retains the maximum bipartite
matching cardinality used for each grid-cell pair and omits the unrelated
classic one-to-one executable.

Build it from the repository root after installing OpenCV core and features2d:

```bash
cmake -S . -B build -DDINO_M2M_BUILD_LORANSAC=OFF
cmake --build build --target gms_filter_csv_m2m -j
```

Filter every `matching_<id>.csv` in a method directory:

```bash
./build/gms_filter/gms_filter_csv_m2m \
  --root artifacts/matching_estimation_results/NAVI_wild/wildset_0-40/combined_interpolate_v3/layer19/debias_svd200/progressive_k5 \
  --threshold-factor 4.0 --grid-size 20 --auto-mask 1
```

Input requires `left_idx,right_idx,x1,y1,x2,y2`; additional columns are
preserved. Invalid numeric rows are skipped, a header-only file produces a
header-only result, and retained feature IDs are compacted deterministically.
The output directory is created beside the input and encodes the filter
parameters in its name. Keep the dataset's `pose_intrinsics.csv` at the subset
root documented in the repository README; it is shared by all method
directories for that subset. If the input contains `association_manifest.json`,
GMS copies it unchanged to the filtered directory.
