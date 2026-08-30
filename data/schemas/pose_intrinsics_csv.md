# Pose and intrinsics CSV

`pose_intrinsics.csv` has one row per pair. Required columns are:

```text
pair_idx,fx1,fy1,cx1,cy1,fx2,fy2,cx2,cy2,qw,qx,qy,qz,tx,ty,tz
```

Quaternions use scalar-first order (`wxyz`). Optional `dist0_coeffs` and
`dist1_coeffs` cells contain comma-separated OpenCV coefficients. Pair IDs
must be unique and agree with the number in `matching_NNN.csv`.
Intrinsics must be expressed in the same long-edge-resized pixel coordinates
as the association CSVs.

The stored quaternion and translation define the relative transform from the
first camera to the second:

```text
X_camera2 = R(qw, qx, qy, qz) * X_camera1 + t
```

Preprocess dataset-native ground truth into this convention before writing the
CSV. Geometry and evaluation use `R, t` exactly as stored; they do not invert
or auto-select a convention. Historical result files may contain a
`gt_mode_used` column, but current tools ignore it and do not generate it.

For generated NAVI metadata, `pose_intrinsics_manifest.json` sits beside the
CSV and records the source pair-list SHA256, ordered pair-identity SHA256,
pair count, resize long edge, and pose CSV SHA256. Pair IDs alone are not a
safe binding because every split/bin restarts numbering at 1.
