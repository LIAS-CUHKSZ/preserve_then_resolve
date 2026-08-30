# NAVI pose-estimation splits

The three `pairs_multiview_*.csv` files preserve the rows and ordering of the
original `NAVI_splits/1500pairs` NAVI-Multi inputs in a headered CSV format.
There are three angular bins and 500 pairs per bin.
`generate_pose_intrinsics.py`
derives normalized pose metadata from NAVI's per-image annotations. Its stored
pose maps camera 1 to camera 2, as required by
`data/schemas/pose_intrinsics_csv.md`. The `pairs_wildset_*.csv` files in this
same directory are the canonical NAVI-Wild estimation split.

Each row also stores the original height and width of both images in
`image_1_height`, `image_1_width`, `image_2_height`, and `image_2_width`.
For NAVI-Wild, these values are read and validated directly by
`../generate.py` while pairs are sampled; there is no separate size-annotation
step. The bundled NAVI-Multi CSVs already contain their imported dimensions.

Regenerate the CSVs from a downloaded NAVI tree with:

```bash
python data/splits/navi/estimation/generate_pose_intrinsics.py \
  --navi-root PATH/TO/navi_v1.0 --family multiview --overwrite
```

The generator reproduces the experiment's 1024-pixel long-edge intrinsics and
writes the NAVI-Multi outputs directly to
`artifacts/matching_estimation_results/NAVI_resized/<subset>/pose_intrinsics.csv`.
The generated files therefore live beside each subset's method directories and
remain outside version control with the rest of `artifacts/`.
Use `artifacts/matching_estimation_results/NAVI_resized` as the C++ runner's
`matching_result_root`; the three subset directory names are the pair-list
stem after `pairs_`.

For NAVI-Wild, pass its canonical split directory and result namespace:

```bash
python data/splits/navi/estimation/generate_pose_intrinsics.py \
  --navi-root PATH/TO/navi_v1.0 \
  --splits-dir data/splits/navi/estimation \
  --family wildset \
  --output-root artifacts/matching_estimation_results/NAVI_wild
```

Each output also receives `pose_intrinsics_manifest.json`, which binds the pose
rows to the exact pair-list SHA256 and ordered pair identities.

`manifest.csv` and the local `SHA256SUMS` cover the NAVI-Multi imports. The
NAVI-Wild files are covered by `../manifest.json` and `../SHA256SUMS`. Check
the Multi files from the repository root with
`sha256sum -c data/splits/navi/estimation/SHA256SUMS`.

The NAVI-Multi files cover 35 object directories and omit
`shoe_right_gray_s`.
