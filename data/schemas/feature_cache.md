# Feature cache formats

## DINO descriptors

Each current-schema `.dino.npz` archive contains `descriptor_map`, `patch_size`,
`proc_hw`, and `orig_hw`. Spatial sizes are `[height, width]`. Here `proc_hw`
is the patch-padded tensor size, while the legacy field name `orig_hw` denotes
the long-edge-resized size before padding (not necessarily the raw image
size). Descriptors remain in the representation used by extraction until
matching performs interpolation and normalization.

The archive also records `model_name`, `layer`, `weights_id` (a SHA-256
checkpoint identifier), `long_edge`, `downscale_only`, `normalization_id`,
`resize_id`, and `padding_id`. Extraction validates the full archive before
skipping it. Matching validates these fields against the requested model,
checkpoint, and preprocessing; an old or mismatched cache must be regenerated
with `extract-dino --overwrite`.

## Positional-bias bases

Each size-specific basis is named
`layerN/dinov3_vitl16_<height>x<width>_basis.pt` below the configured basis
root, where height and width are the patch-padded DINO input dimensions. Its
payload stores the maximum-rank matrix
in `basis`, its column count in `max_rank`, and provenance in `meta`.
The metadata binds the basis to its DINO model, layer, checkpoint SHA-256,
normalization, patch size, and padded image height/width. A requested rank
`k <= max_rank` uses `basis[:, :k]`, so lower-rank matrices are not duplicated
in the file. Any other payload layout is rejected. Regenerate incompatible
files with `fit-bias-for-pairs`.

## SuperPoint keypoints

Each `.spkp.npz` archive contains `keypoints` as an `N x 2` float array in
`[x, y]` order. New caches also record image dimensions, preprocessing
metadata, and the external SuperPoint weight identifier. For these caches,
`orig_hw` is the raw input size and `proc_hw` is the long-edge-resized size.
The matcher requires this `proc_hw` to equal the DINO cache's unpadded
`orig_hw`. Custom SuperPoint checkpoint identifiers are content SHA-256 values,
not filenames. For LightGlue's bundled default weights, the identifier records
the installed LightGlue version, model variant, and a stable hash of the
extractor state dict. `extract-keypoints` prints this value; cache-only runs
must provide it through `superpoint.weights_id` or
`--superpoint-weights-id`. Legacy-path archives with incomplete metadata are
used only with the explicit `--allow-legacy-keypoint-cache` opt-in; first
verify their resize policy and checkpoint manually.
