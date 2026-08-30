# Association CSV

One file represents one image pair and is named `matching_NNN.csv`, with a
one-based, zero-padded pair number. The canonical columns are:

| Column | Type | Meaning |
| --- | --- | --- |
| `left_idx`, `right_idx` | integer | Compact keypoint IDs in the two images |
| `x1`, `y1`, `x2`, `y2` | float | Pixel coordinates after the shared long-edge resize, before patch padding |
| `similarity` | float | DINO cosine similarity |
| `k` | integer | First Progressive-MKNN rank at which the edge was accepted |

Rows are ordered by increasing `k`, then decreasing `similarity` within each
rank. The estimator also accepts `idx1`/`idx2` aliases and an optional `prob`
column.

Camera intrinsics and dense correspondence annotations must use this same
resized coordinate system.
