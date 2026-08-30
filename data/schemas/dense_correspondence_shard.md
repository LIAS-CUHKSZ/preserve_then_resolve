# Dense correspondence rank shard

One compressed NPZ is written for each `(angular_bin, pair_index, layer)`:

```text
bin_<angular-bin>/shards/layer<layer>/pair_<pair-index:06d>.npz
```

Schema version 3 stores all positional-debiasing ranks for that layer. Every
shard embeds the canonical protocol and SHA-256 fingerprint, pair metadata,
unpadded/padded image sizes, each image's object-center patch-index vector,
`debias_ranks [D]`, and `max_k` (8 by default).

For each `a_to_b` and `b_to_a` direction it stores:

- `source_patch_index [N] int64`: complete object/depth-valid queries;
- `source_min_object_error_m [N] float64`: minimum 3D error to any valid
  destination-object patch center, independent of descriptors;
- `candidate_target_patch_index [D,N,K] int64`;
- `candidate_cosine [D,N,K] float32`, ordered high to low;
- `candidate_target_is_object [D,N,K] bool`;
- `candidate_target_has_depth [D,N,K] bool`;
- `candidate_error_m [D,N,K] float64`;
- `candidate_mutual_entry_k [D,N,K] int16`.

The `candidate_*` arrays use the complete grid and include background, matching
the Section 2 protocol in the main paper.

`candidate_error_m` remains finite for a background candidate when NAVI depth
exists. A candidate is geometrically correct only when it has object support,
valid depth, and error strictly below the selected threshold. Missing depth
uses `+inf`.

The one-based mutual-entry value of an edge is the maximum of its two
directional ranks. `K+1` means the edge is not mutual within the stored range.
`mutual_match_count_at_k [D,K] int64` counts cumulative mutual edges on the
two complete, unmasked patch grids; it is a complexity proxy rather than the
SuperPoint association count used by the final pipeline.

Resume rejects incompatible fingerprints, missing arrays, wrong dtypes or
shapes, padded indices, duplicate candidates, non-monotone cosine rows,
invalid error/depth tuples, and impossible mutual ranks or edge counts.
`evaluation_manifest.json` is the completeness contract: standalone summary
requires exactly its layer-by-pair shard rectangle and rejects extra, missing,
mixed-protocol, or mismatched pair-identity files.

Only top-K candidates are retained. Thresholds and any smaller K can be
recomputed from a shard, but increasing K requires rerunning descriptor
matching with the cached descriptors and bases.
