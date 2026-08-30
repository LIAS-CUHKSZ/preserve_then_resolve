# NAVI splits

- `estimation/pairs_wildset_*.csv` contains the pose-estimation pairs, both NAVI-Multi and NAVI-Wild.
- `correspondence/pairs_wildset_*.csv` contains the dense-correspondence pairs, NAVI-Wild only.

## NAVI-Wild joint split

The Wild split partitions images before pair selection, making its estimation
and correspondence image unions globally disjoint. The shared contract is
stored at this directory level:

- `manifest.json` records the protocol, input identities, counts, and audit;
- `image_partition.csv` records the estimation/correspondence assignment;
- `partition_capacity.csv` records unused edge capacity and angle quotas;
- `SHA256SUMS` covers both Wild pair families and all shared metadata;
- `generate.py` deterministically samples the complete joint split and writes
  original image height and width directly into every selected pair row.

For each of the 35 objects, all 2,180 non-occluded images represented in the
source graph are divided once between estimation and correspondence. Each
family has 500 pairs in every angular bin, matched per-object quotas and
five-degree angle histograms, and exactly 250 deterministically reversed rows
per family/bin. Pair selection uses no descriptor, mask, depth, camera
translation, or evaluation output.

Pair IDs restart at 1 in each CSV. Keep the two Wild families in separate
artifact roots and use their binding manifests rather than identifying a pair
by its integer ID alone.

Image sizes are part of sampling, not a post-processing step. For every
selected endpoint, `generate.py` opens the source image, reads `(width,
height)`, verifies it against `annotations.json`, and emits
`image_1_height,image_1_width,image_2_height,image_2_width`. A missing image or
dimension mismatch aborts before any canonical split file is replaced.

## Regeneration and verification

Place NAVI at `data/navi/`, including `pairs-wild_set.txt`, then run:

```bash
python data/splits/navi/generate.py
python -m unittest tests.python.test_navi_wild_joint_split -v
(cd data/splits/navi && sha256sum -c SHA256SUMS)
```

Set `NAVI_FULL_REGEN_TEST=1` on the unittest command to enable the full
byte-for-byte regeneration check. The generator materializes and validates
all output before replacing files, publishes `manifest.json` after the data
files, and writes `SHA256SUMS` last as the completion marker. Existing outputs
are not replaced unless `--overwrite` is explicitly supplied.

Use separate derived roots:

```text
artifacts/matching_estimation_results/NAVI_wild/
artifacts/matching_estimation_results/NAVI_correspondence/
artifacts/dense_correspondence/rank_evaluation/
```
