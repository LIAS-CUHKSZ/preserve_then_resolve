# Data preparation

No images, model weights, generated descriptors, match files, or pose results
are distributed in this repository. Keep generated files under the ignored
`artifacts/` tree, or point `configs/paths.local.yaml` at external storage.
The local config is also ignored by Git.

The pose-input generator accepts an external dataset tree directly:

```bash
python -m evaluation.estimation.prepare_pose_estimation_data \
  --rss-root . --data-root /path/to/data \
  --output-root /path/to/generated/matching_estimation_results
```

Raw images and external benchmark metadata are read from `--data-root`; the
versioned NAVI pair lists remain in this repository under `data/splits` and do
not need to be copied or linked into the external tree.

## Datasets

Download each dataset from its official source and comply with its terms:

- The [NAVI dataset](https://navidataset.github.io/) is released under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); its accompanying
  code is released under Apache-2.0. Attribute the dataset and cite Jampani et
  al., *NAVI: Category-Agnostic Image Collections with High-Quality 3D Shape
  and Pose Annotations* (NeurIPS 2023), as requested by the authors.
  The geometric-rank patch evaluator reads NAVI RGB, depth, masks, and
  annotations directly. Generate normalized pose/intrinsics CSVs with
  `splits/navi/estimation/generate_pose_intrinsics.py`; the generator also
  writes a pair-file checksum binding beside every pose CSV.
- [ScanNet](https://github.com/ScanNet/ScanNet) requires an approved Terms of
  Use agreement. This repository cannot redistribute ScanNet images.
- [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/) contains images
  collected from the internet; follow the project's download and usage notes.
- [METU-VisTIR](https://github.com/OnderT/XoFTR#metu-vistir-dataset) is released
  under CC BY-NC-SA 4.0 and is downloaded separately from the XoFTR project.

ScanNet and MegaDepth pose preparation also needs the benchmark pair/GT
metadata below. These files are inputs governed by their upstream dataset or
benchmark terms and are not redistributed here:

```text
<data-root>/scannet/pairs_scannet_test_pairs_with_gt.txt
<data-root>/megadepth/megadepth_1500_scales/all_pairs_with_gt.txt
```

The ScanNet list is the 1,500-pair file published with the
[SuperGlue evaluation](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/assets/scannet_test_pairs_with_gt.txt).
The MegaDepth input is the paper's 1,491-pair `pairs_test_1500.txt` list in the
same pair/GT text format. The generator fails closed unless their SHA-256
values are respectively
`522ee01d4e18b5d0182ed934aa1cab9896183ee3321f1ba97fabc77867c412a2`
and
`bd2d6d843fa573ef15665603b8fdb32830efc8345d98908c340822e5f7f17c80`.
This makes a missing paper input explicit instead of silently changing the
evaluation set.

The estimation evaluator expects the result tree documented in
`evaluation/README.md`; it does not prescribe where raw images are stored.
Convert every dataset's pose/intrinsics metadata to the common convention in
`schemas/pose_intrinsics_csv.md` before estimation or evaluation.

## Bundled NAVI splits

The canonical NAVI-Wild files are
`splits/navi/estimation/pairs_wildset_*.csv` and
`splits/navi/correspondence/pairs_wildset_*.csv`. Their estimation and
correspondence image unions are globally disjoint, while both sides have
matched object quotas and 5-degree angle histograms. The shared README,
manifest, image assignment, generator, and checksums live in `splits/navi/`.

`splits/navi/estimation/` contains both the three 500-pair NAVI-Multi pose
files (`pairs_multiview_*`) and the NAVI-Wild pose files
(`pairs_wildset_*`). Each pair-list CSV uses:

```csv
image_1,image_2,angular_distance_degrees,image_1_height,image_1_width,image_2_height,image_2_width
```

Dimension columns contain original image sizes. They are the source of truth
for enumerating processed sizes in `dino-m2m fit-bias-for-pairs`; the command
then applies the same configured long-edge resize policy as DINO extraction.
The NAVI-Wild sampler reads these dimensions directly from each selected image
and verifies them against NAVI annotations before publishing any split file.

Do not resample or silently edit the generated NAVI-Wild split. Verify it with:

```bash
(cd data/splits/navi && sha256sum -c SHA256SUMS)
```

Each of the NAVI-Multi and NAVI-Wild families covers 35 objects, not all 36
NAVI objects. Multi excludes `shoe_right_gray_s`; Wild excludes
`bottle_vitamin_d_tablets`. Their union contains 36 objects.

Write NAVI-Wild pose artifacts below
`artifacts/matching_estimation_results/NAVI_wild/` and dense
artifacts below a separate correspondence root.
