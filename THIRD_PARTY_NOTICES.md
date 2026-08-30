# Third-party notices

The root BSD-3-Clause license applies only to this project's original code.
Dependencies and adapted components retain their own terms.

## Distributed as Git submodules

- [DINOv2](https://github.com/facebookresearch/dinov2), pinned at
  `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, is Apache-2.0, copyright
  Meta Platforms, Inc. and affiliates. No checkpoint is distributed. The
  main-paper DINOv2 path uses the official ViT-L/14 model with four register
  tokens (`dinov2_vitl14_reg`).
- [DINOv3](https://github.com/facebookresearch/dinov3), pinned at
  `6876159a11b4df116f30f667f8c9888617df0751`, is governed by the custom
  DINOv3 License included in that submodule. No checkpoint is distributed.
- [PoseLib](https://github.com/PoseLib/PoseLib), pinned at
  `fa7280fee27f97aff31ae7f98bab7f583fac7d08`, is BSD-3-Clause, copyright
  Viktor Larsson and contributors. This commit is the version compatible with
  the estimator API in this release.

## Adapted or technique-derived code

- The many-to-many GMS implementation is adapted from
  [GMS-Feature-Matcher](https://github.com/JiawangBian/GMS-Feature-Matcher),
  BSD-3-Clause, copyright 2017 JiaWang Bian. Its license notice is retained in
  the source tree.
- Positional-bias correction follows the training-free projection proposed by
  [INSID3](https://github.com/visinf/INSID3), Apache-2.0. No INSID3 source is
  vendored.

## External SuperPoint adapter

The repository does not distribute SuperPoint implementation code or weights.
The keypoint adapter can consume caches produced with
[LightGlue](https://github.com/cvg/LightGlue), which is Apache-2.0. The original
[Magic Leap SuperPoint release](https://github.com/magicleap/SuperPointPretrainedNetwork)
is restricted to academic/non-profit, noncommercial research and forbids
redistribution. Users are responsible for selecting a lawful implementation
and complying with its license.

The tested LightGlue revision is
`eb42fee2d71449efb0aa5c10549752b5d75384d8`. Its default SuperPoint adapter may
download `superpoint_v1.pth` on first use; neither that checkpoint nor a copy of
LightGlue is distributed here.

Dataset terms are summarized separately in `data/README.md` and are not
superseded by this repository's license.

## Dataset attribution

The tracked split lists refer to the
[NAVI dataset](https://navidataset.github.io/), released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Its accompanying
code is Apache-2.0. Users should cite Varun Jampani et al., *NAVI:
Category-Agnostic Image Collections with High-Quality 3D Shape and Pose
Annotations*, NeurIPS 2023, and comply with the attribution terms published by
the dataset authors.
