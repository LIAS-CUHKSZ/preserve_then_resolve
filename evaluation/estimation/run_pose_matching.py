#!/usr/bin/env python3
"""Run DINO pose associations for one layer and one or more debiasing ranks.

The input manifest is produced by ``prepare_pose_estimation_data.py``.  This
script deliberately stops at association generation: GMS filtering, robust
estimation, and model selection are separate so every GMS candidate reuses the
same immutable MNN and Progressive-MKNN associations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_LAYER = 19
DEFAULT_DEBIAS_RANKS = (200,)
DEFAULT_MODEL_NAME = "dinov3_vitl16"
MAX_KEYPOINTS = 2048
LONG_EDGE = 1024
SUPERPOINT_WEIGHTS_ID = (
    "lightglue:0.0:superpoint-default:"
    "sha256:bf5c39ab8163bb20479d061798fee91481649cbe35421fba752914a8a9b87c58"
)
GROUP_ORDER = ("scannet", "megadepth", "navi", "metu")
CANONICAL_DATASETS = (
    "ScanNet",
    "MegaDepth",
    "NAVI-Multi",
    "NAVI-Wild",
    "METU-CC",
    "METU-CS",
)
DEFAULT_WEIGHTS = {
    "dinov3_vitl16": "data/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov2_vitl14_reg": "data/models/dinov2/dinov2_vitl14_reg4_pretrain.pth",
}


@dataclass(frozen=True)
class Subset:
    dataset_label: str
    root_name: str
    subset_name: str
    image_root: Path
    pair_file: Path
    downscale_only: bool

    @property
    def group(self) -> str:
        if self.root_name == "scannet_resized":
            return "scannet"
        if self.root_name == "megadepth_resized":
            return "megadepth"
        # Keep the historical NAVI_resized tree readable while treating the
        # canonical NAVI_wild result tree as the same feature-cache group.
        if self.root_name.startswith(("NAVI_resized", "NAVI_wild")):
            return "navi"
        if self.root_name == "METU_VisTIR_resized":
            return "metu"
        raise ValueError(f"Unknown result-tree root: {self.root_name}")


@dataclass(frozen=True)
class PoseBackbone:
    model_name: str
    model_family: str
    source: Path
    weights: Path
    patch_size: int
    layer: int
    extraction_layers: tuple[int, ...]
    correction: str
    ranks: tuple[int, ...]
    method_prefix: str
    basis_filename_template: str
    layer_selection: Path | None
    layer_selection_payload: dict[str, Any] | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _relative_method_prefix(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("--method-prefix must be relative and may not contain `..`")
    return path.as_posix()


def _load_layer_selection(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: layer selection must be a JSON object")
    candidates = [
        int(payload[key])
        for key in ("best_layer", "selected_layer")
        if key in payload
    ]
    if not candidates:
        raise ValueError(f"{path}: missing `best_layer` (or `selected_layer`)")
    if len(set(candidates)) != 1:
        raise ValueError(f"{path}: best_layer and selected_layer disagree")
    payload = dict(payload)
    payload["best_layer"] = candidates[0]
    return payload


def _resolve_backbone(args: argparse.Namespace, rss_root: Path) -> PoseBackbone:
    from dino_m2m.dino import (
        get_backbone_profile,
        source_checkout_provenance,
        validate_correction_ranks,
    )
    from dino_m2m.provenance import checkpoint_identity

    profile = get_backbone_profile(args.model_name)
    selection_path = (
        args.layer_selection.expanduser().resolve()
        if args.layer_selection is not None
        else None
    )
    selection = _load_layer_selection(selection_path) if selection_path else None
    selected_layer = int(selection["best_layer"]) if selection else None
    if args.layer is not None and selected_layer is not None and args.layer != selected_layer:
        raise ValueError(
            f"--layer={args.layer} disagrees with {selection_path}: {selected_layer}"
        )
    layer = args.layer if args.layer is not None else selected_layer
    if layer is None:
        layer = DEFAULT_LAYER
    if layer < 1 or layer > profile.depth:
        raise ValueError(
            f"Layer {layer} is outside 1..{profile.depth} for {profile.model_name}"
        )

    patch_size = profile.patch_size if args.patch_size is None else args.patch_size
    if patch_size != profile.patch_size:
        raise ValueError(
            f"--patch-size={patch_size} disagrees with {profile.model_name} "
            f"profile ({profile.patch_size})"
        )
    requested_ranks = (
        tuple(args.debias_ranks)
        if args.debias_ranks is not None
        else ((0,) if profile.correction_policy == "none" else DEFAULT_DEBIAS_RANKS)
    )
    correction, ranks = validate_correction_ranks(
        profile.model_name, requested_ranks, args.correction
    )
    extraction_layers = tuple(sorted(set(args.extract_layers or (layer,))))
    if not extraction_layers or any(
        value < 1 or value > profile.depth for value in extraction_layers
    ):
        raise ValueError(
            f"--extract-layers must lie in 1..{profile.depth} for {profile.model_name}"
        )
    if layer not in extraction_layers:
        raise ValueError("--extract-layers must include the layer used for matching")

    if args.source is not None:
        source = args.source.expanduser().resolve()
    else:
        source = (rss_root / "third_party" / profile.family).resolve()
    weights_relative = DEFAULT_WEIGHTS.get(profile.model_name)
    if args.weights is not None:
        weights = args.weights.expanduser().resolve()
    elif weights_relative is not None:
        weights = (rss_root / weights_relative).resolve()
    else:
        raise ValueError(f"--weights is required for {profile.model_name}")

    if selection is not None:
        expected = {
            "model_name": profile.model_name,
            "model_family": profile.family,
            "patch_size": patch_size,
        }
        for key, value in expected.items():
            if key not in selection:
                raise ValueError(f"{selection_path}: missing required field {key!r}")
            if selection[key] != value:
                raise ValueError(
                    f"{selection_path}: {key}={selection[key]!r} disagrees with "
                    f"the pose run ({value!r})"
                )
        if "correction_mode" not in selection:
            raise ValueError(
                f"{selection_path}: missing required field 'correction_mode'"
            )
        if selection["correction_mode"] != correction:
            raise ValueError(
                f"{selection_path}: correction_mode={selection['correction_mode']!r} "
                f"disagrees with the pose run ({correction!r})"
            )
        if "correction" in selection and selection["correction"] != correction:
            raise ValueError(
                f"{selection_path}: correction and correction_mode disagree"
            )
        if not selection.get("selection_metric"):
            raise ValueError(f"{selection_path}: missing selection_metric")
        if selection.get("rank") != 0:
            raise ValueError(
                f"{selection_path}: DINOv2 layer selection must record rank=0"
            )
        sweep_layers = selection.get("layers")
        if sweep_layers != list(range(16, 25)):
            raise ValueError(
                f"{selection_path}: expected the preregistered DINOv2 sweep "
                "layers [16, ..., 24]"
            )
        if selected_layer not in sweep_layers:
            raise ValueError(
                f"{selection_path}: selected layer {selected_layer} is not in "
                "the declared sweep"
            )

        expected_weights_id = checkpoint_identity(weights)
        input_manifests = selection.get("input_manifests")
        if not isinstance(input_manifests, dict) or not input_manifests:
            raise ValueError(
                f"{selection_path}: input_manifests must bind the selection inputs"
            )
        observed_weights = {
            record.get("weights_id")
            for record in input_manifests.values()
            if isinstance(record, dict)
        }
        if observed_weights != {expected_weights_id}:
            raise ValueError(
                f"{selection_path}: input manifest weights {sorted(map(str, observed_weights))} "
                f"do not match pose checkpoint {expected_weights_id}"
            )

        local_source_provenance = source_checkout_provenance(source)
        local_revision = local_source_provenance["source_revision"]
        descriptor_provenance = selection.get("descriptor_provenance")
        if not isinstance(descriptor_provenance, dict):
            raise ValueError(
                f"{selection_path}: missing descriptor_provenance"
            )
        selected_revision = descriptor_provenance.get("source_revision")
        if selected_revision != local_revision:
            raise ValueError(
                f"{selection_path}: descriptor source revision "
                f"{selected_revision!r} does not match local checkout "
                f"{local_revision!r}"
            )
        selected_dirty = descriptor_provenance.get("source_dirty")
        local_dirty = local_source_provenance["source_dirty"]
        if selected_dirty != local_dirty:
            raise ValueError(
                f"{selection_path}: descriptor source dirty state "
                f"{selected_dirty!r} does not match local checkout "
                f"{local_dirty!r}"
            )
        per_manifest_revisions = {
            record.get("descriptor_provenance", {}).get("source_revision")
            for record in input_manifests.values()
            if isinstance(record, dict)
            and isinstance(record.get("descriptor_provenance"), dict)
        }
        if per_manifest_revisions and per_manifest_revisions != {local_revision}:
            raise ValueError(
                f"{selection_path}: input manifests do not share the local source "
                f"revision {local_revision}"
            )
        per_manifest_dirty = {
            record.get("descriptor_provenance", {}).get("source_dirty")
            for record in input_manifests.values()
            if isinstance(record, dict)
            and isinstance(record.get("descriptor_provenance"), dict)
        }
        if per_manifest_dirty and per_manifest_dirty != {local_dirty}:
            raise ValueError(
                f"{selection_path}: input manifests do not share the local source "
                f"dirty state {local_dirty!r}"
            )
    method_prefix = _relative_method_prefix(
        args.method_prefix
        or (
            "combined_interpolate_v3"
            if profile.model_name == DEFAULT_MODEL_NAME
            else f"combined_interpolate_{profile.model_name}"
        )
    )
    basis_template = (
        args.basis_filename_template
        or f"{profile.model_name}_{{height}}x{{width}}_basis.pt"
    )
    return PoseBackbone(
        model_name=profile.model_name,
        model_family=profile.family,
        source=source,
        weights=weights,
        patch_size=patch_size,
        layer=layer,
        extraction_layers=extraction_layers,
        correction=correction,
        ranks=ranks,
        method_prefix=method_prefix,
        basis_filename_template=basis_template,
        layer_selection=selection_path,
        layer_selection_payload=selection,
    )


def _dino_cache_root(cache_root: Path, backbone: PoseBackbone, group: str) -> Path:
    # Preserve the established DINOv3 cache path and namespace DINOv2 so
    # incompatible descriptors can never be mixed on a resume.
    base = cache_root / "dino"
    if backbone.model_name != DEFAULT_MODEL_NAME:
        base /= backbone.model_name
    return base / group


def _load_subsets(manifest_path: Path) -> list[Subset]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    subsets = [
        Subset(
            dataset_label=str(row["dataset_label"]),
            root_name=str(row["root_name"]),
            subset_name=str(row["subset_name"]),
            image_root=Path(row["image_root"]).resolve(),
            pair_file=Path(row["pair_file"]).resolve(),
            downscale_only=bool(row["downscale_only"]),
        )
        for row in payload["subsets"]
    ]
    if not subsets:
        raise ValueError(f"No subsets in {manifest_path}")
    return subsets


def _selected(
    subsets: Sequence[Subset],
    groups: Sequence[str],
    dataset_labels: Sequence[str],
) -> list[Subset]:
    requested = set(groups)
    requested_labels = set(dataset_labels)
    selected = [
        subset
        for subset in subsets
        if subset.group in requested
        and (not requested_labels or subset.dataset_label in requested_labels)
    ]
    found_labels = {subset.dataset_label for subset in selected}
    missing_labels = requested_labels - found_labels
    if missing_labels:
        raise ValueError(
            f"Input manifest is missing requested datasets: {sorted(missing_labels)}"
        )
    identities = [(subset.root_name, subset.subset_name) for subset in selected]
    if len(identities) != len(set(identities)):
        raise ValueError("Input manifest contains duplicate result-tree subsets")
    return selected


def _unique_images(pair_files: Sequence[Path]) -> list[Path]:
    from dino_m2m.pairs import read_pairs

    images: set[Path] = set()
    for pair_file in pair_files:
        for pair in read_pairs(pair_file):
            images.update((pair.left_rel, pair.right_rel))
    return sorted(images, key=str)


def _write_image_list(path: Path, images: Sequence[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{image.as_posix()}\n" for image in images), encoding="utf-8")


def _resolve_runtime_device(device_arg: str) -> str:
    """Resolve ``auto`` before adapters that pass the value to ``torch.device``."""
    from dino_m2m.matching import resolve_device

    return str(resolve_device(device_arg))


def _fit_bases(
    *,
    subsets: Sequence[Subset],
    basis_root: Path,
    backbone: PoseBackbone,
    device: str,
) -> None:
    from dino_m2m.dino import fit_bias_for_pair_files

    if backbone.correction == "none":
        print(
            f"{backbone.model_name} uses correction=none/rank 0; "
            "skipping positional-basis fitting.",
            flush=True,
        )
        return
    positive_ranks = tuple(rank for rank in backbone.ranks if rank > 0)
    if not positive_ranks:
        raise ValueError("positional-debias correction requires a positive rank")
    if backbone.model_family != "dinov3":
        raise ValueError(
            f"Basis fitting is unsupported for model family {backbone.model_family}"
        )
    for downscale_only, policy_name in ((False, "upscale"), (True, "downscale_only")):
        pair_files = tuple(
            subset.pair_file
            for subset in subsets
            if subset.downscale_only == downscale_only
        )
        if not pair_files:
            continue
        output = basis_root / policy_name
        outputs = fit_bias_for_pair_files(
            pair_files=pair_files,
            dinov3_source=backbone.source,
            weights=backbone.weights,
            output_root=output,
            filename_template=backbone.basis_filename_template,
            model_name=backbone.model_name,
            layers=backbone.extraction_layers,
            long_edge=LONG_EDGE,
            downscale_only=downscale_only,
            components=positive_ranks,
            device_arg=device,
            existing="skip",
            save_json=True,
        )
        print(
            f"[{policy_name}] wrote {len(outputs)} basis files under {output}",
            flush=True,
        )


def _extract_descriptor_group(
    *,
    group: str,
    subsets: Sequence[Subset],
    cache_root: Path,
    batch_size: int,
    backbone: PoseBackbone,
    device: str,
) -> None:
    from PIL import Image

    from dino_m2m.dino import ExtractionOptions
    from dino_m2m.resize import resized_hw_long_edge

    import dino_m2m.dino as extraction_module

    extract = extraction_module.extract_dino

    group_subsets = [subset for subset in subsets if subset.group == group]
    if not group_subsets:
        return
    image_roots = {subset.image_root for subset in group_subsets}
    policies = {subset.downscale_only for subset in group_subsets}
    if len(image_roots) != 1 or len(policies) != 1:
        raise ValueError(f"Group {group} does not share one image root/resize policy")
    images = _unique_images([subset.pair_file for subset in group_subsets])
    image_list = cache_root / "image_lists" / f"{group}.txt"
    _write_image_list(image_list, images)
    output_root = _dino_cache_root(cache_root, backbone, group)
    started = time.perf_counter()
    options = ExtractionOptions(
            input_root=next(iter(image_roots)),
            output_root=output_root,
            source=backbone.source,
            weights=backbone.weights,
            layers=backbone.extraction_layers,
            model_name=backbone.model_name,
            image_list=image_list,
            batch_size=batch_size,
            long_edge=LONG_EDGE,
            downscale_only=next(iter(policies)),
            device=device,
            overwrite=False,
            compile_model=False,
        )

    # ``extract_dino`` groups only within each consecutive batch.  Sorting by
    # padded tensor size avoids splitting heterogeneous NAVI batches into
    # several batch-size-one forwards while leaving every numerical operation
    # and artifact contract unchanged.
    original_select_images = extraction_module.select_images

    def select_images_grouped(extraction_options: ExtractionOptions) -> list[Path]:
        selected_images = original_select_images(extraction_options)

        def tensor_size(path: Path) -> tuple[int, int, str]:
            with Image.open(path) as image:
                width, height = image.size
            resized_height, resized_width, _ = resized_hw_long_edge(
                height,
                width,
                extraction_options.long_edge,
                downscale_only=extraction_options.downscale_only,
            )
            patch_size = backbone.patch_size
            padded_height = (
                resized_height + patch_size - 1
            ) // patch_size * patch_size
            padded_width = (
                resized_width + patch_size - 1
            ) // patch_size * patch_size
            return padded_height, padded_width, str(path)

        return sorted(selected_images, key=tensor_size)

    extraction_module.select_images = select_images_grouped
    try:
        written = extract(options)
    finally:
        extraction_module.select_images = original_select_images
    elapsed = time.perf_counter() - started
    print(
        f"[{group}] descriptors selected={len(images)} written={written} "
        f"model={backbone.model_name} layers={list(backbone.extraction_layers)} "
        f"elapsed={elapsed:.1f}s cache={output_root}",
        flush=True,
    )


def _extract_superpoint_group(
    *,
    group: str,
    subsets: Sequence[Subset],
    cache_root: Path,
    device: str,
    max_num_keypoints: int,
) -> None:
    from dino_m2m.superpoint import (
        CacheBackedSuperPoint,
        ExternalLightGlueSuperPoint,
        SuperPointConfig,
    )

    group_subsets = [subset for subset in subsets if subset.group == group]
    if not group_subsets:
        return
    image_roots = {subset.image_root for subset in group_subsets}
    policies = {subset.downscale_only for subset in group_subsets}
    if len(image_roots) != 1 or len(policies) != 1:
        raise ValueError(f"Group {group} does not share one image root/resize policy")
    images = _unique_images([subset.pair_file for subset in group_subsets])
    config = SuperPointConfig(
        max_num_keypoints=max_num_keypoints,
        long_edge=LONG_EDGE,
        downscale_only=next(iter(policies)),
        expected_weights_id=SUPERPOINT_WEIGHTS_ID,
    )
    adapter = ExternalLightGlueSuperPoint(device, config)
    cache = CacheBackedSuperPoint(
        image_root=next(iter(image_roots)),
        cache_root=cache_root / "superpoint" / group,
        config=config,
        adapter=adapter,
        overwrite=False,
    )
    started = time.perf_counter()
    for index, image in enumerate(images, start=1):
        cache.load_or_extract(image, include_descriptors=False)
        if index == 1 or index % 100 == 0 or index == len(images):
            elapsed = time.perf_counter() - started
            print(
                f"[{group}] SuperPoint {index}/{len(images)} elapsed={elapsed:.1f}s",
                flush=True,
            )


def _match_subset(
    *,
    subset: Subset,
    results_root: Path,
    basis_root: Path,
    cache_root: Path,
    keypoint_cache_root: Path,
    max_ks: Sequence[int],
    association_upperbound: int,
    superpoint_keypoints: int,
    backbone: PoseBackbone,
    device: str,
) -> None:
    from dino_m2m.dino import source_checkout_provenance
    from dino_m2m.pipeline import MatchOptions, run_matching
    from dino_m2m.superpoint import SuperPointConfig

    output_root = (
        results_root
        / subset.root_name
        / subset.subset_name
        / backbone.method_prefix
        / f"layer{backbone.layer}"
    )
    failure_manifest = output_root / "failures.json"
    if failure_manifest.is_file():
        retained_failures = json.loads(failure_manifest.read_text(encoding="utf-8"))
        if not isinstance(retained_failures, list):
            raise ValueError(f"{failure_manifest}: expected a JSON list")
        if retained_failures:
            print(
                f"[{subset.root_name}/{subset.subset_name}] resuming "
                f"{len(retained_failures)} previously failed pair(s); completed "
                "association CSVs remain immutable.",
                flush=True,
            )
    policy_name = "downscale_only" if subset.downscale_only else "upscale"
    group_dino_root = _dino_cache_root(cache_root, backbone, subset.group)
    layer_dino_root = group_dino_root / f"layer{backbone.layer}"
    dino_root = layer_dino_root if layer_dino_root.is_dir() else group_dino_root
    summary = run_matching(
        MatchOptions(
            pair_file=subset.pair_file,
            image_root=subset.image_root,
            dino_root=dino_root,
            keypoint_cache_root=keypoint_cache_root / "superpoint" / subset.group,
            output_root=output_root,
            weights=backbone.weights,
            source=backbone.source,
            model_name=backbone.model_name,
            layer=backbone.layer,
            patch_size=backbone.patch_size,
            correction=backbone.correction,
            basis_root=(
                basis_root / policy_name
                if backbone.correction == "positional-debias"
                else None
            ),
            basis_filename_template=backbone.basis_filename_template,
            svd_components=backbone.ranks,
            max_ks=tuple(max_ks),
            association_upperbound=association_upperbound,
            device=device,
            existing="skip",
            compute_missing_keypoints=False,
            superpoint=SuperPointConfig(
                max_num_keypoints=superpoint_keypoints,
                long_edge=LONG_EDGE,
                downscale_only=subset.downscale_only,
                expected_weights_id=SUPERPOINT_WEIGHTS_ID,
            ),
        )
    )
    print(
        f"[{subset.root_name}/{subset.subset_name}] pairs={summary.pair_count} "
        f"outputs={summary.output_count} failures={summary.failure_count}",
        flush=True,
    )
    if summary.failure_count:
        raise RuntimeError(f"Association failures recorded in {summary.failure_manifest}")
    keypoint_derivation_path = (
        keypoint_cache_root
        / "superpoint"
        / subset.group
        / "derivation_manifest.json"
    )
    keypoint_derivation = (
        {
            "path": str(keypoint_derivation_path),
            "sha256": _sha256(keypoint_derivation_path),
        }
        if keypoint_derivation_path.is_file()
        else None
    )
    selection_record: dict[str, Any] | None = None
    if backbone.layer_selection is not None:
        selection_record = {
            "path": str(backbone.layer_selection),
            "sha256": _sha256(backbone.layer_selection),
            "selection_metric": (
                backbone.layer_selection_payload or {}
            ).get("selection_metric"),
        }
    _write_json_atomic(
        output_root / "pose_matching_manifest.json",
        {
            "schema_version": 1,
            "experiment": "main_paper_pose_associations",
            "dataset_label": subset.dataset_label,
            "root_name": subset.root_name,
            "subset_name": subset.subset_name,
            "pair_file": str(subset.pair_file),
            "model_name": backbone.model_name,
            "model_family": backbone.model_family,
            "source": str(backbone.source),
            **source_checkout_provenance(backbone.source),
            "weights": str(backbone.weights),
            "weights_sha256": _sha256(backbone.weights),
            "patch_size": backbone.patch_size,
            "layer": backbone.layer,
            "correction": backbone.correction,
            "svd_components": list(backbone.ranks),
            "superpoint_keypoints": superpoint_keypoints,
            **(
                {"superpoint_cache_derivation": keypoint_derivation}
                if keypoint_derivation is not None
                else {}
            ),
            "dino_sampling": "bilinear",
            "max_ks": list(max_ks),
            "association_upperbound": association_upperbound,
            "layer_selection": selection_record,
        },
    )


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    default_results = repository_root / "artifacts" / "matching_estimation_results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "fit-basis",
            "extract-dino",
            "extract-features",
            "extract-superpoint",
            "match",
            "all",
        ),
    )
    parser.add_argument(
        "--rss-root",
        type=Path,
        default=repository_root,
    )
    parser.add_argument("--results-root", type=Path, default=default_results)
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help=(
            "One-based DINO layer number. If omitted, read --layer-selection; "
            f"otherwise default to {DEFAULT_LAYER}."
        ),
    )
    parser.add_argument(
        "--layer-selection",
        type=Path,
        default=None,
        help=(
            "Selection JSON with top-level best_layer (selected_layer is a "
            "backward-compatible alias). Model/profile fields are validated."
        ),
    )
    parser.add_argument(
        "--extract-layers",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Layers jointly fitted/extracted by fit-basis and extract-dino. "
            "Defaults to --layer; matching still uses the single --layer value."
        ),
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=default_results / "experiment_inputs.json",
    )
    parser.add_argument(
        "--basis-root",
        type=Path,
        default=repository_root / "artifacts" / "debiasing_basis" / "pose_estimation",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=repository_root / "artifacts" / "feature_cache" / "pose_estimation",
    )
    parser.add_argument(
        "--keypoint-cache-root",
        type=Path,
        default=None,
        help=(
            "Optional root containing the SuperPoint cache. Defaults to "
            "--cache-root; setting it separately permits an isolated DINO "
            "cache to reuse immutable keypoints."
        ),
    )
    parser.add_argument(
        "--debias-ranks",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Correction ranks. Defaults to 200 for DINOv3 and rank 0 for "
            "backbones whose profile disables positional-bias correction."
        ),
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="DINO source checkout; defaults to third_party/<model-family>.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Backbone checkpoint; known paper backbones have repository defaults.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Optional explicit patch size; must match the registered model profile.",
    )
    parser.add_argument(
        "--correction",
        choices=("auto", "none", "positional-debias"),
        default="auto",
    )
    parser.add_argument(
        "--method-prefix",
        default=None,
        help="Result-tree method directory; defaults to a model-specific name.",
    )
    parser.add_argument(
        "--basis-filename-template",
        default=None,
        help="Basis filename template using {height} and {width}.",
    )
    parser.add_argument(
        "--max-ks",
        type=int,
        nargs="+",
        default=(1, 5),
        help="Progressive mutual-KNN endpoints generated during matching.",
    )
    parser.add_argument(
        "--association-upperbound",
        type=int,
        default=2048,
        help=(
            "Maximum number of association edges retained per pair; use 0 for "
            "an uncapped progressive graph."
        ),
    )
    parser.add_argument(
        "--superpoint-keypoints",
        type=int,
        default=MAX_KEYPOINTS,
        help=(
            "Maximum SuperPoint detections per image. This value is encoded in "
            "the cache identity and pose manifest."
        ),
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=GROUP_ORDER,
        default=list(GROUP_ORDER),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(CANONICAL_DATASETS),
        help=(
            "Dataset labels from experiment_inputs.json; defaults to the canonical "
            "six-dataset protocol and therefore excludes historical NAVI-Wild-v1."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--opencv-site-packages",
        type=Path,
        default=None,
        help=(
            "Optional existing site-packages directory appended to sys.path "
            "when the active experiment environment lacks cv2."
        ),
    )
    parser.add_argument(
        "--lightglue-source",
        type=Path,
        default=None,
        help="Optional source checkout whose root contains the lightglue package.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.opencv_site_packages is not None:
        sys.path.append(str(args.opencv_site_packages.resolve()))
    if args.lightglue_source is not None:
        sys.path.insert(0, str(args.lightglue_source.resolve()))
    rss_root = args.rss_root.resolve()
    sys.path.insert(0, str(rss_root / "src"))
    runtime_device = _resolve_runtime_device(args.device)
    backbone = _resolve_backbone(args, rss_root)
    subsets = _selected(
        _load_subsets(args.input_manifest.resolve()),
        args.groups,
        args.datasets,
    )
    if not subsets:
        raise ValueError("No subsets selected")
    results_root = args.results_root.resolve()
    basis_root = args.basis_root.resolve()
    cache_root = args.cache_root.resolve()
    keypoint_cache_root = (
        args.keypoint_cache_root.resolve()
        if args.keypoint_cache_root is not None
        else cache_root
    )
    max_ks = tuple(sorted(set(args.max_ks)))
    if not max_ks or min(max_ks) <= 0:
        raise ValueError("--max-ks must contain positive integers")
    if args.association_upperbound < 0:
        raise ValueError("--association-upperbound must be non-negative")
    if args.superpoint_keypoints <= 0:
        raise ValueError("--superpoint-keypoints must be positive")

    if args.stage in (
        "fit-basis",
        "extract-dino",
        "extract-features",
        "match",
        "all",
    ):
        if not backbone.source.is_dir():
            raise NotADirectoryError(backbone.source)
        if not backbone.weights.is_file():
            raise FileNotFoundError(backbone.weights)

    if args.stage in ("fit-basis", "all"):
        _fit_bases(
            subsets=subsets,
            basis_root=basis_root,
            backbone=backbone,
            device=runtime_device,
        )
    if args.stage in ("extract-dino", "extract-features", "all"):
        for group in args.groups:
            _extract_descriptor_group(
                group=group,
                subsets=subsets,
                cache_root=cache_root,
                batch_size=args.batch_size,
                backbone=backbone,
                device=runtime_device,
            )
    if args.stage in ("extract-superpoint", "all"):
        for group in args.groups:
            _extract_superpoint_group(
                group=group,
                subsets=subsets,
                cache_root=keypoint_cache_root,
                device=runtime_device,
                max_num_keypoints=args.superpoint_keypoints,
            )
    if args.stage in ("match", "all"):
        for subset in subsets:
            _match_subset(
                subset=subset,
                results_root=results_root,
                basis_root=basis_root,
                cache_root=cache_root,
                keypoint_cache_root=keypoint_cache_root,
                max_ks=max_ks,
                association_upperbound=args.association_upperbound,
                superpoint_keypoints=args.superpoint_keypoints,
                backbone=backbone,
                device=runtime_device,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
