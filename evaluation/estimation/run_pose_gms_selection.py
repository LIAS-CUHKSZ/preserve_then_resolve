#!/usr/bin/env python3
"""Filter DINO associations, run LO-RANSAC, and summarize pose AUC.

The script is intentionally stage based.  GMS variants are deterministic and
share the same raw Progressive-MKNN files.  Estimator tasks use independent
config/output files, so subsets and parameter candidates can run concurrently
and resume without mixing results.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from evaluation.estimation.metrics import (
    expected_pose_pair_indices,
    validated_pose_result_pair_ids,
)


DEFAULT_LAYER = 19
DEFAULT_DEBIAS_RANK = 200
DEFAULT_WITH_SCALE = True
DEFAULT_WITH_ROTATION = True
DEFAULT_METHOD_PREFIX = "combined_interpolate_v3"
DEFAULT_MODEL_NAME = "dinov3_vitl16"
DINOV2_MODEL_NAME = "dinov2_vitl14_reg"
PAPER_WEIGHTS_IDS = {
    DEFAULT_MODEL_NAME: "sha256:8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035",
    DINOV2_MODEL_NAME: "sha256:36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51",
}
GMS_CONFIG_FILENAME = "gms_filter_manifest.json"
RAW_M2M_VARIANT = "progressive_k5"
MNN_VARIANT = "progressive_k1"
VALID_MODES = ("CM", "MCM", "HCM", "HCM_MC")
CANONICAL_DATASETS = (
    "ScanNet",
    "MegaDepth",
    "NAVI-Multi",
    "NAVI-Wild",
    "METU-CC",
    "METU-CS",
)


@dataclass(frozen=True)
class Subset:
    dataset_label: str
    root_name: str
    subset_name: str
    pair_count: int


@dataclass(frozen=True)
class GmsVariant:
    threshold: float
    grid_size: int

    @property
    def directory_name(self) -> str:
        threshold = format(self.threshold, ".6f").rstrip("0")
        if threshold.endswith("."):
            threshold += "0"
        return (
            f"{RAW_M2M_VARIANT}_GMSm2m_ThrFact_{threshold}_"
            f"Gridsz_{self.grid_size}_auto_mask"
        )

    @property
    def label(self) -> str:
        return f"gms_t{self.threshold:g}_g{self.grid_size}"


@dataclass(frozen=True)
class EstimatorTask:
    subset: Subset
    variant_label: str
    variant_dir: str
    mode: str
    max_iterations: int
    seed: int


@dataclass(frozen=True)
class PipelineSpec:
    """One exact association/estimator combination from the paper."""

    variant_label: str
    variant_dir: str
    mode: str

    @property
    def label(self) -> str:
        return f"{self.variant_label}/{self.mode}"


def _load_subsets(manifest: Path, dataset_labels: Sequence[str]) -> list[Subset]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    requested = set(dataset_labels)
    subsets = [
        Subset(
            dataset_label=str(row["dataset_label"]),
            root_name=str(row["root_name"]),
            subset_name=str(row["subset_name"]),
            pair_count=int(row["pair_count"]),
        )
        for row in payload["subsets"]
        if not requested or str(row["dataset_label"]) in requested
    ]
    if not subsets:
        raise ValueError("No subsets match --datasets")
    found = {subset.dataset_label for subset in subsets}
    missing = requested - found
    if missing:
        raise ValueError(f"Input manifest is missing datasets: {sorted(missing)}")
    identities = [(subset.root_name, subset.subset_name) for subset in subsets]
    if len(identities) != len(set(identities)):
        raise ValueError("Input manifest contains duplicate result-tree subsets")
    return subsets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_layer_selection(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: layer selection must be a JSON object")
    candidates = [
        int(payload[key])
        for key in ("best_layer", "selected_layer")
        if key in payload
    ]
    if not candidates or len(set(candidates)) != 1:
        raise ValueError(
            f"{path}: expected one consistent best_layer/selected_layer value"
        )
    expected = {
        "model_name": "dinov2_vitl14_reg",
        "model_family": "dinov2",
        "patch_size": 14,
        "correction_mode": "none",
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{path}: incompatible DINOv2 layer selection: {mismatches}")
    if not payload.get("selection_metric"):
        raise ValueError(f"{path}: missing selection_metric")
    if payload.get("rank") != 0:
        raise ValueError(f"{path}: DINOv2 layer selection must record rank=0")
    if payload.get("layers") != list(range(16, 25)):
        raise ValueError(
            f"{path}: expected the DINOv2 sweep layers [16, ..., 24]"
        )
    if candidates[0] not in payload["layers"]:
        raise ValueError(f"{path}: selected layer is not in the declared sweep")
    payload = dict(payload)
    payload["best_layer"] = candidates[0]
    return payload


def _resolve_layer_contract(args: argparse.Namespace) -> dict[str, Any] | None:
    selection_path = (
        args.layer_selection.expanduser().resolve()
        if args.layer_selection is not None
        else None
    )
    selection = _load_layer_selection(selection_path) if selection_path else None
    if selection is not None and args.model_name != DINOV2_MODEL_NAME:
        raise ValueError(
            "--layer-selection is a DINOv2 contract; pass "
            f"--model-name {DINOV2_MODEL_NAME}"
        )
    selected_layer = int(selection["best_layer"]) if selection else None
    if args.layer is not None and selected_layer is not None and args.layer != selected_layer:
        raise ValueError(
            f"--layer={args.layer} disagrees with {selection_path}: {selected_layer}"
        )
    args.layer = (
        args.layer
        if args.layer is not None
        else (selected_layer if selected_layer is not None else DEFAULT_LAYER)
    )
    selected_rank = 0 if args.model_name == DINOV2_MODEL_NAME else None
    if (
        args.debias_rank is not None
        and selected_rank is not None
        and args.debias_rank != selected_rank
    ):
        raise ValueError(
            f"--debias-rank={args.debias_rank} disagrees with the no-correction "
            f"DINOv2 selection {selection_path}"
        )
    args.debias_rank = (
        args.debias_rank
        if args.debias_rank is not None
        else (selected_rank if selected_rank is not None else DEFAULT_DEBIAS_RANK)
    )
    args.method_prefix = args.method_prefix or (
        "combined_interpolate_dinov2_vitl14_reg"
        if args.model_name == DINOV2_MODEL_NAME
        else DEFAULT_METHOD_PREFIX
    )
    args.layer_selection = selection_path
    return selection


def _gms_variants(thresholds: Sequence[float], grids: Sequence[int]) -> list[GmsVariant]:
    return [
        GmsVariant(threshold, grid)
        for threshold in thresholds
        for grid in grids
    ]


def _pipeline_gms_variant(pipeline: PipelineSpec) -> GmsVariant | None:
    fields = pipeline.variant_label.split("_")
    if len(fields) != 3 or fields[0] != "gms":
        return None
    try:
        return GmsVariant(
            float(fields[1].removeprefix("t")),
            int(fields[2].removeprefix("g")),
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid internal GMS pipeline label: {pipeline.variant_label}"
        ) from exc


def _parse_gms_candidate(value: str) -> GmsVariant:
    fields = value.split(":")
    if len(fields) != 2:
        raise argparse.ArgumentTypeError(
            "GMS candidate must be THRESHOLD:GRID"
        )
    try:
        return GmsVariant(float(fields[0]), int(fields[1]))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "GMS candidate must be THRESHOLD:GRID"
        ) from error


def _method_root(
    results_root: Path,
    subset: Subset,
    layer: int,
    debias_rank: int,
    method_prefix: str,
) -> Path:
    return (
        results_root
        / subset.root_name
        / subset.subset_name
        / method_prefix
        / f"layer{layer}"
        / f"debias_svd{debias_rank}"
    )


def _validate_selection_binding(
    *,
    results_root: Path,
    subset: Subset,
    method_prefix: str,
    layer: int,
    debias_rank: int,
    selection_path: Path | None,
) -> None:
    if selection_path is None:
        return
    layer_root = (
        results_root
        / subset.root_name
        / subset.subset_name
        / method_prefix
        / f"layer{layer}"
    )
    pose_manifest_path = layer_root / "pose_matching_manifest.json"
    pose_manifest = json.loads(pose_manifest_path.read_text(encoding="utf-8"))
    expected_pose = {
        "dataset_label": subset.dataset_label,
        "root_name": subset.root_name,
        "subset_name": subset.subset_name,
        "model_name": "dinov2_vitl14_reg",
        "model_family": "dinov2",
        "patch_size": 14,
        "layer": layer,
        "correction": "none",
        "dino_sampling": "bilinear",
        "superpoint_keypoints": 2048,
    }
    pose_mismatches = {
        key: (pose_manifest.get(key), value)
        for key, value in expected_pose.items()
        if pose_manifest.get(key) != value
    }
    if pose_mismatches:
        raise ValueError(
            f"{pose_manifest_path}: DINOv2 pose contract mismatch: {pose_mismatches}"
        )
    if pose_manifest.get("svd_components") != [0] or debias_rank != 0:
        raise ValueError(
            f"{pose_manifest_path}: DINOv2 pose run must use correction=none/rank 0"
        )
    selection_record = pose_manifest.get("layer_selection")
    expected_selection_hash = _sha256(selection_path)
    if (
        not isinstance(selection_record, dict)
        or selection_record.get("sha256") != expected_selection_hash
    ):
        raise ValueError(
            f"{pose_manifest_path}: pose associations are not bound to "
            f"{selection_path}"
        )

    raw_manifest_path = (
        layer_root / f"debias_svd{debias_rank}" / RAW_M2M_VARIANT
        / "association_manifest.json"
    )
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    expected_raw = {
        "pair_count": subset.pair_count,
        "model_name": "dinov2_vitl14_reg",
        "model_family": "dinov2",
        "patch_size": 14,
        "descriptor_dim": 1024,
        "register_tokens": 4,
        "correction": "none",
        "layer": layer,
        "debias_rank": 0,
        "progressive_max_k": 5,
    }
    raw_mismatches = {
        key: (raw_manifest.get(key), value)
        for key, value in expected_raw.items()
        if raw_manifest.get(key) != value
    }
    if raw_mismatches:
        raise ValueError(
            f"{raw_manifest_path}: DINOv2 association mismatch: {raw_mismatches}"
        )


def _validate_selected_subsets(
    args: argparse.Namespace, subsets: Sequence[Subset]
) -> None:
    for subset in subsets:
        _validate_selection_binding(
            results_root=args.results_root.resolve(),
            subset=subset,
            method_prefix=args.method_prefix,
            layer=args.layer,
            debias_rank=args.debias_rank,
            selection_path=args.layer_selection,
        )
        method_root = _method_root(
            args.results_root.resolve(),
            subset,
            args.layer,
            args.debias_rank,
            args.method_prefix,
        )
        if args.command == "filter":
            variants = ((RAW_M2M_VARIANT, 5, None),)
        else:
            variants = tuple(
                (
                    pipeline.variant_dir,
                    1 if pipeline.variant_label == "mnn" else 5,
                    _pipeline_gms_variant(pipeline),
                )
                for pipeline in _pipeline_specs(args.pipelines)
            )
        for variant_dir, progressive_max_k, gms_variant in variants:
            variant_root = method_root / variant_dir
            manifest_path = variant_root / "association_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = {
                "pair_count": subset.pair_count,
                "model_name": args.model_name,
                "layer": args.layer,
                "debias_rank": args.debias_rank,
                "progressive_max_k": progressive_max_k,
                "association_upperbound": 2048,
                "weights_id": PAPER_WEIGHTS_IDS[args.model_name],
            }
            mismatches = {
                key: (manifest.get(key), value)
                for key, value in expected.items()
                if manifest.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    f"{manifest_path}: main-paper association mismatch: {mismatches}"
                )
            if gms_variant is not None:
                raw_manifest_path = (
                    method_root / RAW_M2M_VARIANT / "association_manifest.json"
                )
                raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
                if manifest != raw_manifest:
                    raise ValueError(
                        f"{manifest_path}: GMS association provenance differs from "
                        f"{raw_manifest_path}"
                    )
                filter_manifest_path = variant_root / GMS_CONFIG_FILENAME
                filter_manifest = json.loads(
                    filter_manifest_path.read_text(encoding="utf-8")
                )
                expected_filter_manifest = {
                    "schema_version": 1,
                    "filter": "gms_m2m",
                    "threshold_factor": gms_variant.threshold,
                    "grid_size": gms_variant.grid_size,
                    "auto_mask": True,
                    "with_scale": True,
                    "with_rotation": True,
                    "source_directory": str(method_root / RAW_M2M_VARIANT),
                }
                if filter_manifest != expected_filter_manifest:
                    raise ValueError(
                        f"{filter_manifest_path}: main-paper GMS settings mismatch"
                    )


def _require_safe_protocol(args: argparse.Namespace) -> None:
    if args.model_name != DINOV2_MODEL_NAME:
        return
    if getattr(args, "command", None) == "filter":
        if not args.with_scale or not args.with_rotation:
            raise ValueError(
                "The Q3 DINOv2 run requires GMS scale and rotation search"
            )
        requested_variants = (
            args.candidates
            if args.candidates
            else _gms_variants(args.thresholds or (), args.grids or ())
        )
        if requested_variants != [GmsVariant(4.0, 20)]:
            raise ValueError(
                "The Q3 DINOv2 run requires exactly GMS threshold=4/grid=20"
            )
    if getattr(args, "command", None) in {"estimate", "summarize"}:
        requested = {spec.label for spec in _pipeline_specs(args.pipelines)}
        required = {"mnn/CM", "gms_t4_g20/HCM_MC"}
        if requested != required:
            raise ValueError(
                "The DINOv2 Q3 comparison requires exactly --pipelines "
                "mnn:CM gms:4:20:HCM_MC"
            )
        if args.max_iterations != 100000:
            raise ValueError(
                "The Q3 DINOv2 run requires --max-iterations 100000"
            )
        if tuple(args.seeds) != (0, 1, 2, 3, 4):
            raise ValueError("The DINOv2 Q3 comparison requires seeds 0 1 2 3 4")


def _run_logged(command: Sequence[str], log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        list(command),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    return completed.returncode, completed.stdout[-2000:]


def _expected_matching_names(pair_count: int) -> set[str]:
    width = max(3, len(str(pair_count)))
    return {
        f"matching_{index:0{width}d}.csv"
        for index in range(1, pair_count + 1)
    }


def _require_exact_matching_files(directory: Path, pair_count: int) -> int:
    expected = _expected_matching_names(pair_count)
    actual = {
        path.name for path in directory.glob("matching_*.csv") if path.is_file()
    }
    if actual != expected:
        raise ValueError(
            f"{directory}: matching-file identity mismatch; "
            f"missing={sorted(expected - actual)[:5]}, "
            f"extras={sorted(actual - expected)[:5]}"
        )
    return len(actual)


def _filter_one(
    *,
    gms_binary: Path,
    results_root: Path,
    work_root: Path,
    subset: Subset,
    variant: GmsVariant,
    layer: int,
    debias_rank: int,
    method_prefix: str,
    with_scale: bool,
    with_rotation: bool,
    overwrite_existing: bool,
) -> str:
    input_dir = (
        _method_root(results_root, subset, layer, debias_rank, method_prefix)
        / RAW_M2M_VARIANT
    )
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    _require_exact_matching_files(input_dir, subset.pair_count)
    raw_association_manifest = input_dir / "association_manifest.json"
    raw_provenance = json.loads(
        raw_association_manifest.read_text(encoding="utf-8")
    )
    log_path = (
        work_root
        / "gms_logs"
        / method_prefix
        / subset.root_name
        / subset.subset_name
        / f"{variant.label}.log"
    )
    command = [
        str(gms_binary),
        "--root",
        str(input_dir),
        "--threshold-factor",
        f"{variant.threshold:g}",
        "--grid-size",
        str(variant.grid_size),
        "--auto-mask",
        "1",
    ]
    if with_scale:
        command.append("--with-scale")
    if with_rotation:
        command.append("--with-rotation")
    output_dir = (
        _method_root(results_root, subset, layer, debias_rank, method_prefix)
        / variant.directory_name
    )
    filter_manifest = {
        "schema_version": 1,
        "filter": "gms_m2m",
        "threshold_factor": variant.threshold,
        "grid_size": variant.grid_size,
        "auto_mask": True,
        "with_scale": with_scale,
        "with_rotation": with_rotation,
        "source_directory": str(input_dir),
    }
    existing_artifacts = (
        output_dir.is_dir()
        and any(path.is_file() for path in output_dir.iterdir())
    )
    if existing_artifacts and not overwrite_existing:
        count = _require_exact_matching_files(output_dir, subset.pair_count)
        copied_provenance = json.loads(
            (output_dir / "association_manifest.json").read_text(encoding="utf-8")
        )
        if copied_provenance != raw_provenance:
            raise ValueError(
                f"{output_dir}: retained GMS association provenance differs from "
                f"{raw_association_manifest}; refusing to overwrite"
            )
        existing_filter = json.loads(
            (output_dir / GMS_CONFIG_FILENAME).read_text(encoding="utf-8")
        )
        if existing_filter != filter_manifest:
            raise ValueError(
                f"{output_dir}: retained GMS settings differ from this request; "
                "refusing to overwrite (use --overwrite-existing explicitly)"
            )
        if not log_path.is_file():
            raise FileNotFoundError(
                f"{log_path}: retained GMS output has no execution log; use "
                "--overwrite-existing explicitly to regenerate it"
            )
        return (
            f"{subset.root_name}/{subset.subset_name} {variant.label}: "
            f"validated {count} retained files"
        )
    return_code, tail = _run_logged(command, log_path)
    if return_code:
        raise RuntimeError(
            f"GMS failed for {subset.root_name}/{subset.subset_name} "
            f"{variant.label}:\n{tail}"
        )
    expected_summary = (
        f"with_scale={str(with_scale).lower()} "
        f"with_rotation={str(with_rotation).lower()}"
    )
    if expected_summary not in tail:
        raise RuntimeError(
            f"GMS mode verification failed for {subset.root_name}/"
            f"{subset.subset_name} {variant.label}: expected `{expected_summary}` "
            f"in the executable summary.\n{tail}"
        )
    count = _require_exact_matching_files(output_dir, subset.pair_count)
    copied_provenance = json.loads(
        (output_dir / "association_manifest.json").read_text(encoding="utf-8")
    )
    if copied_provenance != raw_provenance:
        raise ValueError(
            f"{output_dir}: GMS did not preserve the raw association provenance"
        )
    (output_dir / GMS_CONFIG_FILENAME).write_text(
        json.dumps(filter_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return f"{subset.root_name}/{subset.subset_name} {variant.label}: {count} files"


def run_filter(args: argparse.Namespace) -> None:
    if args.candidates:
        variants = args.candidates
    elif args.thresholds and args.grids:
        variants = _gms_variants(args.thresholds, args.grids)
    else:
        raise ValueError(
            "Specify --candidates or both --thresholds and --grids"
        )
    if len(variants) != len(set(variants)):
        raise ValueError("Duplicate GMS candidates are not allowed")
    subsets = _load_subsets(args.input_manifest.resolve(), args.datasets)
    _validate_selected_subsets(args, subsets)
    jobs = [
        (subset, variant)
        for subset in subsets
        for variant in variants
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _filter_one,
                gms_binary=args.gms_binary.resolve(),
                results_root=args.results_root.resolve(),
                work_root=args.work_root.resolve(),
                subset=subset,
                variant=variant,
                layer=args.layer,
                debias_rank=args.debias_rank,
                method_prefix=args.method_prefix,
                with_scale=args.with_scale,
                with_rotation=args.with_rotation,
                overwrite_existing=args.overwrite_existing,
            )
            for subset, variant in jobs
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            print(f"[{index}/{len(futures)}] {future.result()}", flush=True)


def _parse_variant(value: str) -> tuple[str, str]:
    if value == "mnn":
        return "mnn", MNN_VARIANT
    if value == "raw":
        return "raw", RAW_M2M_VARIANT
    fields = value.split(":")
    if len(fields) == 3 and fields[0].lower() == "gms":
        variant = GmsVariant(float(fields[1]), int(fields[2]))
        return variant.label, variant.directory_name
    raise argparse.ArgumentTypeError(
        "variant must be mnn, raw, or gms:THRESHOLD:GRID"
    )


def _parse_pipeline(value: str) -> PipelineSpec:
    """Parse ``mnn:CM``, ``raw:HCM``, or ``gms:T:G:MODE``."""
    fields = value.split(":")
    if len(fields) == 2 and fields[0].lower() in {"mnn", "raw"}:
        variant_label, variant_dir = _parse_variant(fields[0].lower())
        mode = fields[1].upper()
    elif len(fields) == 4 and fields[0].lower() == "gms":
        variant_label, variant_dir = _parse_variant(":".join(fields[:3]))
        mode = fields[3].upper()
    else:
        raise argparse.ArgumentTypeError(
            "pipeline must be mnn:CM, raw:MODE, or gms:THRESHOLD:GRID:MODE"
        )
    if mode not in VALID_MODES:
        raise argparse.ArgumentTypeError(
            f"unsupported estimator mode {mode!r}; choose from {VALID_MODES}"
        )
    if variant_label == "mnn" and mode != "CM":
        raise argparse.ArgumentTypeError("the MNN baseline is defined only with CM")
    return PipelineSpec(variant_label, variant_dir, mode)


def _pipeline_specs(values: Sequence[str | PipelineSpec]) -> list[PipelineSpec]:
    specs = [value if isinstance(value, PipelineSpec) else _parse_pipeline(value) for value in values]
    labels = [spec.label for spec in specs]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate pipeline requests: {labels}")
    return specs


def _result_filename(mode: str, max_iterations: int, seed: int = 0) -> str:
    base = f"pose_seed{seed}_iter{max_iterations}"
    if mode in ("HCM", "HCM_MC"):
        return f"{mode}_{base}_q_ub_0.30.csv"
    return f"{mode}_{base}.csv"


def _config_text(
    *,
    results_root: Path,
    task: EstimatorTask,
    layer: int,
    debias_rank: int,
    method_prefix: str,
) -> str:
    return "\n".join(
        (
            f"matching_result_root={results_root / task.subset.root_name}",
            f"datasets={task.subset.subset_name}",
            (
                f"method={method_prefix}/layer{layer}/"
                f"debias_svd{debias_rank}/{task.variant_dir}"
            ),
            f"ransac_mode={task.mode}",
            "top_n_candidates=100",
            "q_ub=0.3",
            "m2m_delta=0.01",
            "max_error_px=1.0",
            "similarity_threshold=0.0",
            "max_matching_num=1024",
            "min_iterations=1000",
            f"max_iterations={task.max_iterations}",
            "success_prob=0.9999",
            f"seed={task.seed}",
            "tangent_sampson=false",
            "init_with_gt=false",
            "skip_existing_pairs=true",
            "allow_unbound_pose=false",
            "ransac_times=1",
            f"output_csv=pose_seed{task.seed}_iter{task.max_iterations}.csv",
            "",
        )
    )


def _result_pair_ids(path: Path, pair_count: int) -> list[int]:
    return list(validated_pose_result_pair_ids(path, range(1, pair_count + 1)))


def _estimate_one(
    *,
    runner_binary: Path,
    results_root: Path,
    work_root: Path,
    task: EstimatorTask,
    layer: int,
    debias_rank: int,
    method_prefix: str,
) -> str:
    method_dir = (
        _method_root(results_root, task.subset, layer, debias_rank, method_prefix)
        / task.variant_dir
    )
    if not method_dir.is_dir():
        raise FileNotFoundError(method_dir)
    config_path = (
        work_root
        / "estimator_configs"
        / method_prefix
        / f"layer{layer}_rank{debias_rank}"
        / f"iter{task.max_iterations}"
        / f"seed{task.seed}"
        / task.variant_label
        / task.subset.root_name
        / task.subset.subset_name
        / f"{task.mode}.cfg"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    requested_config = _config_text(
        results_root=results_root,
        task=task,
        layer=layer,
        debias_rank=debias_rank,
        method_prefix=method_prefix,
    )
    result_path = method_dir / _result_filename(
        task.mode, task.max_iterations, task.seed
    )
    if config_path.is_file():
        retained_config = config_path.read_text(encoding="utf-8")
        if retained_config != requested_config:
            raise ValueError(
                f"{config_path}: retained estimator configuration differs from "
                "this request; refusing to rebind an existing result"
            )
    elif result_path.is_file():
        raise FileNotFoundError(
            f"{config_path}: existing pose result is missing its generating config"
        )
    else:
        config_path.write_text(requested_config, encoding="utf-8")
    log_path = config_path.with_suffix(".log")
    if result_path.is_file():
        pair_ids = _result_pair_ids(result_path, task.subset.pair_count)
        if len(pair_ids) == task.subset.pair_count and log_path.is_file():
            return (
                f"{task.subset.root_name}/{task.subset.subset_name} "
                f"{task.variant_label}/{task.mode}/seed{task.seed}: validated "
                f"{len(pair_ids)} retained rows"
            )
    return_code, tail = _run_logged((str(runner_binary), str(config_path)), log_path)
    if return_code:
        raise RuntimeError(
            f"Estimator failed for {task.subset.root_name}/{task.subset.subset_name} "
            f"{task.variant_label}/{task.mode}/seed{task.seed}:\n{tail}"
        )
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    row_count = len(_result_pair_ids(result_path, task.subset.pair_count))
    if row_count != task.subset.pair_count:
        raise RuntimeError(
            f"{result_path}: expected {task.subset.pair_count} rows, found {row_count}"
        )
    return (
        f"{task.subset.root_name}/{task.subset.subset_name} "
        f"{task.variant_label}/{task.mode}/seed{task.seed}: {row_count} rows"
    )


def run_estimate(args: argparse.Namespace) -> None:
    subsets = _load_subsets(args.input_manifest.resolve(), args.datasets)
    _validate_selected_subsets(args, subsets)
    pipelines = _pipeline_specs(args.pipelines)
    tasks: list[EstimatorTask] = []
    for subset in subsets:
        for pipeline in pipelines:
            for seed in args.seeds:
                tasks.append(
                    EstimatorTask(
                        subset=subset,
                        variant_label=pipeline.variant_label,
                        variant_dir=pipeline.variant_dir,
                        mode=pipeline.mode,
                        max_iterations=args.max_iterations,
                        seed=seed,
                    )
                )
    if not tasks:
        raise ValueError("No compatible variant/mode tasks were requested")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _estimate_one,
                runner_binary=args.runner_binary.resolve(),
                results_root=args.results_root.resolve(),
                work_root=args.work_root.resolve(),
                task=task,
                layer=args.layer,
                debias_rank=args.debias_rank,
                method_prefix=args.method_prefix,
            )
            for task in tasks
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            print(f"[{index}/{len(futures)}] {future.result()}", flush=True)


def _error_auc(errors: np.ndarray) -> dict[str, float]:
    sorted_errors = [0.0] + sorted(float(value) for value in errors.reshape(-1))
    recall = list(np.linspace(0.0, 1.0, len(sorted_errors)))
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    report: dict[str, float] = {}
    for threshold in (5.0, 10.0, 20.0):
        last_index = int(np.searchsorted(sorted_errors, threshold))
        y = recall[:last_index] + [recall[last_index - 1]]
        x = sorted_errors[:last_index] + [threshold]
        report[f"auc@{threshold:g}"] = float(trapezoid(y, x)) / threshold * 100.0
    return report


def _pose_errors(result_path: Path, pose_path: Path) -> np.ndarray:
    expected = pd.read_csv(pose_path, usecols=["pair_idx"])["pair_idx"].astype(int)
    if expected.duplicated().any():
        raise ValueError(f"Duplicate pair IDs in {pose_path}")
    frame = pd.read_csv(result_path)
    if frame["pair_idx"].duplicated().any():
        raise ValueError(f"Duplicate pair IDs in {result_path}")
    frame = frame.assign(pair_idx=frame["pair_idx"].astype(int))
    expected_ids = set(expected.tolist())
    actual_ids = set(frame["pair_idx"].tolist())
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extras = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"Pair-ID mismatch in {result_path}: "
            f"missing={missing[:10]}, extras={extras[:10]}"
        )
    aligned = frame.set_index("pair_idx").reindex(expected)
    rotation = pd.to_numeric(aligned["rotation_error_deg"], errors="coerce").to_numpy()
    translation = pd.to_numeric(aligned["translation_error_deg"], errors="coerce").to_numpy()
    rotation = np.where(np.isfinite(rotation), rotation, 180.0)
    translation = np.where(np.isfinite(translation), translation, 180.0)
    return np.maximum(rotation, translation)


def _publication_variant(pipeline: PipelineSpec) -> str:
    aliases = {
        ("gms_t4_g20", "CM"): "m2m_poselib_cm",
        ("gms_t4_g20", "HCM"): "hcm",
        ("gms_t4_g20", "MCM"): "mcm",
        ("gms_t4_g20", "HCM_MC"): "full_hcm_mc",
        ("mnn", "CM"): "mnn_cm",
    }
    return aliases.get(
        (pipeline.variant_label, pipeline.mode),
        f"{pipeline.variant_label}_{pipeline.mode.lower()}",
    )


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _table1_runtime_rows(
    runtime_summary_rows: Sequence[dict[str, Any]],
    dataset_labels: Sequence[str],
) -> list[dict[str, Any]]:
    """Build a wide runtime table from the selected-seed means."""

    required_variants = {
        "m2m_poselib_cm",
        "hcm",
        "mcm",
        "full_hcm_mc",
    }
    observed_variants = {str(row["variant"]) for row in runtime_summary_rows}
    if observed_variants != required_variants:
        return []
    seed_labels = {str(row["seeds"]) for row in runtime_summary_rows}
    if len(seed_labels) != 1:
        raise ValueError(f"Table 1 runtime rows disagree on seeds: {seed_labels}")
    seed_label = next(iter(seed_labels))
    lookup: dict[tuple[str, str, str], float] = {}
    for row in runtime_summary_rows:
        key = (str(row["variant"]), str(row["dataset"]), str(row["metric"]))
        if key in lookup:
            raise ValueError(f"Duplicate runtime-mean cell for {key}")
        lookup[key] = float(row["mean"])
    columns = {
        "hcm_scorer_us_per_eval": ("hcm", "score_us_per_eval", 1.0),
        "mcm_scorer_us_per_eval": ("mcm", "score_us_per_eval", 1.0),
        "cm_runner_ms_per_pair": ("m2m_poselib_cm", "running_time_s", 1000.0),
        "hcm_runner_ms_per_pair": ("hcm", "running_time_s", 1000.0),
        "mcm_runner_ms_per_pair": ("mcm", "running_time_s", 1000.0),
        "hcm_mc_runner_ms_per_pair": ("full_hcm_mc", "running_time_s", 1000.0),
    }
    rows: list[dict[str, Any]] = []
    for dataset in dataset_labels:
        row: dict[str, Any] = {"dataset": dataset, "seeds": seed_label}
        for column, (variant, metric, scale) in columns.items():
            key = (variant, dataset, metric)
            if key not in lookup:
                raise ValueError(f"Missing Table 1 runtime-mean cell for {key}")
            row[column] = lookup[key] * scale
        rows.append(row)
    return rows


def run_summarize(args: argparse.Namespace) -> None:
    subsets = _load_subsets(args.input_manifest.resolve(), args.datasets)
    _validate_selected_subsets(args, subsets)
    pipelines = _pipeline_specs(args.pipelines)
    dataset_labels = [
        label
        for label in CANONICAL_DATASETS
        if any(subset.dataset_label == label for subset in subsets)
    ]
    extras = sorted(
        {subset.dataset_label for subset in subsets} - set(dataset_labels)
    )
    dataset_labels.extend(extras)

    report: dict[str, dict[str, dict[str, Any]]] = {}
    per_seed_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for pipeline in pipelines:
        publication_variant = _publication_variant(pipeline)
        pipeline_report: dict[str, dict[str, Any]] = {}
        for seed in args.seeds:
            seed_report: dict[str, Any] = {}
            dataset_aucs: dict[str, dict[str, float]] = {}
            pair_total = 0
            for dataset_label in dataset_labels:
                vectors: list[np.ndarray] = []
                result_paths: list[str] = []
                successful_rows = 0
                runtime_vectors: dict[str, list[np.ndarray]] = {
                    "running_time_s": [],
                    "score_us_per_eval": [],
                }
                for subset in subsets:
                    if subset.dataset_label != dataset_label:
                        continue
                    method_dir = (
                        _method_root(
                            args.results_root.resolve(),
                            subset,
                            args.layer,
                            args.debias_rank,
                            args.method_prefix,
                        )
                        / pipeline.variant_dir
                    )
                    result_path = method_dir / _result_filename(
                        pipeline.mode, args.max_iterations, seed
                    )
                    pose_path = (
                        args.results_root.resolve()
                        / subset.root_name
                        / subset.subset_name
                        / "pose_intrinsics.csv"
                    )
                    if not result_path.is_file():
                        raise FileNotFoundError(result_path)
                    expected_pair_ids = expected_pose_pair_indices(pose_path)
                    validated_pose_result_pair_ids(
                        result_path,
                        expected_pair_ids,
                        require_complete=True,
                    )
                    vectors.append(_pose_errors(result_path, pose_path))
                    timing = pd.read_csv(
                        result_path,
                        usecols=lambda column: column
                        in {"status", "running_time_s", "score_us_per_eval"},
                    )
                    missing_timing = ({"status"} | set(runtime_vectors)) - set(
                        timing.columns
                    )
                    if missing_timing:
                        raise ValueError(
                            f"{result_path}: missing runtime columns "
                            f"{sorted(missing_timing)}"
                        )
                    successful_rows += int(
                        timing["status"]
                        .astype("string")
                        .str.strip()
                        .str.lower()
                        .eq("success")
                        .sum()
                    )
                    for metric in runtime_vectors:
                        values = pd.to_numeric(timing[metric], errors="coerce").to_numpy(float)
                        runtime_vectors[metric].append(values[np.isfinite(values)])
                    result_paths.append(str(result_path))
                if not vectors:
                    raise ValueError(f"No subsets found for dataset {dataset_label}")
                errors = np.concatenate(vectors)
                aucs = _error_auc(errors)
                pair_total += int(errors.size)
                dataset_aucs[dataset_label] = aucs
                seed_report[dataset_label] = {
                    "pairs": int(errors.size),
                    **aucs,
                    "result_files": result_paths,
                }
                for threshold in (5.0, 10.0, 20.0):
                    per_seed_rows.append(
                        {
                            "pipeline": pipeline.label,
                            "variant": publication_variant,
                            "association": pipeline.variant_label,
                            "mode": pipeline.mode,
                            "dataset": dataset_label,
                            "threshold_deg": threshold,
                            "seed": seed,
                            "auc": aucs[f"auc@{threshold:g}"],
                            "pairs": int(errors.size),
                            "result_files": ";".join(result_paths),
                        }
                    )
                for metric, chunks in runtime_vectors.items():
                    values = np.concatenate(chunks)
                    expected_runtime_rows = (
                        int(errors.size)
                        if metric == "running_time_s"
                        else successful_rows
                    )
                    if values.size != expected_runtime_rows:
                        raise ValueError(
                            f"Incomplete {metric} coverage for "
                            f"{pipeline.label}/{dataset_label}/seed{seed}: "
                            f"expected {expected_runtime_rows}, found {values.size}"
                        )
                    if values.size == 0:
                        raise ValueError(
                            f"No finite {metric} values for "
                            f"{pipeline.label}/{dataset_label}/seed{seed}"
                        )
                    runtime_rows.append(
                        {
                            "pipeline": pipeline.label,
                            "variant": publication_variant,
                            "association": pipeline.variant_label,
                            "mode": pipeline.mode,
                            "dataset": dataset_label,
                            "seed": seed,
                            "metric": metric,
                            "count": int(values.size),
                            "mean": float(np.mean(values)),
                            "median": float(np.median(values)),
                            "p95": float(np.percentile(values, 95.0)),
                        }
                    )
            macro = {
                f"auc@{threshold:g}": float(
                    np.mean(
                        [
                            dataset_aucs[label][f"auc@{threshold:g}"]
                            for label in dataset_labels
                        ]
                    )
                )
                for threshold in (5.0, 10.0, 20.0)
            }
            seed_report["Macro"] = {"pairs": pair_total, **macro}
            for threshold in (5.0, 10.0, 20.0):
                per_seed_rows.append(
                    {
                        "pipeline": pipeline.label,
                        "variant": publication_variant,
                        "association": pipeline.variant_label,
                        "mode": pipeline.mode,
                        "dataset": "Macro",
                        "threshold_deg": threshold,
                        "seed": seed,
                        "auc": macro[f"auc@{threshold:g}"],
                        "pairs": pair_total,
                        "result_files": "",
                    }
                )
            pipeline_report[str(seed)] = seed_report
        report[pipeline.label] = pipeline_report

    summary_rows: list[dict[str, Any]] = []
    group_keys = sorted(
        {
            (
                str(row["pipeline"]),
                str(row["variant"]),
                str(row["association"]),
                str(row["mode"]),
                str(row["dataset"]),
                float(row["threshold_deg"]),
            )
            for row in per_seed_rows
        }
    )
    seed_text = ",".join(str(seed) for seed in args.seeds)
    for pipeline_label, variant, association, mode, dataset, threshold in group_keys:
        cells = [
            row
            for row in per_seed_rows
            if row["pipeline"] == pipeline_label
            and row["dataset"] == dataset
            and float(row["threshold_deg"]) == threshold
        ]
        observed_seeds = [int(row["seed"]) for row in cells]
        if observed_seeds != list(args.seeds):
            raise ValueError(
                f"Seed rectangle differs for {pipeline_label}/{dataset}/{threshold}: "
                f"{observed_seeds}"
            )
        values = np.asarray([float(row["auc"]) for row in cells], dtype=float)
        summary_rows.append(
            {
                "pipeline": pipeline_label,
                "variant": variant,
                "association": association,
                "mode": mode,
                "dataset": dataset,
                "threshold_deg": threshold,
                "seeds": seed_text,
                "mean": float(np.mean(values)),
                "seed_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )

    runtime_summary_rows: list[dict[str, Any]] = []
    runtime_keys = sorted(
        {
            (
                str(row["pipeline"]),
                str(row["variant"]),
                str(row["association"]),
                str(row["mode"]),
                str(row["dataset"]),
                str(row["metric"]),
            )
            for row in runtime_rows
        }
    )
    for pipeline_label, variant, association, mode, dataset, metric in runtime_keys:
        cells = [
            row
            for row in runtime_rows
            if row["pipeline"] == pipeline_label
            and row["dataset"] == dataset
            and row["metric"] == metric
        ]
        observed_seeds = [int(row["seed"]) for row in cells]
        if observed_seeds != list(args.seeds):
            raise ValueError(
                f"Runtime seed rectangle differs for "
                f"{pipeline_label}/{dataset}/{metric}: {observed_seeds}"
            )
        values = np.asarray([float(row["mean"]) for row in cells], dtype=float)
        runtime_summary_rows.append(
            {
                "pipeline": pipeline_label,
                "variant": variant,
                "association": association,
                "mode": mode,
                "dataset": dataset,
                "metric": metric,
                "seeds": seed_text,
                "mean": float(np.mean(values)),
                "seed_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    per_seed_output = (
        args.per_seed_csv.resolve()
        if args.per_seed_csv is not None
        else output.parent / "auc_per_seed.csv"
    )
    summary_output = (
        args.summary_csv.resolve()
        if args.summary_csv is not None
        else output.parent / "auc_five_seed_summary.csv"
    )
    runtime_per_seed_output = output.parent / "runtime_per_seed.csv"
    runtime_summary_output = output.parent / "runtime_five_seed_summary.csv"
    table1_rows = _table1_runtime_rows(runtime_summary_rows, dataset_labels)
    table1_output = output.parent / "table1_runtime.csv"
    payload = {
        "schema_version": 2,
        "seeds": list(args.seeds),
        "pipelines": [pipeline.label for pipeline in pipelines],
        "results": report,
        "outputs": {
            "per_seed_csv": str(per_seed_output),
            "summary_csv": str(summary_output),
            "runtime_per_seed_csv": str(runtime_per_seed_output),
            "runtime_summary_csv": str(runtime_summary_output),
            **(
                {"table1_runtime_csv": str(table1_output)}
                if table1_rows
                else {}
            ),
        },
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        per_seed_output,
        (
            "pipeline",
            "variant",
            "association",
            "mode",
            "dataset",
            "threshold_deg",
            "seed",
            "auc",
            "pairs",
            "result_files",
        ),
        per_seed_rows,
    )
    _write_csv(
        summary_output,
        (
            "pipeline",
            "variant",
            "association",
            "mode",
            "dataset",
            "threshold_deg",
            "seeds",
            "mean",
            "seed_std",
            "min",
            "max",
        ),
        summary_rows,
    )
    _write_csv(
        runtime_per_seed_output,
        (
            "pipeline",
            "variant",
            "association",
            "mode",
            "dataset",
            "seed",
            "metric",
            "count",
            "mean",
            "median",
            "p95",
        ),
        runtime_rows,
    )
    _write_csv(
        runtime_summary_output,
        (
            "pipeline",
            "variant",
            "association",
            "mode",
            "dataset",
            "metric",
            "seeds",
            "mean",
            "seed_std",
            "min",
            "max",
        ),
        runtime_summary_rows,
    )
    if table1_rows:
        _write_csv(
            table1_output,
            (
                "dataset",
                "seeds",
                "hcm_scorer_us_per_eval",
                "mcm_scorer_us_per_eval",
                "cm_runner_ms_per_pair",
                "hcm_runner_ms_per_pair",
                "mcm_runner_ms_per_pair",
                "hcm_mc_runner_ms_per_pair",
            ),
            table1_rows,
        )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print(output)
    print(per_seed_output)
    print(summary_output)
    print(runtime_per_seed_output)
    print(runtime_summary_output)
    if table1_rows:
        print(table1_output)


def _common_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[2]
    results_root = repository_root / "artifacts" / "matching_estimation_results"
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results-root", type=Path, default=results_root)
    parser.add_argument(
        "--model-name",
        choices=tuple(PAPER_WEIGHTS_IDS),
        default=DEFAULT_MODEL_NAME,
        help="Frozen backbone identity used to validate association manifests.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help=(
            f"One-based matching layer (default {DEFAULT_LAYER}); when "
            "--layer-selection is supplied, defaults to and must equal its best layer."
        ),
    )
    parser.add_argument(
        "--debias-rank",
        type=int,
        default=None,
        help=(
            f"Correction rank (default {DEFAULT_DEBIAS_RANK}); a DINOv2 "
            "--layer-selection forces rank 0."
        ),
    )
    parser.add_argument(
        "--method-prefix",
        default=None,
        help="DINO method directory below each subset root.",
    )
    parser.add_argument(
        "--layer-selection",
        type=Path,
        default=None,
        help=(
            "Optional DINOv2 best_layer.json. It binds layer 16..24, raw rank 0, "
            "the model-specific result tree, and every pose_matching_manifest."
        ),
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=results_root / "experiment_inputs.json",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=repository_root
        / "artifacts"
        / "evaluation"
        / "estimation"
        / "gms_selection",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(CANONICAL_DATASETS),
        help=(
            "Dataset labels; defaults to the canonical six-dataset protocol "
            "and excludes historical NAVI-Wild-v1."
        ),
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    common = _common_parser()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    filter_parser = commands.add_parser("filter", parents=[common])
    filter_parser.add_argument(
        "--gms-binary",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "build"
        / "gms_filter"
        / "gms_filter_csv_m2m",
    )
    filter_parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help=(
            "Explicitly regenerate retained GMS CSVs. By default existing complete "
            "outputs are validated and preserved byte-for-byte."
        ),
    )
    filter_parser.add_argument("--thresholds", type=float, nargs="+")
    filter_parser.add_argument("--grids", type=int, nargs="+")
    filter_parser.add_argument(
        "--candidates",
        type=_parse_gms_candidate,
        nargs="+",
        default=[],
        help="Exact THRESHOLD:GRID points; avoids a Cartesian-product grid.",
    )
    filter_parser.add_argument("--workers", type=int, default=8)
    filter_parser.add_argument(
        "--with-scale",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_WITH_SCALE,
        help="Enable the historical five-scale GMS search (default: enabled).",
    )
    filter_parser.add_argument(
        "--with-rotation",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_WITH_ROTATION,
        help="Enable the historical eight-rotation GMS search (default: enabled).",
    )

    estimate_parser = commands.add_parser("estimate", parents=[common])
    estimate_parser.add_argument(
        "--runner-binary",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "build"
        / "m2m_loransac"
        / "m2m_loransac_runner",
    )
    estimate_parser.add_argument(
        "--pipelines",
        nargs="+",
        required=True,
        help=(
            "Exact association/estimator pairs: mnn:CM, raw:MODE, or "
            "gms:THRESHOLD:GRID:MODE."
        ),
    )
    estimate_parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    estimate_parser.add_argument("--max-iterations", type=int, default=100000)
    estimate_parser.add_argument("--workers", type=int, default=12)

    summarize_parser = commands.add_parser("summarize", parents=[common])
    summarize_parser.add_argument(
        "--pipelines",
        nargs="+",
        required=True,
        help=(
            "Exact association/estimator pairs: mnn:CM, raw:MODE, or "
            "gms:THRESHOLD:GRID:MODE."
        ),
    )
    summarize_parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    summarize_parser.add_argument("--max-iterations", type=int, default=100000)
    summarize_parser.add_argument("--output", type=Path, required=True)
    summarize_parser.add_argument("--per-seed-csv", type=Path)
    summarize_parser.add_argument("--summary-csv", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _resolve_layer_contract(args)
    if args.layer <= 0 or args.debias_rank < 0:
        raise ValueError("--layer must be positive and --debias-rank non-negative")
    if hasattr(args, "seeds"):
        if not args.seeds or min(args.seeds) < 0 or len(args.seeds) != len(set(args.seeds)):
            raise ValueError("--seeds must contain unique non-negative integers")
        args.seeds = tuple(sorted(args.seeds))
    _require_safe_protocol(args)
    method_prefix = Path(args.method_prefix)
    if method_prefix.is_absolute() or not method_prefix.parts or ".." in method_prefix.parts:
        raise ValueError("--method-prefix must be a relative directory without `..`")
    if args.command == "filter":
        run_filter(args)
    elif args.command == "estimate":
        run_estimate(args)
    elif args.command == "summarize":
        run_summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
