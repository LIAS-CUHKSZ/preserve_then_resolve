"""Pose AUC and runtime metrics used by the paper experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Subset:
    """One result-tree leaf beneath the shared estimation root."""

    root_name: str
    subset_name: str
    require_pose_manifest: bool = False


DATASETS: dict[str, tuple[Subset, ...]] = {
    "ScanNet": (Subset("scannet_resized", "scannet_test_pairs_with_gt"),),
    "MegaDepth": (Subset("megadepth_resized", "test_1500"),),
    "NAVI-Multi": tuple(
        Subset("NAVI_resized", f"multiview_{angular_bin}")
        for angular_bin in ("0-40", "40-80", "80-120")
    ),
    "NAVI-Wild": tuple(
        Subset(
            "NAVI_wild",
            f"wildset_{angular_bin}",
            require_pose_manifest=True,
        )
        for angular_bin in ("0-40", "40-80", "80-120")
    ),
    "METU-CC": (Subset("METU_VisTIR_resized", "cloudy_cloudy"),),
    "METU-CS": (Subset("METU_VisTIR_resized", "cloudy_sunny"),),
}

def dataset_registry() -> dict[str, tuple[Subset, ...]]:
    """Return the current paper datasets."""
    return dict(DATASETS)


def _trapz(y: list[float], x: list[float]) -> float:
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(y, x))


def error_auc(
    errors: np.ndarray | list[float],
    thresholds: tuple[float, ...] = (5.0, 10.0, 20.0),
) -> dict[str, float]:
    """Area under recall-vs-pose-error, normalized by each threshold.

    This is the same integration rule as ``analyze_all.py`` and returns
    percentages in ``[0, 100]``.
    """
    sorted_errors = [0.0] + sorted(float(value) for value in np.asarray(errors).ravel())
    recall = list(np.linspace(0.0, 1.0, len(sorted_errors)))
    aucs: dict[str, float] = {}
    for threshold in thresholds:
        last_index = int(np.searchsorted(sorted_errors, threshold))
        y = recall[:last_index] + [recall[last_index - 1]]
        x = sorted_errors[:last_index] + [threshold]
        aucs[f"auc@{threshold:g}"] = _trapz(y, x) / threshold * 100.0
    return aucs


def pose_error_vector(result_csv: Path, total_pairs: int) -> np.ndarray:
    """Return max(rotation, translation) error, padding failed/missing pairs to 180 deg."""
    if total_pairs < 0:
        raise ValueError("total_pairs must be non-negative")
    frame = pd.read_csv(result_csv)
    required = {"rotation_error_deg", "translation_error_deg"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{result_csv} is missing columns: {sorted(missing)}")
    rotation = frame["rotation_error_deg"].to_numpy().reshape(-1, 1)
    translation = frame["translation_error_deg"].to_numpy().reshape(-1, 1)
    rotation = np.where(np.isnan(rotation), 180.0, rotation)
    translation = np.where(np.isnan(translation), 180.0, translation)
    row_count = rotation.shape[0]
    if row_count < total_pairs:
        pad = total_pairs - row_count
        rotation = np.concatenate((rotation, np.full((pad, 1), 180.0)))
        translation = np.concatenate((translation, np.full((pad, 1), 180.0)))
    elif row_count > total_pairs:
        rotation = rotation[:total_pairs]
        translation = translation[:total_pairs]
    return np.max(np.concatenate((rotation, translation), axis=1), axis=1)


def pose_error_vector_for_indices(
    result_csv: Path, expected_pair_indices: Sequence[int]
) -> np.ndarray:
    """Align pose errors by pair ID and score every missing expected pair as failure."""
    expected = tuple(int(value) for value in expected_pair_indices)
    if len(set(expected)) != len(expected):
        raise ValueError("Expected pose pair indices must be unique")
    frame = pd.read_csv(result_csv)
    required = {"pair_idx", "rotation_error_deg", "translation_error_deg"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{result_csv} is missing columns: {sorted(missing)}")
    pair_indices = pd.to_numeric(frame["pair_idx"], errors="raise").astype(np.int64)
    if pair_indices.duplicated().any():
        raise ValueError(f"{result_csv} contains duplicate pair_idx values")
    unexpected = sorted(set(pair_indices.tolist()) - set(expected))
    if unexpected:
        raise ValueError(f"{result_csv} contains unexpected pair_idx values: {unexpected[:5]}")
    aligned = frame.assign(pair_idx=pair_indices).set_index("pair_idx").reindex(expected)
    rotation = pd.to_numeric(aligned["rotation_error_deg"], errors="coerce").to_numpy()
    translation = pd.to_numeric(aligned["translation_error_deg"], errors="coerce").to_numpy()
    rotation = np.where(np.isfinite(rotation), rotation, 180.0)
    translation = np.where(np.isfinite(translation), translation, 180.0)
    return np.maximum(rotation, translation)


def validated_pose_result_pair_ids(
    result_csv: Path,
    expected_pair_indices: Sequence[int],
    *,
    require_complete: bool = False,
) -> tuple[int, ...]:
    """Validate rows needed for safe estimator resume and return their pair IDs.

    Skipped estimates may store non-finite pose errors and scorer timings, but
    every retained row must be structurally complete and carry a finite
    runtime. Successful rows must additionally carry finite pose errors and a
    finite scorer timing.  Resume callers may accept a validated prefix; final
    summaries must pass ``require_complete=True`` so an interrupted result is
    never silently scored as estimator failure.
    """

    expected = tuple(int(value) for value in expected_pair_indices)
    if len(expected) != len(set(expected)):
        raise ValueError("Expected pose pair indices must be unique")
    frame = pd.read_csv(result_csv)
    required = {
        "pair_idx",
        "status",
        "rotation_error_deg",
        "translation_error_deg",
        "running_time_s",
        "score_us_per_eval",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{result_csv} is missing columns: {sorted(missing)}")

    raw_pair_ids = pd.to_numeric(frame["pair_idx"], errors="raise").to_numpy(float)
    if not np.isfinite(raw_pair_ids).all() or not np.equal(
        raw_pair_ids, np.floor(raw_pair_ids)
    ).all():
        raise ValueError(f"{result_csv} contains non-integral pair_idx values")
    pair_ids = raw_pair_ids.astype(np.int64)
    if len(pair_ids) != len(set(pair_ids.tolist())):
        raise ValueError(f"{result_csv} contains duplicate pair_idx values")
    unexpected = sorted(set(pair_ids.tolist()) - set(expected))
    if unexpected:
        raise ValueError(
            f"{result_csv} contains unexpected pair_idx values: {unexpected[:5]}"
        )

    status = frame["status"].astype("string")
    if status.isna().any() or status.str.strip().eq("").any():
        raise ValueError(f"{result_csv} contains an empty estimator status")
    normalized_status = status.str.strip().str.lower()
    if not normalized_status.isin(("success", "skipped")).all():
        raise ValueError(f"{result_csv} contains an unsupported estimator status")
    successful = normalized_status.eq("success").to_numpy(bool)
    for column in ("rotation_error_deg", "translation_error_deg"):
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        if np.any(np.isfinite(values) & (values < 0.0)) or np.any(
            successful & ~np.isfinite(values)
        ):
            raise ValueError(f"{result_csv} contains invalid {column} values")

    runtime = pd.to_numeric(frame["running_time_s"], errors="raise").to_numpy(float)
    if not np.isfinite(runtime).all() or np.any(runtime < 0.0):
        raise ValueError(f"{result_csv} contains invalid running_time_s values")
    scorer = pd.to_numeric(frame["score_us_per_eval"], errors="raise").to_numpy(float)
    if np.any(np.isfinite(scorer) & (scorer < 0.0)) or np.any(
        successful & ~np.isfinite(scorer)
    ):
        raise ValueError(f"{result_csv} contains invalid score_us_per_eval values")
    if require_complete and set(pair_ids.tolist()) != set(expected):
        missing_ids = sorted(set(expected) - set(pair_ids.tolist()))
        raise ValueError(
            f"{result_csv} is incomplete; missing expected pair_idx values: "
            f"{missing_ids[:5]}"
        )
    return tuple(int(value) for value in pair_ids)


def expected_pose_pair_indices(
    pose_csv: Path, *, require_manifest: bool = False
) -> tuple[int, ...]:
    """Read the fixed evaluation denominator and validate its pose binding."""
    frame = pd.read_csv(pose_csv, usecols=["pair_idx"])
    values = pd.to_numeric(frame["pair_idx"], errors="raise").astype(np.int64)
    if values.duplicated().any():
        raise ValueError(f"{pose_csv} contains duplicate pair_idx values")
    expected = tuple(int(value) for value in values)
    binding_path = pose_csv.parent / "pose_intrinsics_manifest.json"
    if not binding_path.is_file():
        if require_manifest:
            raise FileNotFoundError(
                f"Pair-bound pose manifest is required: {binding_path}"
            )
        return expected
    payload = json.loads(binding_path.read_text(encoding="utf-8"))
    pose_hash = hashlib.sha256(pose_csv.read_bytes()).hexdigest()
    if payload.get("pose_intrinsics_sha256") != pose_hash:
        raise ValueError(f"{binding_path}: pose_intrinsics_sha256 mismatch")
    if payload.get("pair_count") != len(expected):
        raise ValueError(f"{binding_path}: pair_count mismatch")
    return expected


def count_matching_csvs(method_dir: Path) -> int:
    """Count ``matching_*.csv`` files used to infer the evaluation denominator."""
    return len(tuple(method_dir.glob("matching_*.csv")))


def cost_stats(result_csv: Path) -> dict[str, float]:
    """Mean refinements, iterations, and running time for one subset CSV."""
    frame = pd.read_csv(result_csv)

    def column_mean(name: str) -> float:
        if name not in frame:
            return float("nan")
        return float(pd.to_numeric(frame[name], errors="coerce").mean())

    return {
        "refinements": column_mean("refinements"),
        "iterations": column_mean("iterations"),
        "running_time_s": column_mean("running_time_s"),
    }
