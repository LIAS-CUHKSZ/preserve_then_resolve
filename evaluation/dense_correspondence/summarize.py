"""Aggregate dense GT-rank CDFs without pooling patches across image pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.json_utils import strict_json_dumps

from .protocol import (
    ANGULAR_BINS,
    DEFAULT_MAX_K,
    DEFAULT_THRESHOLDS_M,
    DIRECTIONS,
    validate_shard,
)


@dataclass(frozen=True)
class DenseSummaryOptions:
    shard_root: Path
    angular_bin: str
    output_dir: Path
    thresholds_m: tuple[float, ...] = DEFAULT_THRESHOLDS_M
    max_k: int = DEFAULT_MAX_K


@dataclass(frozen=True)
class DenseSummaryResult:
    shard_count: int
    direction_row_count: int
    pair_row_count: int
    category_row_count: int
    summary_row_count: int
    output_dir: Path


def _cdf_columns(max_k: int) -> list[str]:
    columns = ["gt_coverage"]
    for candidate_k in range(1, max_k + 1):
        columns.extend(
            (
                f"directional_cdf_at_{candidate_k}",
                f"mutual_cdf_at_{candidate_k}",
                f"directional_recall_all_at_{candidate_k}",
                f"mutual_recall_all_at_{candidate_k}",
            )
        )
    return columns


def compute_direction_cdf(
    *,
    candidate_error_m: np.ndarray,
    candidate_target_is_object: np.ndarray,
    candidate_target_has_depth: np.ndarray,
    candidate_mutual_entry_k: np.ndarray,
    source_min_object_error_m: np.ndarray,
    threshold_m: float,
    max_k: int,
) -> dict[str, float | int]:
    """Compute conditional GT-rank CDF and all-source recall for one direction.

    The CDF denominator contains queries for which the complete destination
    grid has at least one object patch center below the geometric threshold.
    ``*_recall_all_*`` retains every valid source-object query in the
    denominator and therefore exposes discretization/visibility coverage.
    """
    error = np.asarray(candidate_error_m, dtype=np.float64)
    target_object = np.asarray(candidate_target_is_object, dtype=np.bool_)
    target_depth = np.asarray(candidate_target_has_depth, dtype=np.bool_)
    mutual_entry = np.asarray(candidate_mutual_entry_k)
    minimum_error = np.asarray(source_min_object_error_m, dtype=np.float64)
    if max_k <= 0:
        raise ValueError("max_k must be positive")
    if not np.isfinite(threshold_m) or threshold_m <= 0.0:
        raise ValueError("threshold_m must be positive and finite")
    if error.ndim != 2 or error.shape[1] < max_k:
        raise ValueError("Candidate arrays do not contain the requested K")
    expected = error.shape
    if any(array.shape != expected for array in (target_object, target_depth, mutual_entry)):
        raise ValueError("Candidate geometry arrays must share one shape")
    if minimum_error.shape != (error.shape[0],):
        raise ValueError("Minimum geometry errors must align with source queries")
    if np.any(np.isnan(error)) or np.any(error < 0.0):
        raise ValueError("Candidate errors must be non-negative or +inf")

    n_source = len(minimum_error)
    has_gt = minimum_error < threshold_m
    n_gt = int(np.count_nonzero(has_gt))
    correct = target_object & target_depth & (error < threshold_m)
    row: dict[str, float | int] = {
        "n_source": n_source,
        "n_gt": n_gt,
        "gt_coverage": float(n_gt / n_source) if n_source else float("nan"),
    }
    previous_directional = previous_mutual = -1.0
    for candidate_k in range(1, max_k + 1):
        directional_success = np.any(correct[:, :candidate_k], axis=1)
        mutual_success = np.any(
            correct[:, :max_k]
            & (mutual_entry[:, :max_k] <= candidate_k),
            axis=1,
        )
        if np.any(mutual_success & ~directional_success):
            raise AssertionError("Mutual success must imply directional top-K success")
        if np.any(directional_success & ~has_gt):
            raise AssertionError("Stored top-K positive is absent from geometry coverage")
        directional_count = int(np.count_nonzero(directional_success))
        mutual_count = int(np.count_nonzero(mutual_success))
        directional_cdf = (
            float(directional_count / n_gt) if n_gt else float("nan")
        )
        mutual_cdf = float(mutual_count / n_gt) if n_gt else float("nan")
        directional_all = (
            float(directional_count / n_source) if n_source else float("nan")
        )
        mutual_all = float(mutual_count / n_source) if n_source else float("nan")
        row[f"directional_cdf_at_{candidate_k}"] = directional_cdf
        row[f"mutual_cdf_at_{candidate_k}"] = mutual_cdf
        row[f"directional_recall_all_at_{candidate_k}"] = directional_all
        row[f"mutual_recall_all_at_{candidate_k}"] = mutual_all
        if np.isfinite(directional_cdf) and directional_cdf + 1e-15 < previous_directional:
            raise AssertionError("Directional CDF must be monotonic in K")
        if np.isfinite(mutual_cdf) and mutual_cdf + 1e-15 < previous_mutual:
            raise AssertionError("Mutual CDF must be monotonic in K")
        previous_directional = directional_cdf
        previous_mutual = mutual_cdf
    return row


def _discover_shards(
    options: DenseSummaryOptions,
) -> tuple[list[Path], dict[str, Any]]:
    bin_root = Path(options.shard_root) / f"bin_{options.angular_bin}"
    manifest_path = bin_root / "evaluation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Evaluation manifest is required before summarization: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid evaluation manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("angular_bin") != options.angular_bin:
        raise ValueError("Evaluation manifest angular bin is invalid")
    layers = tuple(int(value) for value in manifest.get("layers", ()))
    pair_indices = tuple(int(value) for value in manifest.get("pair_indices", ()))
    if (
        not layers
        or not pair_indices
        or len(set(layers)) != len(layers)
        or len(set(pair_indices)) != len(pair_indices)
    ):
        raise ValueError("Evaluation manifest needs unique layers and pair_indices")
    if int(manifest.get("pair_count", -1)) != len(pair_indices):
        raise ValueError("Evaluation manifest pair_count disagrees with pair_indices")
    expected_count = len(layers) * len(pair_indices)
    if int(manifest.get("expected_shards", -1)) != expected_count:
        raise ValueError("Evaluation manifest expected_shards is inconsistent")
    if int(manifest.get("max_k", -1)) < options.max_k:
        raise ValueError("Evaluation manifest stores a smaller K than requested")
    root = bin_root / "shards"
    if not root.is_dir():
        raise NotADirectoryError(f"Raw shard directory does not exist: {root}")
    paths = [
        root / f"layer{layer}" / f"pair_{pair_index:06d}.npz"
        for layer in layers
        for pair_index in pair_indices
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Evaluation is incomplete: {len(missing)} expected shards are missing; "
            f"first={missing[0]}"
        )
    extras = sorted(set(root.glob("layer*/pair_*.npz")).difference(paths))
    if extras:
        raise ValueError(
            f"Shard tree contains {len(extras)} files outside the current manifest; "
            "use a fresh output root. First extra: " + str(extras[0])
        )
    return paths, manifest


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty report: {path}")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("CSV rows do not share one stable schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as stream:
        temporary_path = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    try:
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary_path = Path(stream.name)
        stream.write(strict_json_dumps(value, indent=2, sort_keys=True))
        stream.write("\n")
    try:
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _mean_finite(rows: Sequence[Mapping[str, Any]], column: str) -> float:
    values = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def _average_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    metric_columns: Sequence[str],
) -> dict[str, Any]:
    result = dict(identity)
    for column in metric_columns:
        result[column] = _mean_finite(rows, column)
    return result


def _build_pair_rows(
    direction_rows: Sequence[Mapping[str, Any]], metric_columns: Sequence[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in direction_rows:
        grouped[
            (
                int(row["pair_index"]),
                int(row["layer"]),
                int(row["debias_rank"]),
                float(row["threshold_m"]),
            )
        ].append(row)
    results: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: str(row["direction"]))
        if [row["direction"] for row in rows] != list(DIRECTIONS):
            raise ValueError(f"Pair/layer/rank/threshold {key} lacks two directions")
        first = rows[0]
        identity = {
            "angular_bin": first["angular_bin"],
            "pair_index": key[0],
            "object_name": first["object_name"],
            "image_a": first["image_a"],
            "image_b": first["image_b"],
            "angle_degrees": first["angle_degrees"],
            "layer": key[1],
            "debias_rank": key[2],
            "threshold_m": key[3],
            "a_to_b_n_source": int(rows[0]["n_source"]),
            "b_to_a_n_source": int(rows[1]["n_source"]),
            "a_to_b_n_gt": int(rows[0]["n_gt"]),
            "b_to_a_n_gt": int(rows[1]["n_gt"]),
            "cdf_pair_eligible": int(
                int(rows[0]["n_gt"]) > 0 and int(rows[1]["n_gt"]) > 0
            ),
        }
        result = _average_metric_rows(rows, identity, metric_columns)
        if not identity["cdf_pair_eligible"]:
            for column in metric_columns:
                if "_cdf_at_" in column or "_gap_at_" in column:
                    result[column] = float("nan")
        results.append(result)
    return results


def _build_category_rows(
    pair_rows: Sequence[Mapping[str, Any]], metric_columns: Sequence[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[
            (
                str(row["object_name"]),
                int(row["layer"]),
                int(row["debias_rank"]),
                float(row["threshold_m"]),
            )
        ].append(row)
    results: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        identity = {
            "angular_bin": rows[0]["angular_bin"],
            "object_name": key[0],
            "layer": key[1],
            "debias_rank": key[2],
            "threshold_m": key[3],
            "pair_count": len(rows),
            "cdf_eligible_pair_count": sum(int(row["cdf_pair_eligible"]) for row in rows),
        }
        identity["cdf_category_eligible"] = int(
            identity["cdf_eligible_pair_count"] > 0
        )
        results.append(_average_metric_rows(rows, identity, metric_columns))
    return results


def _build_summary_rows(
    category_rows: Sequence[Mapping[str, Any]], metric_columns: Sequence[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in category_rows:
        grouped[
            (int(row["layer"]), int(row["debias_rank"]), float(row["threshold_m"]))
        ].append(row)
    results: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        identity = {
            "angular_bin": rows[0]["angular_bin"],
            "layer": key[0],
            "debias_rank": key[1],
            "threshold_m": key[2],
            "category_count": len(rows),
            "cdf_eligible_category_count": sum(
                int(row["cdf_category_eligible"]) for row in rows
            ),
            "pair_count": sum(int(row["pair_count"]) for row in rows),
            "cdf_eligible_pair_count": sum(
                int(row["cdf_eligible_pair_count"]) for row in rows
            ),
        }
        results.append(_average_metric_rows(rows, identity, metric_columns))
    return results


def _build_threshold_mean_rows(
    rows: Sequence[Mapping[str, Any]],
    metric_columns: Sequence[str],
    thresholds_m: Sequence[float],
) -> list[dict[str, Any]]:
    """Average already-aggregated metrics equally across strict thresholds."""

    expected_thresholds = tuple(sorted(float(value) for value in thresholds_m))
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), int(row["debias_rank"]))].append(row)
    results: list[dict[str, Any]] = []
    for key in sorted(grouped):
        threshold_rows = sorted(
            grouped[key], key=lambda row: float(row["threshold_m"])
        )
        observed = tuple(float(row["threshold_m"]) for row in threshold_rows)
        if observed != expected_thresholds:
            raise ValueError(
                f"Layer/rank {key} thresholds {observed} do not match "
                f"{expected_thresholds}"
            )
        result: dict[str, Any] = {
            "angular_bin": threshold_rows[0]["angular_bin"],
            "layer": key[0],
            "debias_rank": key[1],
            "thresholds_m": "|".join(f"{value:g}" for value in expected_thresholds),
            "threshold_count": len(expected_thresholds),
        }
        for column in metric_columns:
            values = np.asarray(
                [float(row[column]) for row in threshold_rows], dtype=np.float64
            )
            result[column] = (
                float(np.mean(values)) if np.isfinite(values).all() else float("nan")
            )
        results.append(result)
    return results


def _build_match_count_reports(
    pair_rows: Sequence[Mapping[str, Any]], max_k: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped_category: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped_category[
            (str(row["object_name"]), int(row["layer"]), int(row["debias_rank"]))
        ].append(row)
    category_rows: list[dict[str, Any]] = []
    for key in sorted(grouped_category):
        rows = grouped_category[key]
        result: dict[str, Any] = {
            "angular_bin": rows[0]["angular_bin"],
            "object_name": key[0],
            "layer": key[1],
            "debias_rank": key[2],
            "pair_count": len(rows),
        }
        for candidate_k in range(1, max_k + 1):
            result[f"pair_mean_match_count_at_{candidate_k}"] = _mean_finite(
                rows, f"mutual_match_count_at_{candidate_k}"
            )
        category_rows.append(result)

    grouped_config: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped_config[(int(row["layer"]), int(row["debias_rank"]))].append(row)
    category_lookup: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in category_rows:
        category_lookup[(int(row["layer"]), int(row["debias_rank"]))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped_config):
        pairs = grouped_config[key]
        categories = category_lookup[key]
        row: dict[str, Any] = {
            "angular_bin": pairs[0]["angular_bin"],
            "layer": key[0],
            "debias_rank": key[1],
            "pair_count": len(pairs),
            "category_count": len(categories),
        }
        for candidate_k in range(1, max_k + 1):
            count_column = f"mutual_match_count_at_{candidate_k}"
            pair_values = np.asarray(
                [float(item[count_column]) for item in pairs], dtype=np.float64
            )
            row[f"category_macro_mean_match_count_at_{candidate_k}"] = _mean_finite(
                categories, f"pair_mean_match_count_at_{candidate_k}"
            )
            row[f"pair_median_at_{candidate_k}"] = float(np.median(pair_values))
            row[f"pair_p90_at_{candidate_k}"] = float(np.percentile(pair_values, 90))
        summary_rows.append(row)
    return category_rows, summary_rows


def summarize_dense_correspondence(options: DenseSummaryOptions) -> DenseSummaryResult:
    """Regenerate hierarchical CDF and complexity reports from raw top-K shards."""
    if options.angular_bin not in ANGULAR_BINS:
        raise ValueError(f"angular_bin must be one of {ANGULAR_BINS}")
    thresholds = tuple(sorted(set(float(value) for value in options.thresholds_m)))
    if not thresholds or min(thresholds) <= 0.0 or not np.isfinite(thresholds).all():
        raise ValueError("thresholds_m must contain positive finite values")
    if options.max_k <= 0:
        raise ValueError("max_k must be positive")
    paths, evaluation_manifest = _discover_shards(options)
    manifest_layers = tuple(int(value) for value in evaluation_manifest["layers"])
    manifest_pair_indices = tuple(
        int(value) for value in evaluation_manifest["pair_indices"]
    )
    manifest_debias_ranks = tuple(
        int(value) for value in evaluation_manifest.get("debias_ranks", ())
    )
    if not manifest_debias_ranks or len(set(manifest_debias_ranks)) != len(
        manifest_debias_ranks
    ):
        raise ValueError("Evaluation manifest debias_ranks are invalid")
    manifest_max_k = int(evaluation_manifest["max_k"])
    manifest_fingerprints = evaluation_manifest.get("protocol_fingerprints")
    expected_fingerprint_keys = {f"layer{layer}" for layer in manifest_layers}
    if not isinstance(manifest_fingerprints, dict) or set(manifest_fingerprints) != (
        expected_fingerprint_keys
    ):
        raise ValueError("Evaluation manifest protocol_fingerprints are incomplete")
    raw_identities = evaluation_manifest.get("pair_identities")
    if not isinstance(raw_identities, list) or len(raw_identities) != len(
        manifest_pair_indices
    ):
        raise ValueError("Evaluation manifest pair_identities are incomplete")
    manifest_identities: dict[int, Mapping[str, Any]] = {}
    for identity in raw_identities:
        if not isinstance(identity, dict) or "pair_index" not in identity:
            raise ValueError("Evaluation manifest contains an invalid pair identity")
        pair_index = int(identity["pair_index"])
        if pair_index in manifest_identities:
            raise ValueError("Evaluation manifest contains duplicate pair identities")
        manifest_identities[pair_index] = identity
    if set(manifest_identities) != set(manifest_pair_indices):
        raise ValueError("Evaluation manifest pair identities disagree with pair_indices")
    direction_rows: list[dict[str, Any]] = []
    match_pair_rows: list[dict[str, Any]] = []
    layer_pair_sets: dict[int, set[int]] = defaultdict(set)
    fingerprints: dict[int, str] = {}
    common_debias_ranks: tuple[int, ...] | None = None
    stored_max_k: int | None = None
    shared_protocol: dict[str, Any] | None = None

    for path in paths:
        try:
            expected_layer = int(path.parent.name.removeprefix("layer"))
            expected_pair_index = int(path.stem.removeprefix("pair_"))
        except ValueError as exc:
            raise ValueError(f"Unexpected shard path: {path}") from exc
        expected_fingerprint = str(
            manifest_fingerprints[f"layer{expected_layer}"]
        )
        data = validate_shard(
            path,
            expected_fingerprint=expected_fingerprint,
            expected_bin=options.angular_bin,
            expected_pair_index=expected_pair_index,
            expected_layer=expected_layer,
            expected_debias_ranks=manifest_debias_ranks,
            expected_max_k=manifest_max_k,
        )
        layer = int(np.asarray(data["layer"]).item())
        pair_index = int(np.asarray(data["pair_index"]).item())
        debias_ranks = tuple(int(value) for value in data["debias_ranks"])
        shard_max_k = int(np.asarray(data["max_k"]).item())
        if shard_max_k < options.max_k:
            raise ValueError(
                f"Shard {path} stores K={shard_max_k}, requested K={options.max_k}"
            )
        if stored_max_k is None:
            stored_max_k = shard_max_k
        elif shard_max_k != stored_max_k:
            raise ValueError("All rank shards must store the same maximum K")
        fingerprint = str(np.asarray(data["protocol_fingerprint"]).item())
        previous = fingerprints.setdefault(layer, fingerprint)
        if previous != fingerprint:
            raise ValueError(f"Layer {layer} shards contain mixed protocol fingerprints")
        if common_debias_ranks is None:
            common_debias_ranks = debias_ranks
        elif debias_ranks != common_debias_ranks:
            raise ValueError("All rank shards must contain the same debias-rank set")
        if pair_index in layer_pair_sets[layer]:
            raise ValueError(f"Duplicate shard for layer {layer}, pair {pair_index}")
        layer_pair_sets[layer].add(pair_index)

        image_a = str(np.asarray(data["image_a"]).item())
        image_b = str(np.asarray(data["image_b"]).item())
        object_name = str(np.asarray(data["object_name"]).item())
        angle = float(np.asarray(data["angle_degrees"]).item())
        identity = manifest_identities[pair_index]
        if (
            str(identity.get("object_name")) != object_name
            or str(identity.get("image_a")) != image_a
            or str(identity.get("image_b")) != image_b
            or not math.isclose(
                float(identity.get("angle_degrees", float("nan"))),
                angle,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"Shard identity disagrees with manifest for pair {pair_index}"
            )
        embedded_protocol = json.loads(str(np.asarray(data["protocol_json"]).item()))
        comparable_protocol = {
            key: value
            for key, value in embedded_protocol.items()
            if key not in {"layer", "basis_identities", "descriptor_snapshot_id"}
        }
        if shared_protocol is None:
            shared_protocol = comparable_protocol
        elif comparable_protocol != shared_protocol:
            raise ValueError(
                "Layers do not share one dataset/model/preprocessing protocol"
            )
        common_identity = {
            "angular_bin": options.angular_bin,
            "pair_index": pair_index,
            "object_name": object_name,
            "image_a": image_a,
            "image_b": image_b,
            "angle_degrees": angle,
            "layer": layer,
        }
        for debias_row, debias_rank in enumerate(debias_ranks):
            match_row: dict[str, Any] = {
                **common_identity,
                "debias_rank": debias_rank,
            }
            for candidate_k in range(1, options.max_k + 1):
                count = int(data["mutual_match_count_at_k"][debias_row, candidate_k - 1])
                match_row[f"mutual_match_count_at_{candidate_k}"] = count
            match_pair_rows.append(match_row)
            for direction in DIRECTIONS:
                for threshold in thresholds:
                    metrics = compute_direction_cdf(
                        candidate_error_m=data[f"{direction}_candidate_error_m"][debias_row],
                        candidate_target_is_object=data[
                            f"{direction}_candidate_target_is_object"
                        ][debias_row],
                        candidate_target_has_depth=data[
                            f"{direction}_candidate_target_has_depth"
                        ][debias_row],
                        candidate_mutual_entry_k=data[
                            f"{direction}_candidate_mutual_entry_k"
                        ][debias_row],
                        source_min_object_error_m=data[
                            f"{direction}_source_min_object_error_m"
                        ],
                        threshold_m=threshold,
                        max_k=options.max_k,
                    )
                    direction_rows.append(
                        {
                            **common_identity,
                            "debias_rank": debias_rank,
                            "threshold_m": threshold,
                            "direction": direction,
                            **metrics,
                        }
                    )

    if common_debias_ranks is None or stored_max_k is None:
        raise RuntimeError("No rank-CDF rows were produced")
    reference_pairs = next(iter(layer_pair_sets.values()))
    for layer, pair_set in layer_pair_sets.items():
        if pair_set != reference_pairs:
            raise ValueError(f"Incomplete shard rectangle at layer {layer}")
    if set(layer_pair_sets) != set(manifest_layers) or reference_pairs != set(
        manifest_pair_indices
    ):
        raise ValueError("Shard rectangle disagrees with the evaluation manifest")
    if tuple(common_debias_ranks) != manifest_debias_ranks:
        raise ValueError("Shard debias ranks disagree with the evaluation manifest")
    if stored_max_k != manifest_max_k:
        raise ValueError("Stored K disagrees with the evaluation manifest")
    if evaluation_manifest.get("dataset_snapshot_id") != shared_protocol.get(
        "dataset_snapshot_id"
    ):
        raise ValueError("Evaluation manifest dataset snapshot is inconsistent")
    metric_columns = _cdf_columns(options.max_k)
    direction_rows.sort(
        key=lambda row: (
            int(row["pair_index"]),
            int(row["layer"]),
            int(row["debias_rank"]),
            float(row["threshold_m"]),
            str(row["direction"]),
        )
    )
    pair_rows = _build_pair_rows(direction_rows, metric_columns)
    category_rows = _build_category_rows(pair_rows, metric_columns)
    summary_rows = _build_summary_rows(category_rows, metric_columns)
    threshold_mean_summary = _build_threshold_mean_rows(
        summary_rows, metric_columns, thresholds
    )
    match_pair_rows.sort(
        key=lambda row: (
            int(row["pair_index"]), int(row["layer"]), int(row["debias_rank"])
        )
    )
    match_category_rows, match_summary_rows = _build_match_count_reports(
        match_pair_rows, options.max_k
    )

    output_dir = Path(options.output_dir)
    for filename, rows in (
        ("direction_cdf.csv", direction_rows),
        ("pair_cdf.csv", pair_rows),
        ("category_cdf.csv", category_rows),
        ("summary_cdf.csv", summary_rows),
        ("summary_cdf_threshold_mean.csv", threshold_mean_summary),
        ("pair_match_counts.csv", match_pair_rows),
        ("category_match_counts.csv", match_category_rows),
        ("summary_match_counts.csv", match_summary_rows),
    ):
        _atomic_write_csv(output_dir / filename, rows)

    summary_document = {
        "angular_bin": options.angular_bin,
        "thresholds_m": list(thresholds),
        "threshold_comparison": "strict-less-than",
        "primary_threshold_aggregation": (
            "equal arithmetic mean of the configured threshold-specific recalls"
        ),
        "cdf_denominator": "source queries with a destination object patch below threshold",
        "all_source_recall_denominator": "all source object patches with valid depth",
        "conditional_pair_eligibility": (
            "both directions must contain at least one threshold-valid geometric "
            "destination; otherwise that pair CDF is excluded"
        ),
        "aggregation": "direction-equal-pair-then-pair-equal-category-then-category-equal-bin",
        "max_k": options.max_k,
        "stored_max_k": stored_max_k,
        "target_search": "all complete patches including background",
        "layers": sorted(layer_pair_sets),
        "debias_ranks": list(common_debias_ranks),
        "pair_count": len(reference_pairs),
        "category_count": len({str(row["object_name"]) for row in category_rows}),
        "shard_count": len(paths),
        "direction_row_count": len(direction_rows),
        "pair_row_count": len(pair_rows),
        "category_row_count": len(category_rows),
        "summary_row_count": len(summary_rows),
        "protocol_fingerprints": {
            f"layer{layer}": fingerprints[layer] for layer in sorted(fingerprints)
        },
        "summary_cdf": summary_rows,
        "summary_cdf_threshold_mean": threshold_mean_summary,
        "summary_match_counts": match_summary_rows,
    }
    _atomic_write_json(output_dir / "summary.json", summary_document)
    return DenseSummaryResult(
        shard_count=len(paths),
        direction_row_count=len(direction_rows),
        pair_row_count=len(pair_rows),
        category_row_count=len(category_rows),
        summary_row_count=len(summary_rows),
        output_dir=output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize dense directional/mutual GT-rank CDF shards."
    )
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--bin", dest="angular_bin", choices=ANGULAR_BINS, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--threshold-m",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS_M),
        help="Strict 3D thresholds in metres (default: 0.01 0.02 0.05).",
    )
    parser.add_argument("--max-k", type=int, default=DEFAULT_MAX_K)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if argv and argv[0] == "--":
        argv = argv[1:]
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or (
        args.shard_root / f"bin_{args.angular_bin}" / "reports"
    )
    result = summarize_dense_correspondence(
        DenseSummaryOptions(
            shard_root=args.shard_root,
            angular_bin=args.angular_bin,
            output_dir=output_dir,
            thresholds_m=tuple(args.threshold_m),
            max_k=args.max_k,
        )
    )
    print(
        f"Summarized {result.shard_count} shards: "
        f"directions={result.direction_row_count}, pairs={result.pair_row_count}, "
        f"categories={result.category_row_count}, configs={result.summary_row_count}; "
        f"output={result.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
