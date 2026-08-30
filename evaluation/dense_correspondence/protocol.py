"""Fingerprint and atomic shard contract for geometric rank-CDF evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


RAW_SHARD_SCHEMA_VERSION = 3
ANGULAR_BINS = ("0-40", "40-80", "80-120")
DEFAULT_LAYERS = tuple(range(16, 25))
DEFAULT_DEBIAS_RANKS = (0, 100, 200, 300, 400, 500, 600)
ALLOWED_DEBIAS_RANKS = (0, 10, 20, 50, 100, 200, 300, 400, 500, 600)
# Compatibility name used by callers predating the rank-CDF redesign.
DEFAULT_RANKS = DEFAULT_DEBIAS_RANKS
DEFAULT_THRESHOLDS_M = (0.01, 0.02, 0.05)
DEFAULT_MAX_K = 8
DIRECTIONS = ("a_to_b", "b_to_a")

DINOV3_MODEL_NAME = "dinov3_vitl16"
DINOV2_MODEL_NAME = "dinov2_vitl14_reg"

PROTOCOL_SPECIFICATION: dict[str, Any] = {
    "schema_version": RAW_SHARD_SCHEMA_VERSION,
    "study_variant": "main-paper-layer-rank-sweep-v1",
    "model": "dinov3_vitl16",
    "patch_size": 16,
    "layer_numbering": "one-based-block-output",
    "intermediate_normalization": "fixed-final-layer-norm",
    "descriptor_dtype": "float32",
    "long_edge": 1024,
    "preserve_aspect_ratio": True,
    "downscale_only": False,
    "padding": "bottom-right-zero-to-multiple-of-16",
    "evaluation_grid": "native-complete-patch-grid",
    "patch_geometry_sampling": "resized-patch-center-pixel-inverse-nearest-v3",
    "source_rule": "complete-and-center-object-mask-and-valid-depth",
    "target_search_rule": "all-complete-patches-including-background",
    "geometry_positive_rule": "target-object-and-valid-depth-and-strict-3d-error",
    "depth_decode": "z_m=655.35/raw_uint16;zero-invalid",
    "pose": "object-to-camera;quaternion-wxyz;translation-mm-to-m",
    "error": "norm(R_ds@X_s+t_ds-X_d)_m-float64",
    "candidate_order": "cosine-descending;target-index-ascending-on-exact-ties",
    "mutual_entry_k": "max(forward-one-based-rank,reverse-one-based-rank)",
    "filters": "none-no-ratio-no-one-to-one",
    "raw_storage": "full-target-top-k-index-cosine-mask-depth-error-and-mutual-entry",
    "full_graph_count": "all-complete-patch-mutual-edges-without-mask",
}

# Keep ``PROTOCOL_SPECIFICATION`` byte-for-byte stable: its canonical JSON is
# part of every retained DINOv3 shard fingerprint.  The DINOv2 study uses the
# same geometric protocol with the model's native patch-14 lattice and no
# positional-bias projection.
DINOV2_PROTOCOL_SPECIFICATION: dict[str, Any] = {
    **PROTOCOL_SPECIFICATION,
    "study_variant": "dinov2-raw-layer-sweep-v1",
    "model": DINOV2_MODEL_NAME,
    "patch_size": 14,
    "padding": "bottom-right-zero-to-multiple-of-14",
    "positional_bias_correction": "none",
}

MODEL_PROTOCOL_SPECIFICATIONS: dict[str, dict[str, Any]] = {
    DINOV3_MODEL_NAME: PROTOCOL_SPECIFICATION,
    DINOV2_MODEL_NAME: DINOV2_PROTOCOL_SPECIFICATION,
}
MODEL_CORRECTION_MODES = {
    DINOV3_MODEL_NAME: "null-space-projection",
    DINOV2_MODEL_NAME: "none",
}


def protocol_specification_for_model(model_name: str) -> dict[str, Any]:
    """Return the locked dense protocol for a supported frozen backbone."""

    try:
        return MODEL_PROTOCOL_SPECIFICATIONS[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_PROTOCOL_SPECIFICATIONS))
        raise ValueError(
            f"Unsupported dense-correspondence model {model_name!r}; expected {supported}"
        ) from exc


def patch_size_for_model(model_name: str) -> int:
    return int(protocol_specification_for_model(model_name)["patch_size"])


def correction_mode_for_model(model_name: str) -> str:
    protocol_specification_for_model(model_name)
    return MODEL_CORRECTION_MODES[model_name]


_DIRECTION_SUFFIXES = (
    "source_patch_index",
    "source_min_object_error_m",
    "candidate_target_patch_index",
    "candidate_cosine",
    "candidate_target_is_object",
    "candidate_target_has_depth",
    "candidate_error_m",
    "candidate_mutual_entry_k",
)

_PROTOCOL_PAYLOAD_KEYS = {
    "protocol",
    "angular_bin",
    "pair_file_sha256",
    "estimation_split_sha256",
    "model_name",
    "weights_id",
    "layer",
    "debias_ranks",
    "basis_identities",
    "dataset_snapshot_id",
    "descriptor_snapshot_id",
    "long_edge",
    "patch_size",
    "max_k",
}


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def protocol_fingerprint(payload: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def make_protocol_payload(
    *,
    angular_bin: str,
    pair_file_sha256: str,
    estimation_split_sha256: str,
    model_name: str,
    weights_id: str,
    layer: int,
    debias_ranks: Sequence[int],
    basis_identities: Mapping[str, str],
    dataset_snapshot_id: str,
    descriptor_snapshot_id: str,
    long_edge: int,
    patch_size: int,
    max_k: int,
) -> dict[str, Any]:
    """Build the complete layer-level payload embedded in every shard."""
    specification = protocol_specification_for_model(model_name)
    expected_patch_size = int(specification["patch_size"])
    if int(patch_size) != expected_patch_size:
        raise ValueError(
            f"Dense protocol {model_name} requires patch_size={expected_patch_size}, "
            f"got {patch_size}"
        )
    normalized_ranks = [int(rank) for rank in debias_ranks]
    if model_name == DINOV2_MODEL_NAME:
        if normalized_ranks != [0]:
            raise ValueError("DINOv2 dense evaluation is raw-only and requires rank 0")
        if basis_identities:
            raise ValueError("DINOv2 dense evaluation does not accept debias bases")
    return {
        "protocol": specification,
        "angular_bin": angular_bin,
        "pair_file_sha256": pair_file_sha256,
        "estimation_split_sha256": estimation_split_sha256,
        "model_name": model_name,
        "weights_id": weights_id,
        "layer": int(layer),
        "debias_ranks": normalized_ranks,
        "basis_identities": dict(sorted(basis_identities.items())),
        "dataset_snapshot_id": dataset_snapshot_id,
        "descriptor_snapshot_id": descriptor_snapshot_id,
        "long_edge": int(long_edge),
        "patch_size": int(patch_size),
        "max_k": int(max_k),
    }


def shard_path(root: Path, angular_bin: str, layer: int, pair_index: int) -> Path:
    return (
        Path(root)
        / f"bin_{angular_bin}"
        / "shards"
        / f"layer{layer}"
        / f"pair_{pair_index:06d}.npz"
    )


def atomic_save_shard(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.stem}.", suffix=".npz", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(stream, **payload)
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _scalar(data: Mapping[str, Any], key: str) -> Any:
    return np.asarray(data[key]).reshape(()).item()


def _require_dtype(name: str, array: np.ndarray, dtype: np.dtype[Any]) -> None:
    if array.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {array.dtype}")


def _complete_indices(
    grid: np.ndarray, resized: np.ndarray, patch_size: int = 16
) -> np.ndarray:
    grid_h, grid_w = (int(value) for value in grid)
    resized_h, resized_w = (int(value) for value in resized)
    rows, columns = np.meshgrid(
        np.arange(resized_h // patch_size, dtype=np.int64),
        np.arange(resized_w // patch_size, dtype=np.int64),
        indexing="ij",
    )
    return rows.reshape(-1) * grid_w + columns.reshape(-1)


def _validate_embedded_protocol(payload: Mapping[str, Any]) -> None:
    missing = _PROTOCOL_PAYLOAD_KEYS.difference(payload)
    if missing:
        raise ValueError(f"Embedded protocol is missing keys: {sorted(missing)}")
    model_name = payload.get("model_name")
    if not isinstance(model_name, str):
        raise ValueError("Embedded protocol model_name must be a string")
    try:
        expected_specification = protocol_specification_for_model(model_name)
    except ValueError as exc:
        raise ValueError("Embedded protocol model is incompatible") from exc
    if payload.get("protocol") != expected_specification:
        raise ValueError("Embedded protocol specification is incompatible")
    if payload.get("angular_bin") not in ANGULAR_BINS:
        raise ValueError("Embedded protocol has an unsupported angular bin")
    if payload.get("model_name") != expected_specification["model"]:
        raise ValueError("Embedded protocol model is incompatible")
    if payload.get("long_edge") != expected_specification["long_edge"]:
        raise ValueError("Embedded protocol long edge is incompatible")
    if payload.get("patch_size") != expected_specification["patch_size"]:
        raise ValueError("Embedded protocol patch size is incompatible")
    if model_name == DINOV2_MODEL_NAME:
        if payload.get("debias_ranks") != [0]:
            raise ValueError("Embedded DINOv2 protocol must contain only raw rank 0")
        if payload.get("basis_identities") != {}:
            raise ValueError("Embedded DINOv2 protocol must not contain debias bases")
    for key in (
        "pair_file_sha256",
        "estimation_split_sha256",
        "weights_id",
        "dataset_snapshot_id",
        "descriptor_snapshot_id",
    ):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"Embedded protocol {key} must be a non-empty string")
    if not isinstance(payload.get("basis_identities"), dict):
        raise ValueError("Embedded protocol basis_identities must be an object")


def validate_shard(
    path: Path,
    *,
    expected_fingerprint: str | None = None,
    expected_bin: str | None = None,
    expected_pair_index: int | None = None,
    expected_layer: int | None = None,
    expected_debias_ranks: Sequence[int] | None = None,
    expected_ranks: Sequence[int] | None = None,
    expected_max_k: int | None = None,
) -> dict[str, np.ndarray]:
    """Load a rank shard and reject corruption or stale protocol data."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    required = {
        "schema_version",
        "protocol_fingerprint",
        "protocol_json",
        "angular_bin",
        "pair_index",
        "layer",
        "debias_ranks",
        "max_k",
        "angle_degrees",
        "object_name",
        "image_a",
        "image_b",
        "image_a_grid_hw",
        "image_b_grid_hw",
        "image_a_resized_hw",
        "image_b_resized_hw",
        "image_a_object_patch_index",
        "image_b_object_patch_index",
        "mutual_match_count_at_k",
        *(f"{direction}_{suffix}" for direction in DIRECTIONS for suffix in _DIRECTION_SUFFIXES),
    }
    with np.load(path, allow_pickle=False) as loaded:
        missing = required.difference(loaded.files)
        if missing:
            raise ValueError(f"Raw shard is missing arrays: {sorted(missing)}")
        data = {name: np.array(loaded[name], copy=True) for name in loaded.files}

    if int(_scalar(data, "schema_version")) != RAW_SHARD_SCHEMA_VERSION:
        raise ValueError("Raw shard schema version is incompatible")
    fingerprint = str(_scalar(data, "protocol_fingerprint"))
    try:
        embedded_protocol = json.loads(str(_scalar(data, "protocol_json")))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Raw shard protocol_json is not valid JSON") from exc
    if not isinstance(embedded_protocol, dict):
        raise ValueError("Raw shard protocol_json must contain a JSON object")
    _validate_embedded_protocol(embedded_protocol)
    model_name = str(embedded_protocol["model_name"])
    patch_size = int(embedded_protocol["patch_size"])
    actual_embedded = protocol_fingerprint(embedded_protocol)
    if actual_embedded != fingerprint:
        raise ValueError(
            f"Embedded protocol hashes to {actual_embedded}, not {fingerprint}"
        )
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValueError(
            f"Protocol fingerprint is {fingerprint}, expected {expected_fingerprint}"
        )

    angular_bin = str(_scalar(data, "angular_bin"))
    pair_index = int(_scalar(data, "pair_index"))
    layer = int(_scalar(data, "layer"))
    max_k = int(_scalar(data, "max_k"))
    debias_ranks = np.asarray(data["debias_ranks"])
    _require_dtype("debias_ranks", debias_ranks, np.dtype(np.int32))
    if (
        debias_ranks.ndim != 1
        or debias_ranks.size == 0
        or np.any(debias_ranks < 0)
        or len(np.unique(debias_ranks)) != len(debias_ranks)
    ):
        raise ValueError("debias_ranks must be unique non-negative integers")
    if max_k <= 0:
        raise ValueError("max_k must be positive")
    requested_ranks = expected_debias_ranks if expected_debias_ranks is not None else expected_ranks
    for actual, expected, label in (
        (angular_bin, expected_bin, "angular bin"),
        (pair_index, expected_pair_index, "pair index"),
        (layer, expected_layer, "layer"),
        (max_k, expected_max_k, "max K"),
    ):
        if expected is not None and actual != expected:
            raise ValueError(f"Raw shard {label} is {actual!r}, expected {expected!r}")
    if requested_ranks is not None and not np.array_equal(
        debias_ranks, np.asarray(requested_ranks, dtype=np.int32)
    ):
        raise ValueError("Raw shard debias ranks do not match the request")
    if angular_bin not in ANGULAR_BINS:
        raise ValueError(f"Unsupported angular bin in shard: {angular_bin}")
    if layer not in DEFAULT_LAYERS:
        raise ValueError(f"Layer {layer} is outside the locked search range")
    allowed_ranks = (0,) if model_name == DINOV2_MODEL_NAME else ALLOWED_DEBIAS_RANKS
    if not set(int(value) for value in debias_ranks).issubset(allowed_ranks):
        raise ValueError("Raw shard contains a debias rank outside the locked search set")
    if not np.isfinite(float(_scalar(data, "angle_degrees"))):
        raise ValueError("angle_degrees must be finite")
    for key, actual in (
        ("angular_bin", angular_bin),
        ("layer", layer),
        ("max_k", max_k),
    ):
        if embedded_protocol.get(key) != actual:
            raise ValueError(f"Embedded protocol {key} disagrees with shard scalar")
    if embedded_protocol.get("debias_ranks") != debias_ranks.tolist():
        raise ValueError("Embedded protocol debias_ranks disagree with shard array")

    grids: dict[str, np.ndarray] = {}
    complete: dict[str, np.ndarray] = {}
    object_indices: dict[str, np.ndarray] = {}
    for image_label in ("a", "b"):
        grid = np.asarray(data[f"image_{image_label}_grid_hw"])
        resized = np.asarray(data[f"image_{image_label}_resized_hw"])
        _require_dtype(f"image_{image_label}_grid_hw", grid, np.dtype(np.int32))
        _require_dtype(f"image_{image_label}_resized_hw", resized, np.dtype(np.int32))
        if grid.shape != (2,) or resized.shape != (2,) or np.any(grid <= 0) or np.any(resized <= 0):
            raise ValueError("Grid and resized metadata must contain two positive values")
        if not np.array_equal(
            grid,
            (resized.astype(np.int64) + patch_size - 1) // patch_size,
        ):
            raise ValueError(f"image_{image_label} grid disagrees with resized size")
        grids[image_label] = grid
        complete[image_label] = _complete_indices(grid, resized, patch_size)
        if len(complete[image_label]) < max_k:
            raise ValueError("Every image must contain at least max_k complete patches")
        objects = np.asarray(data[f"image_{image_label}_object_patch_index"])
        _require_dtype(
            f"image_{image_label}_object_patch_index", objects, np.dtype(np.int64)
        )
        if objects.ndim != 1 or len(np.unique(objects)) != len(objects):
            raise ValueError("Object patch indices must be a unique vector")
        if np.any(~np.isin(objects, complete[image_label])):
            raise ValueError("Object patch indices must refer to complete patches")
        object_indices[image_label] = objects

    rank_count = len(debias_ranks)
    for direction, source_label, target_label in (
        ("a_to_b", "a", "b"),
        ("b_to_a", "b", "a"),
    ):
        source = np.asarray(data[f"{direction}_source_patch_index"])
        source_min = np.asarray(data[f"{direction}_source_min_object_error_m"])
        _require_dtype(f"{direction}_source_patch_index", source, np.dtype(np.int64))
        _require_dtype(
            f"{direction}_source_min_object_error_m", source_min, np.dtype(np.float64)
        )
        if source.ndim != 1 or len(np.unique(source)) != len(source):
            raise ValueError(f"{direction} source indices must be a unique vector")
        if np.any(~np.isin(source, complete[source_label])):
            raise ValueError(f"{direction} contains a source patch touching padding")
        if np.any(~np.isin(source, object_indices[source_label])):
            raise ValueError(f"{direction} contains a source patch outside the object mask")
        if source_min.shape != source.shape or np.any(np.isnan(source_min)) or np.any(source_min < 0):
            raise ValueError(f"{direction} minimum geometry errors are invalid")

        shape = (rank_count, len(source), max_k)
        target = np.asarray(data[f"{direction}_candidate_target_patch_index"])
        cosine = np.asarray(data[f"{direction}_candidate_cosine"])
        target_object = np.asarray(data[f"{direction}_candidate_target_is_object"])
        target_depth = np.asarray(data[f"{direction}_candidate_target_has_depth"])
        error = np.asarray(data[f"{direction}_candidate_error_m"])
        entry = np.asarray(data[f"{direction}_candidate_mutual_entry_k"])
        for name, array, dtype in (
            ("candidate_target_patch_index", target, np.dtype(np.int64)),
            ("candidate_cosine", cosine, np.dtype(np.float32)),
            ("candidate_target_is_object", target_object, np.dtype(np.bool_)),
            ("candidate_target_has_depth", target_depth, np.dtype(np.bool_)),
            ("candidate_error_m", error, np.dtype(np.float64)),
            ("candidate_mutual_entry_k", entry, np.dtype(np.int16)),
        ):
            _require_dtype(f"{direction}_{name}", array, dtype)
            if array.shape != shape:
                raise ValueError(f"{direction}_{name} has shape {array.shape}, expected {shape}")
        if np.any(~np.isin(target, complete[target_label])):
            raise ValueError(f"{direction} contains a target patch touching padding")
        expected_target_object = np.isin(target, object_indices[target_label])
        if not np.array_equal(target_object, expected_target_object):
            raise ValueError(
                f"{direction} target object flags disagree with stored object indices"
            )
        if np.any(np.diff(np.sort(target, axis=2), axis=2) == 0):
            raise ValueError(f"{direction} candidate targets must be unique per query")
        if not np.isfinite(cosine).all() or np.any(np.diff(cosine, axis=2) > 1e-6):
            raise ValueError(f"{direction} cosine candidates must be finite and non-increasing")
        if np.any(cosine < -1.00001) or np.any(cosine > 1.00001):
            raise ValueError(f"{direction} cosine candidates lie outside [-1, 1]")
        if np.any(np.isnan(error)) or np.any(error < 0) or np.any(np.isneginf(error)):
            raise ValueError(f"{direction} raw errors must be non-negative or +inf")
        if np.any(target_depth & ~np.isfinite(error)) or np.any(~target_depth & ~np.isposinf(error)):
            raise ValueError(f"{direction} depth flags disagree with raw error finiteness")
        if np.any(entry < 1) or np.any(entry > max_k + 1):
            raise ValueError(f"{direction} mutual-entry ranks are out of range")
        forward_rank = np.arange(1, max_k + 1, dtype=np.int16)[None, None, :]
        if np.any((entry <= max_k) & (entry < forward_rank)):
            raise ValueError(f"{direction} mutual-entry rank precedes directional rank")
        valid_object_error = target_object & target_depth
        if np.any(
            valid_object_error
            & (source_min[None, :, None] > error + 1e-12)
        ):
            raise ValueError(f"{direction} source minimum exceeds a valid candidate error")

    counts = np.asarray(data["mutual_match_count_at_k"])
    _require_dtype("mutual_match_count_at_k", counts, np.dtype(np.int64))
    if counts.shape != (rank_count, max_k) or np.any(counts < 0) or np.any(np.diff(counts, axis=1) < 0):
        raise ValueError("mutual_match_count_at_k must be non-negative and cumulative")
    complete_a, complete_b = len(complete["a"]), len(complete["b"])
    upper = np.asarray(
        [min(complete_a * k, complete_b * k) for k in range(1, max_k + 1)],
        dtype=np.int64,
    )
    if np.any(counts > upper[None, :]):
        raise ValueError("mutual match count exceeds the complete graph top-K bound")
    return data
