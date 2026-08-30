"""Validated on-disk contracts for descriptors, keypoints, and associations."""

from __future__ import annotations

import csv
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DINO_SCHEMA_VERSION = 2
SUPERPOINT_SCHEMA_VERSION = 1
DINO_PROVENANCE_KEYS = (
    "model_name",
    "layer",
    "weights_id",
    "long_edge",
    "downscale_only",
    "normalization_id",
    "resize_id",
    "padding_id",
)
DINO_MODEL_PROVENANCE_KEYS = (
    "model_family",
    "descriptor_dim",
    "register_tokens",
    "correction",
    "source_revision",
    "source_dirty",
)
DESCRIPTOR_OPTIONAL_PROVENANCE_KEYS = (
    "input_mode",
    "token_output",
    "inference_dtype",
    "upstream_model_id",
)
ASSOCIATION_COLUMNS = (
    "left_idx",
    "right_idx",
    "x1",
    "y1",
    "x2",
    "y2",
    "similarity",
    "k",
)


@dataclass(frozen=True)
class DinoDescriptorMap:
    descriptor_map: np.ndarray
    patch_size: int
    proc_hw: tuple[int, int]
    orig_hw: tuple[int, int]
    has_orig_hw: bool
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SuperPointFeatures:
    keypoints: np.ndarray
    descriptors: np.ndarray | None
    scores: np.ndarray | None
    metadata: dict[str, Any]


def _hw(data: Mapping[str, Any], key: str, fallback: tuple[int, int]) -> tuple[int, int]:
    if key not in data:
        warnings.warn(f"Legacy cache is missing `{key}`; inferred {fallback}.", UserWarning)
        return fallback
    value = np.asarray(data[key]).reshape(-1)
    if value.shape != (2,) or np.any(value <= 0):
        raise ValueError(f"`{key}` must contain two positive integers, got {value!r}")
    return int(value[0]), int(value[1])


def _scalar(data: Mapping[str, Any], key: str) -> Any:
    value = np.asarray(data[key])
    return value.reshape(()).item()


def _validate_expected_metadata(
    *,
    artifact: str,
    path: Path,
    metadata: Mapping[str, Any],
    expected_metadata: Mapping[str, Any] | None,
    allow_missing: bool = False,
) -> None:
    for key, expected in (expected_metadata or {}).items():
        if key not in metadata:
            if allow_missing:
                continue
            raise ValueError(
                f"Unverifiable {artifact} cache {path}: required metadata `{key}` is missing"
            )
        actual = metadata[key]
        if isinstance(actual, np.ndarray):
            matches = np.array_equal(actual, expected)
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"Stale {artifact} cache {path}: metadata `{key}` is {actual!r}, "
                f"expected {expected!r}"
            )


def _dino_metadata(data: Mapping[str, Any], path: Path) -> dict[str, Any]:
    version = int(_scalar(data, "schema_version")) if "schema_version" in data else 0
    if version not in (0, 1, DINO_SCHEMA_VERSION):
        raise ValueError(
            f"Unsupported DINO schema version {version} in {path}; "
            f"expected at most {DINO_SCHEMA_VERSION}"
        )
    metadata = {
        key: _scalar(data, key)
        for key in (
            "schema_version",
            *DINO_PROVENANCE_KEYS,
            *DINO_MODEL_PROVENANCE_KEYS,
            *DESCRIPTOR_OPTIONAL_PROVENANCE_KEYS,
        )
        if key in data
    }
    if version < DINO_SCHEMA_VERSION:
        warnings.warn(
            f"Legacy DINO cache {path} has no complete model/preprocessing provenance.",
            UserWarning,
        )
    return metadata


def load_dino_map(
    path: Path,
    default_patch_size: int = 16,
    expected_metadata: Mapping[str, Any] | None = None,
) -> DinoDescriptorMap:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DINO descriptor file does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        metadata = _dino_metadata(data, path)
        if "descriptor_map" not in data:
            raise KeyError(f"{path} does not contain `descriptor_map`")
        descriptor_map = np.asarray(data["descriptor_map"], dtype=np.float32)
        if descriptor_map.ndim != 3 or descriptor_map.shape[2] == 0:
            raise ValueError(
                f"`descriptor_map` must have shape [grid_h, grid_w, channels], got "
                f"{descriptor_map.shape} in {path}"
            )
        if "patch_size" in data:
            patch_size = int(np.asarray(data["patch_size"]).reshape(()))
        else:
            warnings.warn(
                f"Legacy DINO cache {path} has no `patch_size`; using {default_patch_size}.",
                UserWarning,
            )
            patch_size = default_patch_size
        if patch_size <= 0:
            raise ValueError(f"`patch_size` must be positive in {path}")
        inferred = (descriptor_map.shape[0] * patch_size, descriptor_map.shape[1] * patch_size)
        proc_hw = _hw(data, "proc_hw", inferred)
        has_orig_hw = "orig_hw" in data
        orig_hw = _hw(data, "orig_hw", proc_hw)

    metadata.update(
        {
            "patch_size": patch_size,
            "proc_hw": proc_hw,
            "orig_hw": orig_hw,
        }
    )
    _validate_expected_metadata(
        artifact="DINO", path=path, metadata=metadata, expected_metadata=expected_metadata
    )

    grid_h = proc_hw[0] // patch_size
    grid_w = proc_hw[1] // patch_size
    if grid_h > descriptor_map.shape[0] or grid_w > descriptor_map.shape[1]:
        raise ValueError(
            f"`proc_hw` {proc_hw} requires a {grid_h}x{grid_w} grid, but {path} stores "
            f"{descriptor_map.shape[:2]}"
        )
    descriptor_map = descriptor_map[:grid_h, :grid_w]
    return DinoDescriptorMap(
        descriptor_map, patch_size, proc_hw, orig_hw, has_orig_hw, metadata
    )


def save_dino_map(
    path: Path,
    descriptor_map: np.ndarray,
    patch_size: int,
    proc_hw: tuple[int, int],
    orig_hw: tuple[int, int],
    metadata: Mapping[str, Any],
) -> None:
    descriptor_map = np.asarray(descriptor_map, dtype=np.float32)
    if descriptor_map.ndim != 3 or descriptor_map.shape[2] == 0:
        raise ValueError("descriptor_map must have shape [grid_h, grid_w, channels]")
    if patch_size <= 0 or min(*proc_hw, *orig_hw) <= 0:
        raise ValueError("patch_size and image dimensions must be positive")
    missing = set(DINO_PROVENANCE_KEYS) - set(metadata)
    if missing:
        raise ValueError(f"DINO metadata is missing required keys: {sorted(missing)}")
    reserved = {"schema_version", "descriptor_map", "patch_size", "proc_hw", "orig_hw"}
    overlap = reserved & set(metadata)
    if overlap:
        raise ValueError(f"DINO metadata uses reserved keys: {sorted(overlap)}")
    payload: dict[str, Any] = {
        "schema_version": np.int32(DINO_SCHEMA_VERSION),
        "descriptor_map": descriptor_map,
        "patch_size": np.int32(patch_size),
        "proc_hw": np.asarray(proc_hw, dtype=np.int32),
        "orig_hw": np.asarray(orig_hw, dtype=np.int32),
    }
    payload.update(metadata)
    _atomic_savez(Path(path), **payload)


def load_superpoint_cache(
    path: Path,
    expected_metadata: Mapping[str, Any] | None = None,
    *,
    allow_missing_metadata: bool = False,
) -> SuperPointFeatures:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SuperPoint cache does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "keypoints" not in data:
            raise KeyError(f"{path} does not contain `keypoints`")
        keypoints = np.asarray(data["keypoints"], dtype=np.float32)
        if keypoints.ndim != 2 or keypoints.shape[1] != 2:
            raise ValueError(f"`keypoints` must have shape [N, 2], got {keypoints.shape}")
        descriptors = (
            np.asarray(data["descriptors"], dtype=np.float32) if "descriptors" in data else None
        )
        scores = np.asarray(data["scores"], dtype=np.float32) if "scores" in data else None
        if descriptors is not None and (descriptors.ndim != 2 or len(descriptors) != len(keypoints)):
            raise ValueError("SuperPoint descriptor and keypoint counts do not agree")
        if scores is not None and (scores.ndim != 1 or len(scores) != len(keypoints)):
            raise ValueError("SuperPoint score and keypoint counts do not agree")

        scalar_metadata_keys = (
            "schema_version",
            "long_edge",
            "max_num_keypoints",
            "downscale_only",
            "weights_id",
            "source_path",
        )
        metadata = {key: _scalar(data, key) for key in scalar_metadata_keys if key in data}
        for key in ("orig_hw", "proc_hw"):
            if key in data:
                value = np.asarray(data[key]).reshape(-1)
                if value.shape != (2,):
                    raise ValueError(f"SuperPoint metadata `{key}` must contain two values")
                metadata[key] = (int(value[0]), int(value[1]))
        if "schema_version" not in metadata:
            warnings.warn(
                f"Legacy SuperPoint cache {path} has no preprocessing metadata; "
                "coordinate compatibility cannot be verified.",
                UserWarning,
            )
        elif int(metadata["schema_version"]) != SUPERPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported SuperPoint schema version {metadata['schema_version']} in {path}; "
                f"expected {SUPERPOINT_SCHEMA_VERSION}"
            )

    _validate_expected_metadata(
        artifact="SuperPoint",
        path=path,
        metadata=metadata,
        expected_metadata=expected_metadata,
        allow_missing=allow_missing_metadata,
    )
    return SuperPointFeatures(keypoints, descriptors, scores, metadata)


def save_superpoint_cache(
    path: Path,
    keypoints: np.ndarray,
    *,
    descriptors: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    long_edge: int,
    max_num_keypoints: int,
    downscale_only: bool,
    weights_id: str,
    orig_hw: tuple[int, int],
    proc_hw: tuple[int, int],
    source_path: str,
) -> None:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError("keypoints must have shape [N, 2]")
    payload: dict[str, Any] = {
        "schema_version": np.int32(SUPERPOINT_SCHEMA_VERSION),
        "keypoints": keypoints,
        "long_edge": np.int32(long_edge),
        "max_num_keypoints": np.int32(max_num_keypoints),
        "downscale_only": np.bool_(downscale_only),
        "weights_id": np.str_(weights_id),
        "orig_hw": np.asarray(orig_hw, dtype=np.int32),
        "proc_hw": np.asarray(proc_hw, dtype=np.int32),
        "source_path": np.str_(source_path),
    }
    if descriptors is not None:
        descriptors = np.asarray(descriptors, dtype=np.float32)
        if descriptors.ndim != 2 or len(descriptors) != len(keypoints):
            raise ValueError("descriptors must have shape [N, D]")
        payload["descriptors"] = descriptors
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float32)
        if scores.ndim != 1 or len(scores) != len(keypoints):
            raise ValueError("scores must have shape [N]")
        payload["scores"] = scores
    _atomic_savez(Path(path), **payload)


def _atomic_savez(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temp_path = Path(handle.name)
        np.savez(handle, **payload)
    try:
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def validate_association_csv(path: Path) -> int:
    """Validate the public association schema and return its row count."""
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ASSOCIATION_COLUMNS:
            raise ValueError(
                f"Association header in {path} must be {ASSOCIATION_COLUMNS}, "
                f"got {tuple(reader.fieldnames or ())}"
            )
        seen: set[tuple[int, int]] = set()
        count = 0
        for line_no, row in enumerate(reader, 2):
            try:
                pair = (int(row["left_idx"]), int(row["right_idx"]))
                coords = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
                float(row["similarity"])
                first_k = int(row["k"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Malformed association row {line_no} in {path}") from exc
            if min(*pair) < 0 or first_k <= 0 or not np.all(np.isfinite(coords)):
                raise ValueError(f"Invalid association values on row {line_no} in {path}")
            if pair in seen:
                raise ValueError(f"Duplicate association {pair} on row {line_no} in {path}")
            seen.add(pair)
            count += 1
    return count
