"""Generate dense geometric rank-CDF shards on NAVI patch grids."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import zipfile
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from dino_m2m.dino import (
    DINO_NORMALIZATION_ID,
    backbone_provenance,
    dino_cache_validation_metadata,
    dino_artifact_metadata,
    required_basis_sizes,
)
from dino_m2m.matching import (
    basis_for_dim,
    load_basis_payload,
    require_torch,
    resolve_device,
)
from dino_m2m.pairs import read_pairs
from dino_m2m.provenance import checkpoint_identity
from dino_m2m.resize import resized_hw_long_edge
from dino_m2m.schemas import DinoDescriptorMap, load_dino_map
from evaluation.json_utils import strict_json_dumps

from .geometry import (
    PatchGeometry,
    build_patch_geometry,
    candidate_correspondence_errors,
    minimum_object_correspondence_errors,
)
from .protocol import (
    ALLOWED_DEBIAS_RANKS,
    ANGULAR_BINS,
    DINOV2_MODEL_NAME,
    DINOV3_MODEL_NAME,
    DEFAULT_DEBIAS_RANKS,
    DEFAULT_LAYERS,
    DEFAULT_MAX_K,
    RAW_SHARD_SCHEMA_VERSION,
    atomic_save_shard,
    canonical_json,
    correction_mode_for_model,
    make_protocol_payload,
    patch_size_for_model,
    protocol_fingerprint,
    protocol_specification_for_model,
    sha256_file,
    shard_path,
    validate_shard,
)
from .ranking import BidirectionalTopK, compute_bidirectional_topk_from_similarity


@dataclass(frozen=True)
class DensePair:
    pair_index: int
    image_a: Path
    image_b: Path
    angle_degrees: float
    object_name: str
    image_a_hw: tuple[int, int]
    image_b_hw: tuple[int, int]


@dataclass(frozen=True)
class DenseEvaluationOptions:
    angular_bin: str
    pair_file: Path
    estimation_pair_file: Path
    image_root: Path
    dino_root: Path
    basis_root: Path | None
    output_root: Path
    weights: Path
    model_name: str = DINOV3_MODEL_NAME
    layers: tuple[int, ...] = DEFAULT_LAYERS
    ranks: tuple[int, ...] | None = None
    max_k: int = DEFAULT_MAX_K
    long_edge: int = 1024
    patch_size: int | None = None
    basis_filename_template: str = "dinov3_vitl16_{height}x{width}_basis.pt"
    device: str = "auto"
    descriptor_cache_images: int = 8
    geometry_chunk_size: int = 512
    max_pairs: int | None = None
    existing: str = "resume"
    summarize: bool = True


@dataclass(frozen=True)
class DenseEvaluationSummary:
    pair_count: int
    layer_count: int
    written_shards: int
    resumed_shards: int
    recomputed_shards: int
    bin_root: Path


@dataclass(frozen=True)
class _LoadedImage:
    dino: DinoDescriptorMap
    descriptors: Any


class NaviAnnotationIndex:
    """Lazy, validated NAVI annotation and patch-geometry reader."""

    def __init__(self, image_root: Path, patch_size: int, long_edge: int) -> None:
        self.image_root = Path(image_root).resolve()
        self.patch_size = patch_size
        self.long_edge = long_edge
        self._annotations: dict[Path, dict[str, dict[str, Any]]] = {}
        self._geometry: dict[
            tuple[Path, tuple[int, int], tuple[int, int]], PatchGeometry
        ] = {}

    def _resolve_relative(self, relative: Path) -> Path:
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"NAVI image path must stay relative to image_root: {relative}")
        path = (self.image_root / relative).resolve()
        try:
            path.relative_to(self.image_root)
        except ValueError as exc:
            raise ValueError(f"NAVI image path escapes image_root: {relative}") from exc
        return path

    def annotation(self, image_relative: Path) -> dict[str, Any]:
        image_path = self._resolve_relative(image_relative)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        scene_root = image_path.parent.parent
        annotation_path = scene_root / "annotations.json"
        if annotation_path not in self._annotations:
            rows = json.loads(annotation_path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError(f"Expected an annotation list: {annotation_path}")
            indexed = {
                str(row["filename"]): row
                for row in rows
                if isinstance(row, dict) and "filename" in row
            }
            if len(indexed) != len(rows):
                raise ValueError(f"Invalid or duplicate annotation filenames: {annotation_path}")
            self._annotations[annotation_path] = indexed
        try:
            return self._annotations[annotation_path][image_path.name]
        except KeyError as exc:
            raise KeyError(f"No NAVI annotation for {image_relative}") from exc

    def geometry(self, image_relative: Path, dino: DinoDescriptorMap) -> PatchGeometry:
        grid_hw = dino.descriptor_map.shape[:2]
        key = (image_relative, tuple(grid_hw), tuple(dino.orig_hw))
        if key in self._geometry:
            return self._geometry[key]
        image_path = self._resolve_relative(image_relative)
        scene_root = image_path.parent.parent
        depth_path = scene_root / "depth" / f"{image_path.stem}.png"
        mask_path = scene_root / "masks" / f"{image_path.stem}.png"
        if not depth_path.is_file() or not mask_path.is_file():
            missing = depth_path if not depth_path.is_file() else mask_path
            raise FileNotFoundError(missing)
        with Image.open(depth_path) as opened:
            raw_depth = np.asarray(opened).copy()
        with Image.open(mask_path) as opened:
            object_mask = np.asarray(opened).copy()
        annotation = self.annotation(image_relative)
        try:
            original_height, original_width = (
                int(value) for value in annotation["image_size"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid NAVI image_size for {image_relative}") from exc
        expected_height, expected_width, _ = resized_hw_long_edge(
            original_height, original_width, self.long_edge, downscale_only=False
        )
        if dino.orig_hw != (expected_height, expected_width):
            raise ValueError(
                f"DINO unpadded size for {image_relative} is {dino.orig_hw}, expected "
                f"{(expected_height, expected_width)}"
            )
        expected_padded = (
            (expected_height + self.patch_size - 1) // self.patch_size * self.patch_size,
            (expected_width + self.patch_size - 1) // self.patch_size * self.patch_size,
        )
        if dino.proc_hw != expected_padded:
            raise ValueError(
                f"DINO padded size for {image_relative} is {dino.proc_hw}, expected "
                f"{expected_padded}"
            )
        geometry = build_patch_geometry(
            raw_depth=raw_depth,
            object_mask=object_mask,
            annotation=annotation,
            grid_hw=tuple(int(value) for value in grid_hw),
            resized_hw=dino.orig_hw,
            patch_size=self.patch_size,
        )
        self._geometry[key] = geometry
        return geometry


class _DescriptorCache:
    def __init__(
        self,
        *,
        dino_root: Path,
        layer: int,
        expected_metadata: Mapping[str, Any],
        device: Any,
        capacity: int,
    ) -> None:
        self.root = Path(dino_root) / f"layer{layer}"
        self.expected_metadata = dict(expected_metadata)
        self.device = device
        self.capacity = capacity
        self._cache: OrderedDict[Path, _LoadedImage] = OrderedDict()

    def get(self, image_relative: Path) -> _LoadedImage:
        if image_relative in self._cache:
            value = self._cache.pop(image_relative)
            self._cache[image_relative] = value
            return value
        path = (self.root / image_relative).with_suffix(".dino.npz")
        dino = load_dino_map(path, expected_metadata=self.expected_metadata)
        expected_descriptor_dim = self.expected_metadata.get("descriptor_dim")
        if (
            expected_descriptor_dim is not None
            and dino.descriptor_map.shape[2] != int(expected_descriptor_dim)
        ):
            raise ValueError(
                f"DINO descriptor dimension in {path} is "
                f"{dino.descriptor_map.shape[2]}, expected {expected_descriptor_dim}"
            )
        torch, functional = require_torch()
        flat = np.ascontiguousarray(
            dino.descriptor_map.reshape(-1, dino.descriptor_map.shape[2])
        )
        descriptors = torch.from_numpy(flat).to(device=self.device, dtype=torch.float32)
        descriptors = functional.normalize(descriptors, p=2, dim=1)
        value = _LoadedImage(dino=dino, descriptors=descriptors)
        self._cache[image_relative] = value
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return value


class _LayerBasisCache:
    def __init__(
        self,
        *,
        basis_root: Path,
        layer: int,
        filename_template: str,
        model_name: str,
        weights_id: str,
        patch_size: int,
        long_edge: int,
        device: Any,
        allowed_sizes: Sequence[tuple[int, int]],
    ) -> None:
        self.root = Path(basis_root) / f"layer{layer}"
        self.layer = layer
        self.template = filename_template
        self.model_name = model_name
        self.weights_id = weights_id
        self.patch_size = patch_size
        self.long_edge = long_edge
        self.device = device
        self.allowed_sizes = frozenset(tuple(size) for size in allowed_sizes)
        self._payloads: dict[tuple[int, int], dict[str, Any]] = {}

    def path_for_size(self, size: tuple[int, int]) -> Path:
        height, width = size
        try:
            relative = Path(self.template.format(height=height, width=width))
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "basis filename template may use only {height} and {width}"
            ) from exc
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("basis filename template must stay below basis_root/layerN")
        return self.root / relative

    def get(self, size: tuple[int, int]) -> dict[str, Any]:
        if size not in self.allowed_sizes:
            raise ValueError(
                f"Runtime descriptor size {size} was absent from the pre-hashed pair CSV sizes"
            )
        if size not in self._payloads:
            height, width = size
            expected = {
                "model_name": self.model_name,
                "layer": self.layer,
                "weights_id": self.weights_id,
                "normalization_id": DINO_NORMALIZATION_ID,
                "patch_size": self.patch_size,
                "image_height": height,
                "image_width": width,
                "long_edge": self.long_edge,
                "downscale_only": False,
            }
            self._payloads[size] = load_basis_payload(
                self.path_for_size(size), self.device, expected_metadata=expected
            )
        return self._payloads[size]


def _read_dense_pairs(
    pair_file: Path, angular_bin: str, max_pairs: int | None
) -> list[DensePair]:
    records = read_pairs(pair_file)
    with Path(pair_file).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "image_1",
            "image_2",
            "angular_distance_degrees",
            "image_1_height",
            "image_1_width",
            "image_2_height",
            "image_2_width",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{pair_file} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if len(rows) != len(records):
        raise ValueError("Dense pair metadata rows disagree with the canonical parser")
    metadata_by_pair: dict[int, Mapping[str, str]] = {}
    for automatic_index, row in enumerate(rows, 1):
        explicit = (row.get("pair_index") or "").strip()
        pair_index = int(explicit) if explicit else automatic_index
        if pair_index in metadata_by_pair:
            raise ValueError(f"Duplicate pair index {pair_index} in {pair_file}")
        metadata_by_pair[pair_index] = row
    lower, upper = (float(value) for value in angular_bin.split("-"))
    result: list[DensePair] = []
    for record in records:
        row = metadata_by_pair.get(record.pair_index)
        if row is None:
            raise ValueError(f"Missing metadata for pair index {record.pair_index}")
        if Path(row["image_1"]) != record.left_rel or Path(row["image_2"]) != record.right_rel:
            raise ValueError(f"Pair metadata/path mismatch for pair {record.pair_index}")
        angle = float(row["angular_distance_degrees"])
        if not np.isfinite(angle) or not lower <= angle < upper:
            raise ValueError(
                f"Pair {record.pair_index} angle {angle} is outside bin {angular_bin}"
            )
        left_object = record.left_rel.parts[0] if record.left_rel.parts else ""
        right_object = record.right_rel.parts[0] if record.right_rel.parts else ""
        if not left_object or left_object != right_object:
            raise ValueError(f"Pair {record.pair_index} crosses NAVI objects")
        result.append(
            DensePair(
                pair_index=record.pair_index,
                image_a=record.left_rel,
                image_b=record.right_rel,
                angle_degrees=angle,
                object_name=left_object,
                image_a_hw=(int(row["image_1_height"]), int(row["image_1_width"])),
                image_b_hw=(int(row["image_2_height"]), int(row["image_2_width"])),
            )
        )
    if max_pairs is not None:
        if max_pairs < 0:
            raise ValueError("max_pairs must be non-negative or None")
        result = result[:max_pairs]
    return result


def audit_split_disjointness(
    correspondence_pair_file: Path, estimation_pair_file: Path, angular_bin: str
) -> dict[str, Any]:
    correspondence = _read_dense_pairs(correspondence_pair_file, angular_bin, None)
    estimation_files = _estimation_split_files(estimation_pair_file)
    estimation_by_bin: dict[str, list[DensePair]] = {}
    for path in estimation_files:
        matching_bins = [value for value in ANGULAR_BINS if value in path.stem]
        if len(matching_bins) != 1:
            if len(estimation_files) != 1:
                raise ValueError(f"Cannot infer angular bin from estimation split: {path}")
            split_bin = angular_bin
        else:
            split_bin = matching_bins[0]
        estimation_by_bin[split_bin] = _read_dense_pairs(path, split_bin, None)
    estimation = [pair for pairs in estimation_by_bin.values() for pair in pairs]
    correspondence_images = {
        str(image)
        for record in correspondence
        for image in (record.image_a, record.image_b)
    }
    estimation_images = {
        str(image)
        for record in estimation
        for image in (record.image_a, record.image_b)
    }
    overlap = sorted(correspondence_images.intersection(estimation_images))
    report = {
        "angular_bin": angular_bin,
        "correspondence_pair_file": str(Path(correspondence_pair_file).resolve()),
        "estimation_pair_files": [str(path.resolve()) for path in estimation_files],
        "correspondence_pairs": len(correspondence),
        "estimation_pairs": len(estimation),
        "estimation_pairs_by_bin": {
            key: len(estimation_by_bin[key]) for key in sorted(estimation_by_bin)
        },
        "correspondence_images": len(correspondence_images),
        "estimation_images": len(estimation_images),
        "overlap_count": len(overlap),
        "overlap_images": overlap,
        "passed": not overlap,
    }
    if overlap:
        raise ValueError(
            f"Split audit failed for {angular_bin}: {len(overlap)} correspondence images "
            "also occur anywhere in the estimation split"
        )
    return report


def _estimation_split_files(primary: Path) -> tuple[Path, ...]:
    """Resolve the complete estimation image set for leakage auditing.

    Passing one canonical ``pairs_wildset_<bin>.csv`` file opts into all three
    sibling bins. A non-canonical custom file remains a single-file test set.
    """
    primary = Path(primary)
    canonical = tuple(
        primary.parent / f"pairs_wildset_{angular_bin}.csv"
        for angular_bin in ANGULAR_BINS
    )
    if primary.name not in {path.name for path in canonical}:
        return (primary,)
    missing = [path for path in canonical if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Global estimation/correspondence isolation requires all canonical "
            f"estimation bins; missing {missing[0]}"
        )
    return canonical


def _estimation_split_identity(primary: Path) -> str:
    payload = {
        path.name: sha256_file(path) for path in _estimation_split_files(primary)
    }
    return protocol_fingerprint({"estimation_split_files": payload})


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


def _stat_snapshot_id(root: Path, relative_paths: Sequence[Path]) -> str:
    """Hash relative path, size, and mtime for a cheap stale-input guard."""
    root = Path(root).resolve()
    records: list[dict[str, Any]] = []
    for relative in sorted(set(Path(path) for path in relative_paths), key=str):
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Snapshot path must be relative to {root}: {relative}")
        path = (root / relative).resolve()
        try:
            normalized = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Snapshot path escapes {root}: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        records.append(
            {
                "path": normalized.as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return protocol_fingerprint({"stat_snapshot_v1": records})


def _dataset_snapshot_id(image_root: Path, pairs: Sequence[DensePair]) -> str:
    paths: set[Path] = set()
    for pair in pairs:
        for image in (pair.image_a, pair.image_b):
            scene = image.parent.parent
            paths.add(image)
            paths.add(scene / "depth" / f"{image.stem}.png")
            paths.add(scene / "masks" / f"{image.stem}.png")
            paths.add(scene / "annotations.json")
    return _stat_snapshot_id(image_root, tuple(paths))


def _descriptor_snapshot_id(
    dino_root: Path,
    layer: int,
    pairs: Sequence[DensePair],
    *,
    bind_extraction_manifest: bool = False,
) -> str:
    images = {
        image
        for pair in pairs
        for image in (pair.image_a, pair.image_b)
    }
    paths = tuple(
        Path(f"layer{layer}") / image.with_suffix(".dino.npz") for image in images
    )
    descriptor_stat_snapshot_id = _stat_snapshot_id(dino_root, paths)
    if not bind_extraction_manifest:
        return descriptor_stat_snapshot_id
    extraction_manifest = (
        Path(dino_root) / f"layer{layer}" / "extraction_manifest.json"
    )
    return protocol_fingerprint(
        {
            "descriptor_stat_snapshot_id": descriptor_stat_snapshot_id,
            "extraction_manifest_sha256": sha256_file(extraction_manifest),
        }
    )


def _descriptor_extraction_provenance(
    dino_root: Path,
    layers: Sequence[int],
    *,
    model_name: str,
    weights_id: str,
    patch_size: int,
) -> dict[str, Any]:
    """Audit uniform descriptor provenance without guessing for legacy caches."""

    per_layer: dict[str, dict[str, Any]] = {}
    observed_revisions: set[str] = set()
    observed_dirty: set[bool | str] = set()
    expected_profile = backbone_provenance(
        model_name,
        correction="none",
    )
    for layer in layers:
        path = Path(dino_root) / f"layer{layer}" / "extraction_manifest.json"
        if not path.is_file():
            if model_name == DINOV2_MODEL_NAME:
                raise FileNotFoundError(
                    "DINOv2 dense evaluation requires the extraction manifest "
                    f"written by extract-dino: {path}"
                )
            per_layer[f"layer{layer}"] = {
                "manifest_path": None,
                "manifest_sha256": None,
                "source_revision": "unknown",
                "source_dirty": "unknown",
            }
            observed_revisions.add("unknown")
            observed_dirty.add("unknown")
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid descriptor extraction manifest: {path}") from exc
        if not isinstance(manifest, dict):
            raise ValueError(f"Descriptor extraction manifest must be an object: {path}")
        for key, expected in (
            ("model_name", model_name),
            ("layer", int(layer)),
            ("weights_id", weights_id),
            ("patch_size", patch_size),
        ):
            if manifest.get(key) != expected:
                raise ValueError(
                    f"Descriptor extraction manifest {path} has {key}="
                    f"{manifest.get(key)!r}, expected {expected!r}"
                )
        profile = manifest.get("backbone_provenance")
        if profile is None:
            profile_keys = (
                "model_family",
                "patch_size",
                "descriptor_dim",
                "register_tokens",
                "correction",
            )
            profile = {key: manifest.get(key) for key in profile_keys}
        if profile != expected_profile:
            raise ValueError(f"Descriptor backbone provenance is incompatible: {path}")
        checkout = manifest.get("source_checkout_provenance", {})
        if not isinstance(checkout, dict):
            raise ValueError(f"Descriptor source checkout provenance is invalid: {path}")
        revision = checkout.get("source_revision", manifest.get("source_revision", "unknown"))
        dirty = checkout.get("source_dirty", manifest.get("source_dirty", "unknown"))
        if not isinstance(revision, str) or not revision:
            raise ValueError(f"Descriptor source_revision is invalid: {path}")
        if dirty != "unknown" and not isinstance(dirty, bool):
            raise ValueError(f"Descriptor source_dirty is invalid: {path}")
        observed_revisions.add(revision)
        observed_dirty.add(dirty)
        per_layer[f"layer{layer}"] = {
            "manifest_path": str(path.resolve()),
            "manifest_sha256": sha256_file(path),
            "source_revision": revision,
            "source_dirty": dirty,
        }
    if len(observed_revisions) != 1:
        raise ValueError(
            "Dense descriptor layers do not share one source_revision: "
            + ", ".join(sorted(observed_revisions))
        )
    if len(observed_dirty) != 1:
        raise ValueError("Dense descriptor layers disagree on source_dirty")
    return {
        "backbone": expected_profile,
        "source_revision": next(iter(observed_revisions)),
        "source_dirty": next(iter(observed_dirty)),
        "layer_manifests": per_layer,
    }


def _resume_may_repair(path: Path, expected_fingerprint: str) -> bool:
    """Allow repair only for unreadable or same-protocol corrupt current shards."""
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if "schema_version" not in loaded.files:
                raise RuntimeError(
                    f"Refusing to overwrite a shard without schema metadata during "
                    f"resume: {path}. Use a new output root or --existing overwrite "
                    "explicitly."
                )
            version = int(np.asarray(loaded["schema_version"]).reshape(()).item())
            if version != RAW_SHARD_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Refusing to overwrite schema-v{version} shard during resume: "
                    f"{path}. Use a new output root or --existing overwrite explicitly."
                )
            if "protocol_fingerprint" not in loaded.files:
                raise RuntimeError(
                    f"Refusing to overwrite a schema-v{version} shard without a protocol "
                    f"fingerprint during resume: {path}. Use --existing overwrite explicitly."
                )
            fingerprint = str(np.asarray(loaded["protocol_fingerprint"]).reshape(()).item())
    except RuntimeError:
        raise
    except (OSError, EOFError, ValueError, zipfile.BadZipFile):
        return True
    if fingerprint != expected_fingerprint:
        raise RuntimeError(
            f"Refusing to overwrite a stale-protocol shard during resume: {path}. "
            "Use a new output root or --existing overwrite explicitly."
        )
    return True


def _complete_descriptor_sequence(
    loaded: _LoadedImage,
    geometry: PatchGeometry,
    ranks: tuple[int, ...],
    basis_cache: _LayerBasisCache | None,
) -> Iterator[Any]:
    """Build rank-prefix residuals incrementally while retaining rank zero."""
    torch, functional = require_torch()
    indices = torch.as_tensor(
        geometry.complete_indices, device=loaded.descriptors.device, dtype=torch.long
    )
    descriptors = loaded.descriptors.index_select(0, indices)
    positive_ranks = tuple(rank for rank in ranks if rank > 0)
    basis = coefficients = None
    if positive_ranks:
        if basis_cache is None:
            raise ValueError("A basis cache is required for positive debias ranks")
        basis = basis_for_dim(
            basis_cache.get(loaded.dino.proc_hw),
            max(positive_ranks),
            int(descriptors.shape[1]),
        )
        coefficients = descriptors @ basis
    residual = descriptors
    previous_rank = 0
    for rank in ranks:
        if rank > previous_rank:
            assert basis is not None and coefficients is not None
            residual = residual - (
                coefficients[:, previous_rank:rank] @ basis[:, previous_rank:rank].T
            )
        yield functional.normalize(residual, p=2, dim=1)
        previous_rank = rank


def _empty_direction_arrays(
    debias_rank_count: int, source_count: int, max_k: int
) -> dict[str, np.ndarray]:
    shape = (debias_rank_count, source_count, max_k)
    return {
        "candidate_target_patch_index": np.empty(shape, dtype=np.int64),
        "candidate_cosine": np.empty(shape, dtype=np.float32),
        "candidate_target_is_object": np.empty(shape, dtype=np.bool_),
        "candidate_target_has_depth": np.empty(shape, dtype=np.bool_),
        "candidate_error_m": np.empty(shape, dtype=np.float64),
        "candidate_mutual_entry_k": np.empty(shape, dtype=np.int16),
    }


def _store_direction_candidates(
    *,
    output: dict[str, np.ndarray],
    debias_row: int,
    topk_positions: np.ndarray,
    topk_cosine: np.ndarray,
    topk_mutual_entry: np.ndarray,
    source_geometry: PatchGeometry,
    destination_geometry: PatchGeometry,
    source_camera: Mapping[str, Any],
    destination_camera: Mapping[str, Any],
) -> None:
    source_indices = source_geometry.valid_object_indices
    source_lookup = np.full(
        source_geometry.grid_hw[0] * source_geometry.grid_hw[1], -1, dtype=np.int64
    )
    source_lookup[source_geometry.complete_indices] = np.arange(
        len(source_geometry.complete_indices), dtype=np.int64
    )
    source_positions = source_lookup[source_indices]
    if np.any(source_positions < 0):
        raise AssertionError("A valid source-object query is absent from its search pool")
    selected_positions = topk_positions[source_positions]
    target_indices = destination_geometry.complete_indices[selected_positions]
    repeated_source = np.repeat(source_indices, target_indices.shape[1])
    target_object, target_depth, errors = candidate_correspondence_errors(
        source=source_geometry,
        destination=destination_geometry,
        source_indices=repeated_source,
        destination_indices=target_indices.reshape(-1),
        source_camera=source_camera,
        destination_camera=destination_camera,
    )
    shape = target_indices.shape
    output["candidate_target_patch_index"][debias_row] = target_indices
    output["candidate_cosine"][debias_row] = topk_cosine[source_positions]
    output["candidate_target_is_object"][debias_row] = target_object.reshape(shape)
    output["candidate_target_has_depth"][debias_row] = target_depth.reshape(shape)
    output["candidate_error_m"][debias_row] = errors.reshape(shape)
    output["candidate_mutual_entry_k"][debias_row] = topk_mutual_entry[
        source_positions
    ]


def _evaluate_pair_layer(
    *,
    pair: DensePair,
    layer: int,
    ranks: tuple[int, ...],
    max_k: int,
    protocol_payload: Mapping[str, Any],
    fingerprint: str,
    descriptor_cache: _DescriptorCache,
    annotation_index: NaviAnnotationIndex,
    basis_cache: _LayerBasisCache | None,
    geometry_chunk_size: int,
) -> dict[str, Any]:
    loaded_a = descriptor_cache.get(pair.image_a)
    loaded_b = descriptor_cache.get(pair.image_b)
    if loaded_a.dino.descriptor_map.shape[2] != loaded_b.dino.descriptor_map.shape[2]:
        raise ValueError("Pair descriptor channel dimensions disagree")
    geometry_a = annotation_index.geometry(pair.image_a, loaded_a.dino)
    geometry_b = annotation_index.geometry(pair.image_b, loaded_b.dino)
    annotation_a = annotation_index.annotation(pair.image_a)
    annotation_b = annotation_index.annotation(pair.image_b)
    for label, annotation, expected_hw in (
        ("image_a", annotation_a, pair.image_a_hw),
        ("image_b", annotation_b, pair.image_b_hw),
    ):
        actual_hw = tuple(int(value) for value in annotation["image_size"])
        if actual_hw != expected_hw:
            raise ValueError(
                f"Pair CSV {label} size {expected_hw} disagrees with NAVI annotation "
                f"{actual_hw} for pair {pair.pair_index}"
            )

    direction_meta = {
        "a_to_b": (geometry_a, geometry_b, annotation_a["camera"], annotation_b["camera"]),
        "b_to_a": (geometry_b, geometry_a, annotation_b["camera"], annotation_a["camera"]),
    }
    arrays: dict[str, dict[str, np.ndarray]] = {}
    source_min_errors: dict[str, np.ndarray] = {}
    for direction, (
        source_geometry,
        destination_geometry,
        source_camera,
        destination_camera,
    ) in direction_meta.items():
        arrays[direction] = _empty_direction_arrays(
            len(ranks), len(source_geometry.valid_object_indices), max_k
        )
        source_min_errors[direction] = minimum_object_correspondence_errors(
            source=source_geometry,
            destination=destination_geometry,
            source_indices=source_geometry.valid_object_indices,
            source_camera=source_camera,
            destination_camera=destination_camera,
            chunk_size=geometry_chunk_size,
        )

    descriptors_a_by_rank = _complete_descriptor_sequence(
        loaded_a, geometry_a, ranks, basis_cache
    )
    descriptors_b_by_rank = _complete_descriptor_sequence(
        loaded_b, geometry_b, ranks, basis_cache
    )
    mutual_counts = np.empty((len(ranks), max_k), dtype=np.int64)
    for debias_row, (complete_a, complete_b) in enumerate(
        zip(descriptors_a_by_rank, descriptors_b_by_rank)
    ):
        torch, _ = require_torch()
        with torch.inference_mode():
            similarity = complete_a @ complete_b.T
        result: BidirectionalTopK = compute_bidirectional_topk_from_similarity(
            similarity, max_k
        )
        mutual_counts[debias_row] = result.mutual_match_count_at_k
        _store_direction_candidates(
            output=arrays["a_to_b"],
            debias_row=debias_row,
            topk_positions=result.a_to_b_indices,
            topk_cosine=result.a_to_b_cosine,
            topk_mutual_entry=result.a_to_b_mutual_entry_k,
            source_geometry=geometry_a,
            destination_geometry=geometry_b,
            source_camera=annotation_a["camera"],
            destination_camera=annotation_b["camera"],
        )
        _store_direction_candidates(
            output=arrays["b_to_a"],
            debias_row=debias_row,
            topk_positions=result.b_to_a_indices,
            topk_cosine=result.b_to_a_cosine,
            topk_mutual_entry=result.b_to_a_mutual_entry_k,
            source_geometry=geometry_b,
            destination_geometry=geometry_a,
            source_camera=annotation_b["camera"],
            destination_camera=annotation_a["camera"],
        )

    payload: dict[str, Any] = {
        "schema_version": np.int32(RAW_SHARD_SCHEMA_VERSION),
        "protocol_fingerprint": np.str_(fingerprint),
        "protocol_json": np.str_(canonical_json(protocol_payload)),
        "angular_bin": np.str_(protocol_payload["angular_bin"]),
        "pair_index": np.int64(pair.pair_index),
        "layer": np.int32(layer),
        "debias_ranks": np.asarray(ranks, dtype=np.int32),
        "max_k": np.int32(max_k),
        "angle_degrees": np.float64(pair.angle_degrees),
        "object_name": np.str_(pair.object_name),
        "image_a": np.str_(str(pair.image_a)),
        "image_b": np.str_(str(pair.image_b)),
        "image_a_grid_hw": np.asarray(geometry_a.grid_hw, dtype=np.int32),
        "image_b_grid_hw": np.asarray(geometry_b.grid_hw, dtype=np.int32),
        "image_a_resized_hw": np.asarray(geometry_a.resized_hw, dtype=np.int32),
        "image_b_resized_hw": np.asarray(geometry_b.resized_hw, dtype=np.int32),
        "image_a_object_patch_index": np.asarray(
            geometry_a.object_indices, dtype=np.int64
        ),
        "image_b_object_patch_index": np.asarray(
            geometry_b.object_indices, dtype=np.int64
        ),
        "mutual_match_count_at_k": mutual_counts,
    }
    for direction, (source_geometry, _, _, _) in direction_meta.items():
        payload[f"{direction}_source_patch_index"] = np.asarray(
            source_geometry.valid_object_indices, dtype=np.int64
        )
        payload[f"{direction}_source_min_object_error_m"] = source_min_errors[
            direction
        ]
        for suffix, values in arrays[direction].items():
            payload[f"{direction}_{suffix}"] = values
    return payload


def _validate_options(
    options: DenseEvaluationOptions,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    if options.angular_bin not in ANGULAR_BINS:
        raise ValueError(f"angular_bin must be one of {ANGULAR_BINS}")
    layers = tuple(sorted(set(int(layer) for layer in options.layers)))
    requested_ranks = options.ranks
    if requested_ranks is None:
        requested_ranks = (
            (0,) if options.model_name == DINOV2_MODEL_NAME else DEFAULT_DEBIAS_RANKS
        )
    ranks = tuple(sorted(set(int(rank) for rank in requested_ranks)))
    if not layers or not set(layers).issubset(DEFAULT_LAYERS):
        raise ValueError(f"Dense-evaluation layers must be a subset of {DEFAULT_LAYERS}")
    if not ranks or not set(ranks).issubset(ALLOWED_DEBIAS_RANKS):
        raise ValueError(
            f"Dense-evaluation debias ranks must be a subset of {ALLOWED_DEBIAS_RANKS}"
        )
    protocol_specification_for_model(options.model_name)
    expected_patch_size = patch_size_for_model(options.model_name)
    patch_size = (
        expected_patch_size if options.patch_size is None else int(options.patch_size)
    )
    if options.long_edge != 1024 or patch_size != expected_patch_size:
        raise ValueError(
            f"The locked {options.model_name} dense protocol requires long_edge=1024 "
            f"and patch_size={expected_patch_size}"
        )
    if options.model_name == DINOV2_MODEL_NAME:
        if ranks != (0,):
            raise ValueError("DINOv2 dense evaluation is raw-only and requires --rank 0")
        if options.basis_root is not None:
            raise ValueError("DINOv2 dense evaluation uses no positional-bias basis")
    if options.max_k <= 0:
        raise ValueError("max_k must be positive")
    if any(rank > 0 for rank in ranks) and options.basis_root is None:
        raise ValueError("basis_root is required when any debias rank is positive")
    if options.descriptor_cache_images <= 0 or options.geometry_chunk_size <= 0:
        raise ValueError("Cache capacity and geometry chunk size must be positive")
    if options.existing not in {"resume", "overwrite", "error"}:
        raise ValueError("existing must be resume, overwrite, or error")
    for label, path, directory in (
        ("pair_file", options.pair_file, False),
        ("estimation_pair_file", options.estimation_pair_file, False),
        ("image_root", options.image_root, True),
        ("dino_root", options.dino_root, True),
        ("weights", options.weights, False),
    ):
        exists = Path(path).is_dir() if directory else Path(path).is_file()
        if not exists:
            raise FileNotFoundError(f"{label} does not exist: {path}")
    return layers, ranks, patch_size


def evaluate_dense_correspondence(
    options: DenseEvaluationOptions,
) -> DenseEvaluationSummary:
    """Evaluate one angular bin and atomically write one shard per pair/layer."""
    layers, ranks, patch_size = _validate_options(options)
    all_pairs = _read_dense_pairs(options.pair_file, options.angular_bin, None)
    pairs = all_pairs
    if options.max_pairs is not None:
        pairs = all_pairs[: options.max_pairs]
    if not pairs:
        raise RuntimeError("No pairs selected for dense correspondence evaluation")
    split_audit = audit_split_disjointness(
        options.pair_file, options.estimation_pair_file, options.angular_bin
    )
    bin_root = Path(options.output_root) / f"bin_{options.angular_bin}"
    expected_shards = {
        shard_path(options.output_root, options.angular_bin, layer, pair.pair_index)
        for layer in layers
        for pair in pairs
    }
    shard_root = bin_root / "shards"
    existing_shards = set(shard_root.glob("layer*/pair_*.npz")) if shard_root.is_dir() else set()
    unexpected_shards = sorted(existing_shards.difference(expected_shards))
    if unexpected_shards:
        raise RuntimeError(
            f"Output tree contains {len(unexpected_shards)} shards outside this run's "
            "layer/pair rectangle. Use a fresh output root; first unexpected shard: "
            f"{unexpected_shards[0]}"
        )
    _atomic_write_json(bin_root / "split_audit.json", split_audit)

    weights_id = checkpoint_identity(options.weights)
    pair_file_id = sha256_file(options.pair_file)
    estimation_file_id = _estimation_split_identity(options.estimation_pair_file)
    required_sizes = (
        required_basis_sizes(
            (options.pair_file,), long_edge=options.long_edge, downscale_only=False
        )
        if any(rank > 0 for rank in ranks)
        else ()
    )
    device = resolve_device(options.device)
    annotation_index = NaviAnnotationIndex(
        options.image_root, patch_size, options.long_edge
    )
    dataset_snapshot_id = _dataset_snapshot_id(options.image_root, all_pairs)
    descriptor_provenance = _descriptor_extraction_provenance(
        options.dino_root,
        layers,
        model_name=options.model_name,
        weights_id=weights_id,
        patch_size=patch_size,
    )
    protocol_by_layer: dict[int, dict[str, Any]] = {}
    fingerprint_by_layer: dict[int, str] = {}
    basis_cache_by_layer: dict[int, _LayerBasisCache | None] = {}
    descriptor_snapshot_by_layer: dict[int, str] = {}
    for layer in layers:
        basis_cache = None
        identities: dict[str, str] = {}
        if any(rank > 0 for rank in ranks):
            assert options.basis_root is not None
            basis_cache = _LayerBasisCache(
                basis_root=options.basis_root,
                layer=layer,
                filename_template=options.basis_filename_template,
                model_name=options.model_name,
                weights_id=weights_id,
                patch_size=patch_size,
                long_edge=options.long_edge,
                device=device,
                allowed_sizes=required_sizes,
            )
            for size in required_sizes:
                basis_payload = basis_cache.get(size)
                if int(basis_payload["max_rank"]) < max(ranks):
                    raise ValueError(
                        f"Basis for layer {layer}, size {size} has max_rank="
                        f"{basis_payload['max_rank']}, requires {max(ranks)}"
                    )
                identities[f"{size[0]}x{size[1]}"] = sha256_file(
                    basis_cache.path_for_size(size)
                )
        descriptor_snapshot_by_layer[layer] = _descriptor_snapshot_id(
            options.dino_root,
            layer,
            all_pairs,
            bind_extraction_manifest=options.model_name == DINOV2_MODEL_NAME,
        )
        protocol = make_protocol_payload(
            angular_bin=options.angular_bin,
            pair_file_sha256=pair_file_id,
            estimation_split_sha256=estimation_file_id,
            model_name=options.model_name,
            weights_id=weights_id,
            layer=layer,
            debias_ranks=ranks,
            basis_identities=identities,
            dataset_snapshot_id=dataset_snapshot_id,
            descriptor_snapshot_id=descriptor_snapshot_by_layer[layer],
            long_edge=options.long_edge,
            patch_size=patch_size,
            max_k=options.max_k,
        )
        protocol_by_layer[layer] = protocol
        fingerprint_by_layer[layer] = protocol_fingerprint(protocol)
        basis_cache_by_layer[layer] = basis_cache

    written = resumed = recomputed = 0
    for layer in layers:
        descriptor_metadata = dino_artifact_metadata(
            model_name=options.model_name,
            layer=layer,
            weights_id=weights_id,
            long_edge=options.long_edge,
            downscale_only=False,
        )
        if options.model_name == DINOV2_MODEL_NAME:
            descriptor_metadata.update(
                {
                    key: value
                    for key, value in descriptor_provenance["backbone"].items()
                    if key != "patch_size"
                }
            )
            if descriptor_provenance["source_revision"] != "unknown":
                descriptor_metadata["source_revision"] = descriptor_provenance[
                    "source_revision"
                ]
            if descriptor_provenance["source_dirty"] != "unknown":
                descriptor_metadata["source_dirty"] = descriptor_provenance[
                    "source_dirty"
                ]
        expected_dino_metadata = dino_cache_validation_metadata(
            descriptor_metadata,
            patch_size=patch_size,
            require_model_provenance=False,
        )
        if options.model_name == DINOV2_MODEL_NAME:
            for key in (
                "model_family",
                "descriptor_dim",
                "register_tokens",
                "correction",
                "source_revision",
                "source_dirty",
            ):
                if key in descriptor_metadata:
                    expected_dino_metadata[key] = descriptor_metadata[key]
        descriptor_cache = _DescriptorCache(
            dino_root=options.dino_root,
            layer=layer,
            expected_metadata=expected_dino_metadata,
            device=device,
            capacity=options.descriptor_cache_images,
        )
        for pair_position, pair in enumerate(pairs, 1):
            output_path = shard_path(
                options.output_root, options.angular_bin, layer, pair.pair_index
            )
            if output_path.exists():
                if options.existing == "error":
                    raise FileExistsError(
                        f"Raw shard exists: {output_path}; use resume or overwrite"
                    )
                if options.existing == "resume":
                    try:
                        validate_shard(
                            output_path,
                            expected_fingerprint=fingerprint_by_layer[layer],
                            expected_bin=options.angular_bin,
                            expected_pair_index=pair.pair_index,
                            expected_layer=layer,
                            expected_debias_ranks=ranks,
                            expected_max_k=options.max_k,
                        )
                    except Exception as exc:
                        _resume_may_repair(
                            output_path, fingerprint_by_layer[layer]
                        )
                        recomputed += 1
                        print(f"[recompute] {output_path}: {type(exc).__name__}: {exc}")
                    else:
                        resumed += 1
                        print(
                            f"[resume {pair_position:03d}/{len(pairs):03d}] "
                            f"layer={layer} pair={pair.pair_index}"
                        )
                        continue
            payload = _evaluate_pair_layer(
                pair=pair,
                layer=layer,
                ranks=ranks,
                max_k=options.max_k,
                protocol_payload=protocol_by_layer[layer],
                fingerprint=fingerprint_by_layer[layer],
                descriptor_cache=descriptor_cache,
                annotation_index=annotation_index,
                basis_cache=basis_cache_by_layer[layer],
                geometry_chunk_size=options.geometry_chunk_size,
            )
            atomic_save_shard(output_path, payload)
            validate_shard(
                output_path,
                expected_fingerprint=fingerprint_by_layer[layer],
                expected_bin=options.angular_bin,
                expected_pair_index=pair.pair_index,
                expected_layer=layer,
                expected_debias_ranks=ranks,
                expected_max_k=options.max_k,
            )
            written += 1
            print(
                f"[write {pair_position:03d}/{len(pairs):03d}] "
                f"layer={layer} pair={pair.pair_index} -> {output_path}"
            )

    manifest = {
        "angular_bin": options.angular_bin,
        "model_name": options.model_name,
        "weights_id": weights_id,
        "patch_size": patch_size,
        "correction_mode": correction_mode_for_model(options.model_name),
        "long_edge": options.long_edge,
        "pair_file_sha256": pair_file_id,
        "estimation_split_sha256": estimation_file_id,
        "pair_count": len(pairs),
        "pair_indices": [pair.pair_index for pair in pairs],
        "pair_identities": [
            {
                "pair_index": pair.pair_index,
                "object_name": pair.object_name,
                "image_a": str(pair.image_a),
                "image_b": str(pair.image_b),
                "angle_degrees": pair.angle_degrees,
            }
            for pair in pairs
        ],
        "layers": list(layers),
        "debias_ranks": list(ranks),
        "max_k": options.max_k,
        "expected_shards": len(pairs) * len(layers),
        "written_shards": written,
        "resumed_shards": resumed,
        "recomputed_shards": recomputed,
        "protocol_fingerprints": {
            f"layer{layer}": fingerprint_by_layer[layer] for layer in layers
        },
        "dataset_snapshot_id": dataset_snapshot_id,
        "descriptor_snapshot_ids": {
            f"layer{layer}": descriptor_snapshot_by_layer[layer] for layer in layers
        },
        "descriptor_provenance": descriptor_provenance,
        "split_audit": split_audit,
    }
    _atomic_write_json(bin_root / "evaluation_manifest.json", manifest)

    if options.summarize:
        from .summarize import DenseSummaryOptions, summarize_dense_correspondence

        summarize_dense_correspondence(
            DenseSummaryOptions(
                shard_root=options.output_root,
                angular_bin=options.angular_bin,
                output_dir=bin_root / "reports",
                max_k=options.max_k,
            )
        )
    return DenseEvaluationSummary(
        pair_count=len(pairs),
        layer_count=len(layers),
        written_shards=written,
        resumed_shards=resumed,
        recomputed_shards=recomputed,
        bin_root=bin_root,
    )


def _default_split_path(kind: str, angular_bin: str) -> Path:
    repository_root = Path(__file__).resolve().parents[2]
    return (
        repository_root
        / "data"
        / "splits"
        / "navi"
        / kind
        / f"pairs_wildset_{angular_bin}.csv"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate dense DINO directional/mutual rank-CDF shards."
    )
    parser.add_argument("--bin", dest="angular_bin", choices=ANGULAR_BINS, required=True)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument(
        "--estimation-pairs",
        type=Path,
        help=(
            "Estimation split used for leakage audit. A canonical "
            "pairs_wildset_<bin>.csv also audits its other two sibling bins."
        ),
    )
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--dino-root", type=Path, required=True)
    parser.add_argument("--basis-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=(DINOV3_MODEL_NAME, DINOV2_MODEL_NAME),
        default=DINOV3_MODEL_NAME,
    )
    parser.add_argument("--layer", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument(
        "--rank",
        dest="debias_rank",
        type=int,
        nargs="+",
        help=(
            "Positional-debias ranks (default: 0..600 for DINOv3; raw rank 0 "
            "for DINOv2)."
        ),
    )
    parser.add_argument("--max-k", type=int, default=DEFAULT_MAX_K)
    parser.add_argument("--long-edge", type=int, default=1024)
    parser.add_argument(
        "--patch-size",
        type=int,
        help="Native model patch size (default: 16 for DINOv3, 14 for DINOv2).",
    )
    parser.add_argument(
        "--basis-filename-template",
        default="dinov3_vitl16_{height}x{width}_basis.pt",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--descriptor-cache-images", type=int, default=8)
    parser.add_argument("--geometry-chunk-size", type=int, default=512)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument(
        "--existing", choices=("resume", "overwrite", "error"), default="resume"
    )
    parser.add_argument("--summarize", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if argv and argv[0] == "--":
        argv = argv[1:]
    args = build_parser().parse_args(argv)
    debias_ranks = (
        tuple(args.debias_rank)
        if args.debias_rank is not None
        else ((0,) if args.model == DINOV2_MODEL_NAME else DEFAULT_DEBIAS_RANKS)
    )
    pair_file = args.pairs or _default_split_path("correspondence", args.angular_bin)
    estimation_pair_file = args.estimation_pairs or _default_split_path(
        "estimation", args.angular_bin
    )
    summary = evaluate_dense_correspondence(
        DenseEvaluationOptions(
            angular_bin=args.angular_bin,
            pair_file=pair_file,
            estimation_pair_file=estimation_pair_file,
            image_root=args.image_root,
            dino_root=args.dino_root,
            basis_root=args.basis_root,
            output_root=args.output_root,
            weights=args.weights,
            model_name=args.model,
            layers=tuple(args.layer),
            ranks=debias_ranks,
            max_k=args.max_k,
            long_edge=args.long_edge,
            patch_size=args.patch_size,
            basis_filename_template=args.basis_filename_template,
            device=args.device,
            descriptor_cache_images=args.descriptor_cache_images,
            geometry_chunk_size=args.geometry_chunk_size,
            max_pairs=args.max_pairs,
            existing=args.existing,
            summarize=args.summarize,
        )
    )
    print(
        f"Dense rank-CDF {args.angular_bin}: pairs={summary.pair_count}, "
        f"layers={summary.layer_count}, written={summary.written_shards}, "
        f"resumed={summary.resumed_shards}, recomputed={summary.recomputed_shards}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
