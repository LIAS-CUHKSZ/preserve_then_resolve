"""Descriptor interpolation, positional debiasing, and many-to-many matching.

The numerical routines in this module are direct relocations of the experiment
scripts. In particular, progressive associations remain ordered by first
mutual rank and then by decreasing cosine similarity.
"""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import ASSOCIATION_COLUMNS


@dataclass(frozen=True)
class ImageFeatures:
    keypoints: np.ndarray
    descriptors: Any
    image_size: tuple[int, int]  # Patch-padded (width, height), used for basis filenames.
    metadata: dict[str, Any] = field(default_factory=dict)


def require_torch():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise RuntimeError(
            "Matching requires PyTorch. Install the `vision` dependencies or use "
            "the provided Conda environment."
        ) from exc
    return torch, functional


def resolve_device(device: str):
    torch, _ = require_torch()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device!r}, but CUDA is not available")
    return resolved


def interpolate_descriptors(
    keypoints: np.ndarray,
    descriptor_map: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample a DINO map at image-space keypoints.

    The patch-center transform ``xy / patch_size - 0.5`` and boundary filtering
    intentionally match the experiment implementation.
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    descriptor_map = np.asarray(descriptor_map, dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"keypoints must have shape [N, 2], got {keypoints.shape}")
    if descriptor_map.ndim != 3:
        raise ValueError("descriptor_map must have shape [grid_h, grid_w, channels]")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")

    desc_dim = descriptor_map.shape[2]
    if len(keypoints) == 0 or descriptor_map.shape[0] == 0 or descriptor_map.shape[1] == 0:
        return np.empty((0, 2), np.float32), np.empty((0, desc_dim), np.float32)

    height, width, _ = descriptor_map.shape
    feature_coords = keypoints / patch_size - 0.5
    x = feature_coords[:, 0]
    y = feature_coords[:, 1]
    valid = (x >= 0.0) & (x <= width - 1) & (y >= 0.0) & (y <= height - 1)
    if not np.any(valid):
        return np.empty((0, 2), np.float32), np.empty((0, desc_dim), np.float32)

    keypoints = keypoints[valid]
    x, y = x[valid], y[valid]
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    dx = (x - x0).astype(np.float32, copy=False)
    dy = (y - y0).astype(np.float32, copy=False)

    descriptors = (
        ((1.0 - dx) * (1.0 - dy))[:, None] * descriptor_map[y0, x0]
        + (dx * (1.0 - dy))[:, None] * descriptor_map[y0, x1]
        + ((1.0 - dx) * dy)[:, None] * descriptor_map[y1, x0]
        + (dx * dy)[:, None] * descriptor_map[y1, x1]
    )
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    np.divide(descriptors, np.maximum(norms, 1e-12), out=descriptors)
    return keypoints.astype(np.float32, copy=False), descriptors.astype(np.float32, copy=False)


def interpolate_resized_descriptors(
    keypoints: np.ndarray,
    descriptor_map: np.ndarray,
    patch_size: int,
    *,
    source_hw: tuple[int, int],
    resized_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a resized patch map while retaining source-frame keypoints.

    SuperPoint coordinates remain in the source image frame used by pose
    estimation.  Sampling uses the half-pixel resize convention, so source
    coordinate ``x`` maps to patch-grid coordinate
    ``(x + 0.5) * resized_width / source_width / patch_size - 0.5`` (and
    analogously for ``y``).  Returned keypoints are the original, unscaled
    SuperPoint coordinates after boundary filtering.
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    descriptor_map = np.asarray(descriptor_map, dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"keypoints must have shape [N, 2], got {keypoints.shape}")
    if descriptor_map.ndim != 3:
        raise ValueError("descriptor_map must have shape [grid_h, grid_w, channels]")
    if patch_size <= 0 or min(*source_hw, *resized_hw) <= 0:
        raise ValueError("patch size and image dimensions must be positive")
    expected_hw = (
        descriptor_map.shape[0] * patch_size,
        descriptor_map.shape[1] * patch_size,
    )
    if tuple(resized_hw) != expected_hw:
        raise ValueError(
            f"resized_hw {resized_hw} differs from descriptor grid {expected_hw}"
        )
    desc_dim = descriptor_map.shape[2]
    if not len(keypoints):
        return np.empty((0, 2), np.float32), np.empty((0, desc_dim), np.float32)

    source_h, source_w = source_hw
    resized_h, resized_w = resized_hw
    feature_x = (
        (keypoints[:, 0] + 0.5) * (resized_w / source_w) / patch_size - 0.5
    )
    feature_y = (
        (keypoints[:, 1] + 0.5) * (resized_h / source_h) / patch_size - 0.5
    )
    height, width, _ = descriptor_map.shape
    valid = (
        (feature_x >= 0.0)
        & (feature_x <= width - 1)
        & (feature_y >= 0.0)
        & (feature_y <= height - 1)
    )
    if not np.any(valid):
        return np.empty((0, 2), np.float32), np.empty((0, desc_dim), np.float32)
    retained = keypoints[valid]
    x, y = feature_x[valid], feature_y[valid]
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    dx = (x - x0).astype(np.float32, copy=False)
    dy = (y - y0).astype(np.float32, copy=False)
    descriptors = (
        ((1.0 - dx) * (1.0 - dy))[:, None] * descriptor_map[y0, x0]
        + (dx * (1.0 - dy))[:, None] * descriptor_map[y0, x1]
        + ((1.0 - dx) * dy)[:, None] * descriptor_map[y1, x0]
        + (dx * dy)[:, None] * descriptor_map[y1, x1]
    )
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    np.divide(descriptors, np.maximum(norms, 1e-12), out=descriptors)
    return retained.astype(np.float32, copy=False), descriptors.astype(
        np.float32, copy=False
    )


def load_basis_payload(
    path: Path,
    device: Any,
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    torch, _ = require_torch()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Debiasing basis does not exist: {path}")
    payload = torch.load(path, map_location="cpu")
    required_payload = {"basis", "max_rank", "meta"}
    if not isinstance(payload, dict) or set(payload) != required_payload:
        raise ValueError(f"Unsupported debiasing basis payload: {path}")
    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        raise ValueError(f"Debiasing basis metadata must be a mapping: {path}")
    metadata = dict(meta)
    for key, expected in (expected_metadata or {}).items():
        if key not in metadata:
            raise ValueError(
                f"Unverifiable debiasing basis {path}: required metadata `{key}` is missing"
            )
        if metadata[key] != expected:
            raise ValueError(
                f"Stale debiasing basis {path}: metadata `{key}` is "
                f"{metadata[key]!r}, expected {expected!r}"
            )
    basis = payload["basis"].to(device=device, dtype=torch.float32)
    try:
        max_rank = int(payload["max_rank"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid max_rank in debiasing basis: {path}") from exc
    if max_rank <= 0:
        raise ValueError(f"Invalid max_rank in debiasing basis: {path}")
    if basis.ndim != 2:
        raise ValueError(f"Debiasing basis must be a matrix: {path}")
    stored_rank = int(basis.shape[1])
    if stored_rank != max_rank:
        raise ValueError(
            f"Debiasing basis max_rank={max_rank} does not match stored rank "
            f"{stored_rank}: {path}"
        )
    return {"basis": basis, "max_rank": max_rank, "meta": metadata}


def basis_for_dim(payload: dict[str, Any], dim: int, descriptor_dim: int):
    if dim <= 0:
        raise ValueError("A debiasing dimension must be positive")
    if not {"basis", "max_rank"}.issubset(payload):
        raise ValueError("Basis payload must contain basis and max_rank")
    max_rank = int(payload["max_rank"])
    if dim > max_rank:
        raise ValueError(f"Requested rank {dim} exceeds stored max_rank {max_rank}")
    basis = payload["basis"][:, :dim]
    if basis.ndim != 2 or basis.shape[0] != descriptor_dim or basis.shape[1] < dim:
        raise ValueError(
            f"Basis shape {tuple(basis.shape)} is incompatible with descriptor dimension "
            f"{descriptor_dim} and requested rank {dim}"
        )
    return basis


def debias_descriptors(descriptors: Any, basis: Any):
    _, functional = require_torch()
    if descriptors.numel() == 0:
        return descriptors
    debiased = descriptors - (descriptors @ basis) @ basis.T
    return functional.normalize(debiased, p=2, dim=1)


def progressive_mutual_knn(
    left_desc: Any,
    right_desc: Any,
    max_k: int,
    association_upperbound: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return associations and the first K at which each pair is mutual.

    Output is sorted by ``first_k`` ascending and, within each K, similarity
    descending. The upper-bound stop reproduces the original incremental K loop.
    """
    torch, functional = require_torch()
    if max_k <= 0:
        raise ValueError("max_k must be positive")
    if association_upperbound < 0:
        raise ValueError("association_upperbound must be non-negative")
    if left_desc.ndim != 2 or right_desc.ndim != 2:
        raise ValueError("descriptor tensors must have shape [N, D]")
    if left_desc.shape[1] != right_desc.shape[1]:
        raise ValueError("left and right descriptor dimensions do not agree")
    if left_desc.shape[0] == 0 or right_desc.shape[0] == 0:
        return _empty_progressive()

    left_desc = functional.normalize(left_desc, p=2, dim=1)
    right_desc = functional.normalize(right_desc, p=2, dim=1)
    similarity = left_desc @ right_desc.T
    n_left, n_right = similarity.shape
    left_max_k = min(max_k, n_right)
    right_max_k = min(max_k, n_left)
    unranked = max(left_max_k, right_max_k) + 1
    if left_max_k <= 0 or right_max_k <= 0:
        return _empty_progressive()

    left_ranked = torch.topk(similarity, k=left_max_k, dim=1).indices
    right_ranked_by_right = torch.topk(similarity, k=right_max_k, dim=0).indices.T

    left_anchor = torch.arange(n_left, device=similarity.device).repeat_interleave(left_max_k)
    right_from_left = left_ranked.reshape(-1)
    left_rank = torch.arange(
        1, left_max_k + 1, device=similarity.device, dtype=torch.int32
    ).repeat(n_left)
    pairs_from_left = torch.stack((left_anchor, right_from_left), dim=1)

    right_anchor = torch.arange(n_right, device=similarity.device).repeat_interleave(right_max_k)
    left_from_right = right_ranked_by_right.reshape(-1)
    right_rank = torch.arange(
        1, right_max_k + 1, device=similarity.device, dtype=torch.int32
    ).repeat(n_right)
    pairs_from_right = torch.stack((left_from_right, right_anchor), dim=1)

    left_pair_count = pairs_from_left.shape[0]
    candidates = torch.cat((pairs_from_left, pairs_from_right), dim=0)
    unique_pairs, candidate_to_unique = torch.unique(candidates, dim=0, return_inverse=True)
    left_rank_at_pair = torch.full(
        (unique_pairs.shape[0],), unranked, dtype=torch.int32, device=similarity.device
    )
    right_rank_at_pair = torch.full_like(left_rank_at_pair, unranked)
    left_rank_at_pair.scatter_(0, candidate_to_unique[:left_pair_count], left_rank)
    right_rank_at_pair.scatter_(0, candidate_to_unique[left_pair_count:], right_rank)

    is_mutual = (left_rank_at_pair < unranked) & (right_rank_at_pair < unranked)
    if not bool(is_mutual.any()):
        return _empty_progressive()
    first_k = torch.maximum(left_rank_at_pair, right_rank_at_pair)[is_mutual]
    pairs = unique_pairs[is_mutual]
    left_indices, right_indices = pairs[:, 0], pairs[:, 1]

    if association_upperbound > 0:
        cumulative = torch.bincount(first_k, minlength=unranked)[1:].cumsum(0)
        stop_at = (cumulative >= association_upperbound).nonzero(as_tuple=False)
        if stop_at.numel() > 0:
            stop_k = int(stop_at[0, 0].item()) + 1
            keep = first_k <= stop_k
            left_indices, right_indices, first_k = (
                left_indices[keep],
                right_indices[keep],
                first_k[keep],
            )

    packed = torch.stack((left_indices, right_indices, first_k.to(torch.int64)), dim=1)
    scores = similarity[left_indices, right_indices]
    packed_np = packed.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy().astype(np.float32, copy=False)
    order = np.lexsort((-scores_np, packed_np[:, 2]))
    if association_upperbound > 0:
        order = order[:association_upperbound]
    packed_np, scores_np = packed_np[order], scores_np[order]
    return (
        packed_np[:, 0].astype(np.int64, copy=False),
        packed_np[:, 1].astype(np.int64, copy=False),
        scores_np,
        packed_np[:, 2].astype(np.int64, copy=False),
    )


def _empty_progressive() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    empty_i = np.empty((0,), dtype=np.int64)
    empty_s = np.empty((0,), dtype=np.float32)
    return empty_i, empty_i.copy(), empty_s, empty_i.copy()


def build_association_rows(
    left_features: ImageFeatures,
    right_features: ImageFeatures,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    scores: np.ndarray,
    first_ks: np.ndarray,
) -> list[list[float | int]]:
    """Build CSV rows while retaining legacy compact node IDs."""
    lengths = {len(left_indices), len(right_indices), len(scores), len(first_ks)}
    if len(lengths) != 1:
        raise ValueError("Association arrays have different lengths")
    left_compact = {old: new for new, old in enumerate(sorted(set(left_indices.tolist())))}
    right_compact = {old: new for new, old in enumerate(sorted(set(right_indices.tolist())))}
    rows: list[list[float | int]] = []
    for left_idx, right_idx, score, first_k in zip(
        left_indices, right_indices, scores, first_ks
    ):
        x1, y1 = left_features.keypoints[left_idx]
        x2, y2 = right_features.keypoints[right_idx]
        rows.append(
            [
                left_compact[int(left_idx)],
                right_compact[int(right_idx)],
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                float(score),
                int(first_k),
            ]
        )
    return rows


def write_association_csv(path: Path, rows: list[list[float | int]]) -> None:
    """Atomically write a schema-valid association CSV, including empty output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.writer(handle)
        writer.writerow(ASSOCIATION_COLUMNS)
        writer.writerows(rows)
    try:
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
