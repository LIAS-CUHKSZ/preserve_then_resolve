"""NAVI camera geometry sampled at complete DINO patch centers.

NAVI depth PNGs contain inverse depth.  Camera annotations are object-to-camera
transforms with ``wxyz`` quaternions and millimetre translations.  This module
keeps every geometric calculation in float64 metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


NAVI_INVERSE_DEPTH_SCALE_M = 655.35
PATCH_CENTER_SAMPLING_ID = "resized-patch-center-pixel-inverse-nearest-v3"


@dataclass(frozen=True)
class PatchGeometry:
    """Geometry and masks indexed by flattened padded-grid patch index."""

    grid_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    original_hw: tuple[int, int]
    complete_indices: np.ndarray
    object_indices: np.ndarray
    valid_object_indices: np.ndarray
    is_object: np.ndarray
    has_valid_depth: np.ndarray
    xyz_m: np.ndarray

    def __post_init__(self) -> None:
        count = self.grid_hw[0] * self.grid_hw[1]
        if self.is_object.shape != (count,):
            raise ValueError("is_object must be indexed by the full patch grid")
        if self.has_valid_depth.shape != (count,):
            raise ValueError("has_valid_depth must be indexed by the full patch grid")
        if self.xyz_m.shape != (count, 3):
            raise ValueError("xyz_m must have shape [grid_h * grid_w, 3]")


def _normalized_quaternion(values: Any) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError(f"Invalid wxyz quaternion: {values!r}")
    norm = float(np.linalg.norm(quaternion))
    if norm == 0.0:
        raise ValueError("Quaternion must be non-zero")
    return quaternion / norm


def quaternion_to_rotation_matrix(values: Any) -> np.ndarray:
    """Convert a normalized-or-unnormalized ``[w, x, y, z]`` quaternion."""
    w, x, y, z = _normalized_quaternion(values)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _camera_pose(camera: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "q" not in camera or "t" not in camera:
        raise ValueError("NAVI camera annotation must contain q and t")
    rotation = quaternion_to_rotation_matrix(camera["q"])
    translation_mm = np.asarray(camera["t"], dtype=np.float64)
    if translation_mm.shape != (3,) or not np.isfinite(translation_mm).all():
        raise ValueError(f"Invalid camera translation: {camera['t']!r}")
    return rotation, translation_mm / 1000.0


def relative_camera_transform(
    source_camera: Mapping[str, Any], destination_camera: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Return source-camera to destination-camera ``R, t`` in metres.

    Each input maps object coordinates into its camera, so
    ``R_ds = R_d R_s.T`` and ``t_ds = t_d - R_ds t_s``.
    """
    source_rotation, source_translation = _camera_pose(source_camera)
    destination_rotation, destination_translation = _camera_pose(destination_camera)
    rotation = destination_rotation @ source_rotation.T
    translation = destination_translation - rotation @ source_translation
    return rotation, translation


def decode_navi_inverse_depth(raw_depth: np.ndarray) -> np.ndarray:
    """Decode NAVI uint16 inverse depth as float64 metres; zero becomes NaN."""
    raw = np.asarray(raw_depth)
    if raw.ndim != 2:
        raise ValueError(f"NAVI depth must be a 2D array, got {raw.shape}")
    if raw.dtype != np.uint16:
        raise ValueError(f"NAVI inverse depth must be uint16, got {raw.dtype}")
    result = np.full(raw.shape, np.nan, dtype=np.float64)
    valid = raw != 0
    result[valid] = NAVI_INVERSE_DEPTH_SCALE_M / raw[valid].astype(np.float64)
    return result


def _annotation_hw(annotation: Mapping[str, Any]) -> tuple[int, int]:
    try:
        height, width = (int(value) for value in annotation["image_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("NAVI annotation image_size must contain height and width") from exc
    if min(height, width) <= 0:
        raise ValueError("NAVI annotation image dimensions must be positive")
    return height, width


def build_patch_geometry(
    *,
    raw_depth: np.ndarray,
    object_mask: np.ndarray,
    annotation: Mapping[str, Any],
    grid_hw: tuple[int, int],
    resized_hw: tuple[int, int],
    patch_size: int = 16,
) -> PatchGeometry:
    """Sample mask/depth at centers of patches wholly inside the resized image.

    Flattened indices always refer to the complete padded descriptor grid.  A
    patch in the partial rightmost or bottommost padded strip is deliberately
    absent from ``complete_indices``.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    grid_height, grid_width = (int(value) for value in grid_hw)
    resized_height, resized_width = (int(value) for value in resized_hw)
    if min(grid_height, grid_width, resized_height, resized_width) <= 0:
        raise ValueError("Grid and resized dimensions must be positive")
    if grid_height * patch_size < resized_height or grid_width * patch_size < resized_width:
        raise ValueError("Descriptor grid is smaller than the resized image")

    original_height, original_width = _annotation_hw(annotation)
    depth_raw = np.asarray(raw_depth)
    mask = np.asarray(object_mask)
    if depth_raw.shape != (original_height, original_width):
        raise ValueError(
            f"Depth shape {depth_raw.shape} disagrees with annotation "
            f"{(original_height, original_width)}"
        )
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != (original_height, original_width):
        raise ValueError(
            f"Mask shape {mask.shape} disagrees with annotation "
            f"{(original_height, original_width)}"
        )

    complete_rows = min(grid_height, resized_height // patch_size)
    complete_columns = min(grid_width, resized_width // patch_size)
    rows, columns = np.meshgrid(
        np.arange(complete_rows, dtype=np.int64),
        np.arange(complete_columns, dtype=np.int64),
        indexing="ij",
    )
    rows = rows.reshape(-1)
    columns = columns.reshape(-1)
    complete_indices = rows * grid_width + columns

    # Pixel-index coordinate of the center of each patch's 16 pixel centers.
    # For the first patch this is 7.5, while 8.0 is the same location in the
    # continuous pixel-boundary convention.
    resized_x = (columns.astype(np.float64) + 0.5) * patch_size - 0.5
    resized_y = (rows.astype(np.float64) + 0.5) * patch_size - 0.5
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height
    # Invert resize in the pixel-center convention. Integer rounding in the
    # resized dimensions makes the two axis scales slightly different.
    original_x = (resized_x + 0.5) / scale_x - 0.5
    original_y = (resized_y + 0.5) / scale_y - 0.5
    sample_x = np.clip(np.floor(original_x + 0.5).astype(np.int64), 0, original_width - 1)
    sample_y = np.clip(np.floor(original_y + 0.5).astype(np.int64), 0, original_height - 1)

    sampled_raw_depth = depth_raw[sample_y, sample_x]
    if sampled_raw_depth.dtype != np.uint16:
        raise ValueError(
            f"NAVI inverse depth must be uint16, got {sampled_raw_depth.dtype}"
        )
    sampled_depth_m = np.full(len(complete_indices), np.nan, dtype=np.float64)
    sampled_depth_valid = sampled_raw_depth != 0
    sampled_depth_m[sampled_depth_valid] = (
        NAVI_INVERSE_DEPTH_SCALE_M
        / sampled_raw_depth[sampled_depth_valid].astype(np.float64)
    )
    sampled_object = mask[sample_y, sample_x] != 0

    try:
        focal_length = float(annotation["camera"]["focal_length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("NAVI annotation is missing camera.focal_length") from exc
    if not np.isfinite(focal_length) or focal_length <= 0.0:
        raise ValueError("NAVI focal length must be finite and positive")
    center_x = original_width / 2.0
    center_y = original_height / 2.0
    sampled_xyz = np.full((len(complete_indices), 3), np.nan, dtype=np.float64)
    sampled_xyz[:, 2] = sampled_depth_m
    sampled_xyz[:, 0] = (original_x - center_x) / focal_length * sampled_depth_m
    sampled_xyz[:, 1] = (original_y - center_y) / focal_length * sampled_depth_m

    patch_count = grid_height * grid_width
    is_object = np.zeros(patch_count, dtype=np.bool_)
    has_valid_depth = np.zeros(patch_count, dtype=np.bool_)
    xyz_m = np.full((patch_count, 3), np.nan, dtype=np.float64)
    is_object[complete_indices] = sampled_object
    has_valid_depth[complete_indices] = sampled_depth_valid
    xyz_m[complete_indices] = sampled_xyz
    object_indices = complete_indices[sampled_object]
    valid_object_indices = complete_indices[sampled_object & sampled_depth_valid]
    return PatchGeometry(
        grid_hw=(grid_height, grid_width),
        resized_hw=(resized_height, resized_width),
        original_hw=(original_height, original_width),
        complete_indices=complete_indices,
        object_indices=object_indices,
        valid_object_indices=valid_object_indices,
        is_object=is_object,
        has_valid_depth=has_valid_depth,
        xyz_m=xyz_m,
    )


def correspondence_errors(
    *,
    source: PatchGeometry,
    destination: PatchGeometry,
    source_indices: np.ndarray,
    destination_indices: np.ndarray,
    source_camera: Mapping[str, Any],
    destination_camera: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return destination-valid flags and scalar 3D errors in float64 metres.

    A destination is protocol-valid only when its sampled center lies on the
    object mask and has non-zero depth.  Background and invalid-depth matches
    receive ``+inf``.
    """
    source_indices = np.asarray(source_indices, dtype=np.int64)
    destination_indices = np.asarray(destination_indices, dtype=np.int64)
    if source_indices.shape != destination_indices.shape or source_indices.ndim != 1:
        raise ValueError("Source and destination patch indices must be equal-length vectors")
    if np.any(source_indices < 0) or np.any(source_indices >= len(source.is_object)):
        raise IndexError("Source patch index is outside the descriptor grid")
    if np.any(destination_indices < 0) or np.any(
        destination_indices >= len(destination.is_object)
    ):
        raise IndexError("Destination patch index is outside the descriptor grid")
    source_xyz = source.xyz_m[source_indices]
    if not np.isfinite(source_xyz).all():
        raise ValueError("Every selected source patch must have finite XYZ")

    destination_valid = (
        destination.is_object[destination_indices]
        & destination.has_valid_depth[destination_indices]
        & np.isfinite(destination.xyz_m[destination_indices]).all(axis=1)
    )
    errors = np.full(len(source_indices), np.inf, dtype=np.float64)
    if np.any(destination_valid):
        rotation, translation = relative_camera_transform(
            source_camera, destination_camera
        )
        predicted = source_xyz @ rotation.T + translation
        delta = predicted[destination_valid] - destination.xyz_m[
            destination_indices[destination_valid]
        ]
        errors[destination_valid] = np.linalg.norm(delta, axis=1)
    if np.any(errors < 0.0) or np.any(np.isnan(errors)):
        raise AssertionError("3D errors must be non-negative finite values or +inf")
    return destination_valid, errors


def candidate_correspondence_errors(
    *,
    source: PatchGeometry,
    destination: PatchGeometry,
    source_indices: np.ndarray,
    destination_indices: np.ndarray,
    source_camera: Mapping[str, Any],
    destination_camera: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return target object/depth flags and raw paired 3D errors in metres.

    Unlike :func:`correspondence_errors`, a background target with valid depth
    keeps its finite raw error. Correctness is decided later using both the
    object flag and the strict geometric threshold; this preserves the audit
    information without allowing background candidates to become positives.
    """
    source_indices = np.asarray(source_indices, dtype=np.int64)
    destination_indices = np.asarray(destination_indices, dtype=np.int64)
    if source_indices.shape != destination_indices.shape or source_indices.ndim != 1:
        raise ValueError("Source and destination patch indices must be equal-length vectors")
    if np.any(source_indices < 0) or np.any(source_indices >= len(source.is_object)):
        raise IndexError("Source patch index is outside the descriptor grid")
    if np.any(destination_indices < 0) or np.any(
        destination_indices >= len(destination.is_object)
    ):
        raise IndexError("Destination patch index is outside the descriptor grid")
    source_xyz = source.xyz_m[source_indices]
    if not np.isfinite(source_xyz).all():
        raise ValueError("Every selected source patch must have finite XYZ")

    target_is_object = destination.is_object[destination_indices].astype(
        np.bool_, copy=True
    )
    target_has_depth = (
        destination.has_valid_depth[destination_indices]
        & np.isfinite(destination.xyz_m[destination_indices]).all(axis=1)
    )
    errors = np.full(len(source_indices), np.inf, dtype=np.float64)
    if np.any(target_has_depth):
        rotation, translation = relative_camera_transform(
            source_camera, destination_camera
        )
        predicted = source_xyz @ rotation.T + translation
        delta = predicted[target_has_depth] - destination.xyz_m[
            destination_indices[target_has_depth]
        ]
        errors[target_has_depth] = np.linalg.norm(delta, axis=1)
    if np.any(errors < 0.0) or np.any(np.isnan(errors)):
        raise AssertionError("3D errors must be non-negative finite values or +inf")
    return target_is_object, target_has_depth.astype(np.bool_), errors


def minimum_object_correspondence_errors(
    *,
    source: PatchGeometry,
    destination: PatchGeometry,
    source_indices: np.ndarray,
    source_camera: Mapping[str, Any],
    destination_camera: Mapping[str, Any],
    chunk_size: int = 512,
) -> np.ndarray:
    """Minimum raw 3D error to any valid destination-object patch center.

    This descriptor-independent quantity records whether the discretized
    destination grid contains a geometrically valid candidate at a requested
    threshold. It is computed for every valid source-object query, including
    those whose correct candidate is absent from the stored top-K ranking.
    """
    indices = np.asarray(source_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("source_indices must be a vector")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    source_xyz = source.xyz_m[indices]
    if not np.isfinite(source_xyz).all():
        raise ValueError("Every selected source patch must have finite XYZ")
    destination_indices = destination.valid_object_indices
    result = np.full(len(indices), np.inf, dtype=np.float64)
    if len(indices) == 0 or len(destination_indices) == 0:
        return result
    rotation, translation = relative_camera_transform(
        source_camera, destination_camera
    )
    predicted = source_xyz @ rotation.T + translation
    target_xyz = destination.xyz_m[destination_indices]
    for start in range(0, len(predicted), chunk_size):
        stop = min(start + chunk_size, len(predicted))
        delta = predicted[start:stop, None, :] - target_xyz[None, :, :]
        squared = np.einsum("ijk,ijk->ij", delta, delta)
        result[start:stop] = np.sqrt(np.min(squared, axis=1))
    if np.any(result < 0.0) or np.any(np.isnan(result)):
        raise AssertionError("Minimum 3D errors must be non-negative or +inf")
    return result
