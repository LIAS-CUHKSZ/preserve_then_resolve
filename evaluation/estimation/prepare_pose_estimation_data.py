#!/usr/bin/env python3
"""Prepare pair-bound pose-estimation inputs from the downloaded raw datasets.

Raw inputs are read from ``--data-root`` (or the repository ``data``
directory), while the versioned NAVI pair lists are read from the repository's
``data/splits`` tree. The script writes one canonical leaf per evaluation
subset containing ``pairs.csv``, ``pose_intrinsics.csv``, and
``pose_intrinsics_manifest.json``. Intrinsics use the same 1024-pixel long-edge
coordinates as DINO/SuperPoint. METU uses the paper's downscale-only policy;
the other datasets retain the historical resize policy, including upsampling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image


LONG_EDGE = 1024
PAPER_PAIR_METADATA_SHA256 = {
    "ScanNet": "522ee01d4e18b5d0182ed934aa1cab9896183ee3321f1ba97fabc77867c412a2",
    "MegaDepth": "bd2d6d843fa573ef15665603b8fdb32830efc8345d98908c340822e5f7f17c80",
}
PAIR_FIELDS = (
    "image_1",
    "image_2",
    "image_1_height",
    "image_1_width",
    "image_2_height",
    "image_2_width",
)
POSE_FIELDS = (
    "pair_idx",
    "fx1",
    "cx1",
    "fy1",
    "cy1",
    "fx2",
    "cx2",
    "fy2",
    "cy2",
    "qw",
    "qx",
    "qy",
    "qz",
    "tx",
    "ty",
    "tz",
)
METU_POSE_FIELDS = (*POSE_FIELDS, "dist0_coeffs", "dist1_coeffs")


@dataclass(frozen=True)
class PreparedSubset:
    """Metadata needed to run extraction and matching for one subset."""

    dataset_label: str
    root_name: str
    subset_name: str
    image_root: Path
    pair_file: Path
    pose_file: Path
    downscale_only: bool
    pair_count: int


class ImageSizeCache:
    """Read image dimensions once and validate every referenced image."""

    def __init__(self, image_root: Path) -> None:
        self.image_root = image_root.resolve()
        self._sizes: dict[str, tuple[int, int]] = {}

    def get(self, relative_path: str) -> tuple[int, int]:
        if relative_path not in self._sizes:
            path = (self.image_root / relative_path).resolve()
            try:
                path.relative_to(self.image_root)
            except ValueError as exc:
                raise ValueError(f"Image path escapes its root: {relative_path}") from exc
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                width, height = image.size
            if min(width, height) <= 0:
                raise ValueError(f"Invalid image dimensions for {path}: {width}x{height}")
            self._sizes[relative_path] = int(height), int(width)
        return self._sizes[relative_path]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _format_float(value: float) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Non-finite value: {numeric}")
    return "0" if numeric == 0.0 else format(numeric, ".17g")


def _rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> tuple[float, ...]:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 1e-15)) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / scale
        qx = 0.25 * scale
        qy = (rotation[0, 1] + rotation[1, 0]) / scale
        qz = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 1e-15)) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / scale
        qx = (rotation[0, 1] + rotation[1, 0]) / scale
        qy = 0.25 * scale
        qz = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 1e-15)) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / scale
        qx = (rotation[0, 2] + rotation[2, 0]) / scale
        qy = (rotation[1, 2] + rotation[2, 1]) / scale
        qz = 0.25 * scale
    quaternion = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm == 0.0:
        raise ValueError("Rotation produced a zero quaternion")
    quaternion /= norm
    first_nonzero = np.flatnonzero(np.abs(quaternion) > 1e-15)
    if first_nonzero.size and quaternion[int(first_nonzero[0])] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


def _resize_scales(
    height: int,
    width: int,
    *,
    downscale_only: bool,
    long_edge: int = LONG_EDGE,
) -> tuple[float, float]:
    if downscale_only and max(height, width) <= long_edge:
        return 1.0, 1.0
    scale = long_edge / float(max(height, width))
    resized_width = int(width * scale)
    resized_height = int(height * scale)
    return resized_width / float(width), resized_height / float(height)


def _pair_identity_sha256(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for pair_index, row in enumerate(rows, start=1):
        digest.update(
            f"{pair_index}\t{row['image_1']}\t{row['image_2']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _write_csv_atomic(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(output.name)
    temporary_path.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as output:
        json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
        temporary_path = Path(output.name)
    temporary_path.replace(path)


def _read_csv_rows(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        return tuple(reader.fieldnames), list(reader)


def _string_rows(
    rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]
) -> list[dict[str, str]]:
    return [
        {field: str(row[field]) for field in fieldnames}
        for row in rows
    ]


def _validate_existing_subset(
    *,
    pair_file: Path,
    pose_file: Path,
    manifest_file: Path,
    pair_rows: Sequence[dict[str, Any]],
    pose_rows: Sequence[dict[str, Any]],
    pose_fields: Sequence[str],
    downscale_only: bool,
    source_files: Sequence[Path],
) -> None:
    required = (pair_file, pose_file, manifest_file)
    present = tuple(path.is_file() for path in required)
    if not any(present):
        raise FileNotFoundError(pair_file)
    if not all(present):
        missing = [str(path) for path, exists in zip(required, present) if not exists]
        raise FileNotFoundError(
            f"Partially prepared pose subset; missing {missing}. "
            "Use --existing overwrite to regenerate all three files."
        )

    pair_header, actual_pairs = _read_csv_rows(pair_file)
    pose_header, actual_poses = _read_csv_rows(pose_file)
    if pair_header != tuple(PAIR_FIELDS):
        raise ValueError(f"{pair_file}: unexpected columns {pair_header}")
    if pose_header != tuple(pose_fields):
        raise ValueError(f"{pose_file}: unexpected columns {pose_header}")
    if actual_pairs != _string_rows(pair_rows, PAIR_FIELDS):
        raise ValueError(
            f"{pair_file}: retained rows differ from the current raw-data inputs"
        )
    if actual_poses != _string_rows(pose_rows, pose_fields):
        raise ValueError(
            f"{pose_file}: retained rows differ from the current raw-data inputs"
        )

    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "pair_file": pair_file.name,
        "pair_file_sha256": _sha256(pair_file),
        "pair_identity_sha256": _pair_identity_sha256(pair_rows),
        "pair_count": len(pair_rows),
        "long_edge": LONG_EDGE,
        "downscale_only": downscale_only,
        "pose_convention": "camera1_to_camera2",
        "pose_intrinsics_sha256": _sha256(pose_file),
        "source_files": [str(path.resolve()) for path in source_files],
    }
    if payload != expected:
        raise ValueError(
            f"{manifest_file}: retained manifest does not match the current "
            "pair/pose inputs"
        )


def _write_subset(
    *,
    output_root: Path,
    dataset_label: str,
    root_name: str,
    subset_name: str,
    image_root: Path,
    pair_rows: list[dict[str, Any]],
    pose_rows: list[dict[str, Any]],
    pose_fields: Sequence[str],
    downscale_only: bool,
    source_files: Sequence[Path],
    existing: str,
) -> PreparedSubset:
    if len(pair_rows) != len(pose_rows):
        raise ValueError(
            f"{root_name}/{subset_name}: {len(pair_rows)} pairs but "
            f"{len(pose_rows)} poses"
        )
    leaf = output_root / root_name / subset_name
    pair_file = leaf / "pairs.csv"
    pose_file = leaf / "pose_intrinsics.csv"
    manifest_file = leaf / "pose_intrinsics_manifest.json"
    if existing not in {"error", "validate", "overwrite"}:
        raise ValueError("existing must be one of: error, validate, overwrite")
    present = [path for path in (pair_file, pose_file, manifest_file) if path.exists()]
    if present and existing == "error":
        raise FileExistsError(
            f"Refusing to overwrite {present[0]}; use --existing validate or overwrite"
        )
    if present and existing == "validate":
        _validate_existing_subset(
            pair_file=pair_file,
            pose_file=pose_file,
            manifest_file=manifest_file,
            pair_rows=pair_rows,
            pose_rows=pose_rows,
            pose_fields=pose_fields,
            downscale_only=downscale_only,
            source_files=source_files,
        )
        return PreparedSubset(
            dataset_label=dataset_label,
            root_name=root_name,
            subset_name=subset_name,
            image_root=image_root.resolve(),
            pair_file=pair_file.resolve(),
            pose_file=pose_file.resolve(),
            downscale_only=downscale_only,
            pair_count=len(pair_rows),
        )
    _write_csv_atomic(pair_file, PAIR_FIELDS, pair_rows)
    _write_csv_atomic(pose_file, pose_fields, pose_rows)
    _write_json_atomic(
        manifest_file,
        {
            "schema_version": 1,
            "pair_file": pair_file.name,
            "pair_file_sha256": _sha256(pair_file),
            "pair_identity_sha256": _pair_identity_sha256(pair_rows),
            "pair_count": len(pair_rows),
            "long_edge": LONG_EDGE,
            "downscale_only": downscale_only,
            "pose_convention": "camera1_to_camera2",
            "pose_intrinsics_sha256": _sha256(pose_file),
            "source_files": [str(path.resolve()) for path in source_files],
        },
    )
    return PreparedSubset(
        dataset_label=dataset_label,
        root_name=root_name,
        subset_name=subset_name,
        image_root=image_root.resolve(),
        pair_file=pair_file.resolve(),
        pose_file=pose_file.resolve(),
        downscale_only=downscale_only,
        pair_count=len(pair_rows),
    )


def _prepare_pairs_with_gt(
    *,
    source: Path,
    image_root: Path,
    invert_transform: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sizes = ImageSizeCache(image_root)
    pair_rows: list[dict[str, Any]] = []
    pose_rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 38:
            raise ValueError(f"{source}:{line_number}: expected 38 fields, got {len(fields)}")
        image_1, image_2 = fields[:2]
        height_1, width_1 = sizes.get(image_1)
        height_2, width_2 = sizes.get(image_2)
        pair_rows.append(
            {
                "image_1": image_1,
                "image_2": image_2,
                "image_1_height": height_1,
                "image_1_width": width_1,
                "image_2_height": height_2,
                "image_2_width": width_2,
            }
        )
        intrinsics_1 = np.asarray(fields[4:13], dtype=np.float64).reshape(3, 3)
        intrinsics_2 = np.asarray(fields[13:22], dtype=np.float64).reshape(3, 3)
        transform = np.asarray(fields[22:38], dtype=np.float64).reshape(4, 4)
        if invert_transform:
            rotation_inverse = transform[:3, :3].T
            translation_inverse = -rotation_inverse @ transform[:3, 3]
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = rotation_inverse
            transform[:3, 3] = translation_inverse
        sx1, sy1 = _resize_scales(height_1, width_1, downscale_only=False)
        sx2, sy2 = _resize_scales(height_2, width_2, downscale_only=False)
        quaternion = _rotation_matrix_to_quaternion_wxyz(transform[:3, :3])
        translation = transform[:3, 3]
        pose_rows.append(
            {
                "pair_idx": len(pose_rows) + 1,
                "fx1": _format_float(intrinsics_1[0, 0] * sx1),
                "cx1": _format_float(intrinsics_1[0, 2] * sx1),
                "fy1": _format_float(intrinsics_1[1, 1] * sy1),
                "cy1": _format_float(intrinsics_1[1, 2] * sy1),
                "fx2": _format_float(intrinsics_2[0, 0] * sx2),
                "cx2": _format_float(intrinsics_2[0, 2] * sx2),
                "fy2": _format_float(intrinsics_2[1, 1] * sy2),
                "cy2": _format_float(intrinsics_2[1, 2] * sy2),
                **{
                    key: _format_float(value)
                    for key, value in zip(("qw", "qx", "qy", "qz"), quaternion)
                },
                **{
                    key: _format_float(value)
                    for key, value in zip(("tx", "ty", "tz"), translation)
                },
            }
        )
    return pair_rows, pose_rows


class NaviAnnotationIndex:
    """Lazy NAVI annotation lookup keyed by pair-list image path."""

    def __init__(self, navi_root: Path) -> None:
        self.navi_root = navi_root.resolve()
        self._scenes: dict[Path, dict[str, dict[str, Any]]] = {}

    def get(self, image_path: str) -> dict[str, Any]:
        relative = Path(image_path)
        annotation_path = self.navi_root / relative.parent.parent / "annotations.json"
        if annotation_path not in self._scenes:
            records = json.loads(annotation_path.read_text(encoding="utf-8"))
            indexed = {str(record["filename"]): record for record in records}
            if len(indexed) != len(records):
                raise ValueError(f"Duplicate NAVI annotations in {annotation_path}")
            self._scenes[annotation_path] = indexed
        try:
            return self._scenes[annotation_path][relative.name]
        except KeyError as exc:
            raise KeyError(f"No NAVI annotation for {image_path}") from exc


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _navi_pose_row(
    pair_index: int,
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    q1 = np.asarray(left["camera"]["q"], dtype=np.float64)
    q2 = np.asarray(right["camera"]["q"], dtype=np.float64)
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)
    q12 = _quaternion_multiply(q2, q1 * np.array([1.0, -1.0, -1.0, -1.0]))
    q12 /= np.linalg.norm(q12)
    first_nonzero = np.flatnonzero(np.abs(q12) > 1e-15)
    if first_nonzero.size and q12[int(first_nonzero[0])] < 0.0:
        q12 *= -1.0
    rotation12 = _quaternion_to_rotation(q12)
    translation1 = np.asarray(left["camera"]["t"], dtype=np.float64)
    translation2 = np.asarray(right["camera"]["t"], dtype=np.float64)
    translation12 = translation2 - rotation12 @ translation1

    def intrinsics(annotation: dict[str, Any]) -> tuple[float, int, int]:
        height, width = (int(value) for value in annotation["image_size"])
        scale = LONG_EDGE / max(height, width)
        focal = float(annotation["camera"]["focal_length"]) * scale
        return focal, int(width * scale) // 2, int(height * scale) // 2

    focal1, cx1, cy1 = intrinsics(left)
    focal2, cx2, cy2 = intrinsics(right)
    return {
        "pair_idx": pair_index,
        "fx1": _format_float(focal1),
        "cx1": cx1,
        "fy1": _format_float(focal1),
        "cy1": cy1,
        "fx2": _format_float(focal2),
        "cx2": cx2,
        "fy2": _format_float(focal2),
        "cy2": cy2,
        **{
            key: _format_float(value)
            for key, value in zip(("qw", "qx", "qy", "qz"), q12)
        },
        **{
            key: _format_float(value)
            for key, value in zip(("tx", "ty", "tz"), translation12)
        },
    }


def _prepare_navi_pair_file(
    pair_file: Path,
    navi_root: Path,
    annotations: NaviAnnotationIndex,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sizes = ImageSizeCache(navi_root)
    pair_rows: list[dict[str, Any]] = []
    pose_rows: list[dict[str, Any]] = []
    with pair_file.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not {"image_1", "image_2"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"Missing image columns in {pair_file}")
        for pair_index, row in enumerate(reader, start=1):
            image_1 = row["image_1"].strip()
            image_2 = row["image_2"].strip()
            height_1, width_1 = sizes.get(image_1)
            height_2, width_2 = sizes.get(image_2)
            for side, actual in ((1, (height_1, width_1)), (2, (height_2, width_2))):
                csv_height = row.get(f"image_{side}_height")
                csv_width = row.get(f"image_{side}_width")
                if csv_height and csv_width and (int(csv_height), int(csv_width)) != actual:
                    raise ValueError(
                        f"{pair_file}: stored dimensions for {row[f'image_{side}']} "
                        f"do not match the image"
                    )
            pair_rows.append(
                {
                    "image_1": image_1,
                    "image_2": image_2,
                    "image_1_height": height_1,
                    "image_1_width": width_1,
                    "image_2_height": height_2,
                    "image_2_width": width_2,
                }
            )
            pose_rows.append(
                _navi_pose_row(
                    pair_index,
                    annotations.get(image_1),
                    annotations.get(image_2),
                )
            )
    return pair_rows, pose_rows


def _distortion_string(values: np.ndarray) -> str:
    coefficients = np.asarray(values, dtype=np.float64).reshape(-1)
    if coefficients.size < 8:
        coefficients = np.pad(coefficients, (0, 8 - coefficients.size))
    return ",".join(_format_float(value) for value in coefficients[:8])


def _prepare_metu_split(
    *,
    index_root: Path,
    image_root: Path,
    split_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    sizes = ImageSizeCache(image_root)
    pair_rows: list[dict[str, Any]] = []
    pose_rows: list[dict[str, Any]] = []
    sources = sorted(
        index_root.glob(f"{split_name}_scene_*.npz"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    if not sources:
        raise FileNotFoundError(f"No {split_name} scene indexes in {index_root}")
    for source in sources:
        with np.load(source, allow_pickle=True) as data:
            image_paths = data["image_paths"]
            intrinsics = np.asarray(data["intrinsics"], dtype=np.float64)
            distortion = np.asarray(data["distortion_coefs"], dtype=np.float64)
            poses = np.asarray(data["poses"], dtype=np.float64)
            pair_infos = data["pair_infos"]
            for first_raw, second_raw in pair_infos:
                first, second = int(first_raw), int(second_raw)
                image_1 = str(image_paths[first, 0])
                image_2 = str(image_paths[second, 1])
                height_1, width_1 = sizes.get(image_1)
                height_2, width_2 = sizes.get(image_2)
                pair_rows.append(
                    {
                        "image_1": image_1,
                        "image_2": image_2,
                        "image_1_height": height_1,
                        "image_1_width": width_1,
                        "image_2_height": height_2,
                        "image_2_width": width_2,
                    }
                )
                sx1, sy1 = _resize_scales(
                    height_1, width_1, downscale_only=True
                )
                sx2, sy2 = _resize_scales(
                    height_2, width_2, downscale_only=True
                )
                relative_pose = poses[second] @ np.linalg.inv(poses[first])
                quaternion = _rotation_matrix_to_quaternion_wxyz(
                    relative_pose[:3, :3]
                )
                translation = relative_pose[:3, 3]
                camera_1 = intrinsics[first, 0]
                camera_2 = intrinsics[second, 1]
                pose_rows.append(
                    {
                        "pair_idx": len(pose_rows) + 1,
                        "fx1": _format_float(camera_1[0, 0] * sx1),
                        "cx1": _format_float(camera_1[0, 2] * sx1),
                        "fy1": _format_float(camera_1[1, 1] * sy1),
                        "cy1": _format_float(camera_1[1, 2] * sy1),
                        "fx2": _format_float(camera_2[0, 0] * sx2),
                        "cx2": _format_float(camera_2[0, 2] * sx2),
                        "fy2": _format_float(camera_2[1, 1] * sy2),
                        "cy2": _format_float(camera_2[1, 2] * sy2),
                        **{
                            key: _format_float(value)
                            for key, value in zip(
                                ("qw", "qx", "qy", "qz"), quaternion
                            )
                        },
                        **{
                            key: _format_float(value)
                            for key, value in zip(
                                ("tx", "ty", "tz"), translation
                            )
                        },
                        "dist0_coeffs": _distortion_string(
                            distortion[first, 0]
                        ),
                        "dist1_coeffs": _distortion_string(
                            distortion[second, 1]
                        ),
                    }
                )
    return pair_rows, pose_rows, sources


def _prepare_all(
    rss_root: Path,
    output_root: Path,
    *,
    data_root: Path | None = None,
    existing: str,
    dataset_labels: Sequence[str] = (),
) -> list[PreparedSubset]:
    data_root = data_root if data_root is not None else rss_root / "data"
    prepared: list[PreparedSubset] = []
    requested = set(dataset_labels)

    def selected(label: str) -> bool:
        return not requested or label in requested

    fixed_datasets = (
        (
            "ScanNet",
            "scannet_resized",
            "scannet_test_pairs_with_gt",
            data_root / "scannet",
            data_root / "scannet" / "pairs_scannet_test_pairs_with_gt.txt",
            False,
        ),
        (
            "MegaDepth",
            "megadepth_resized",
            "test_1500",
            data_root / "megadepth",
            data_root / "megadepth" / "megadepth_1500_scales" / "all_pairs_with_gt.txt",
            False,
        ),
    )
    for (
        dataset_label,
        root_name,
        subset_name,
        image_root,
        source,
        invert_transform,
    ) in fixed_datasets:
        if not selected(dataset_label):
            continue
        observed_source_sha256 = _sha256(source)
        expected_source_sha256 = PAPER_PAIR_METADATA_SHA256[dataset_label]
        if observed_source_sha256 != expected_source_sha256:
            raise ValueError(
                f"{source}: {dataset_label} pair/GT metadata SHA-256 mismatch; "
                f"expected {expected_source_sha256}, got {observed_source_sha256}"
            )
        pair_rows, pose_rows = _prepare_pairs_with_gt(
            source=source,
            image_root=image_root,
            invert_transform=invert_transform,
        )
        prepared.append(
            _write_subset(
                output_root=output_root,
                dataset_label=dataset_label,
                root_name=root_name,
                subset_name=subset_name,
                image_root=image_root,
                pair_rows=pair_rows,
                pose_rows=pose_rows,
                pose_fields=POSE_FIELDS,
                downscale_only=False,
                source_files=(source,),
                existing=existing,
            )
        )

    navi_root = data_root / "navi"
    navi_splits_root = rss_root / "data" / "splits" / "navi" / "estimation"
    annotations = NaviAnnotationIndex(navi_root)
    navi_specs: list[tuple[str, str, Path]] = [
        ("NAVI-Multi", "NAVI_resized", navi_splits_root),
        ("NAVI-Wild", "NAVI_wild", navi_splits_root),
    ]
    for dataset_label, root_name, splits_root in navi_specs:
        if not selected(dataset_label):
            continue
        prefix = "multiview" if dataset_label == "NAVI-Multi" else "wildset"
        for angular_bin in ("0-40", "40-80", "80-120"):
            subset_name = f"{prefix}_{angular_bin}"
            source = splits_root / f"pairs_{subset_name}.csv"
            pair_rows, pose_rows = _prepare_navi_pair_file(
                source, navi_root, annotations
            )
            prepared.append(
                _write_subset(
                    output_root=output_root,
                    dataset_label=dataset_label,
                    root_name=root_name,
                    subset_name=subset_name,
                    image_root=navi_root,
                    pair_rows=pair_rows,
                    pose_rows=pose_rows,
                    pose_fields=POSE_FIELDS,
                    downscale_only=False,
                    source_files=(source,),
                    existing=existing,
                )
            )

    metu_root = data_root / "METU_VisTIR"
    metu_index = metu_root / "index" / "scene_info_test"
    for dataset_label, subset_name in (
        ("METU-CC", "cloudy_cloudy"),
        ("METU-CS", "cloudy_sunny"),
    ):
        if not selected(dataset_label):
            continue
        pair_rows, pose_rows, sources = _prepare_metu_split(
            index_root=metu_index,
            image_root=metu_root,
            split_name=subset_name,
        )
        prepared.append(
            _write_subset(
                output_root=output_root,
                dataset_label=dataset_label,
                root_name="METU_VisTIR_resized",
                subset_name=subset_name,
                image_root=metu_root,
                pair_rows=pair_rows,
                pose_rows=pose_rows,
                pose_fields=METU_POSE_FIELDS,
                downscale_only=True,
                source_files=sources,
                existing=existing,
            )
        )
    return prepared


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rss-root",
        type=Path,
        default=repository_root,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repository_root / "artifacts" / "matching_estimation_results",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Dataset root containing scannet/, megadepth/, navi/, METU_VisTIR/, "
            "and the benchmark pair/GT metadata. The tracked NAVI splits are read "
            "from <rss-root>/data/splits. Defaults to <rss-root>/data."
        ),
    )
    parser.add_argument(
        "--existing",
        choices=("error", "validate", "overwrite"),
        default="validate",
        help=(
            "Existing-subset policy. `validate` resumes by verifying every "
            "retained CSV and manifest against current raw inputs (default)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Backward-compatible alias for --existing overwrite.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=("ScanNet", "MegaDepth", "NAVI-Multi", "NAVI-Wild", "METU-CC", "METU-CS"),
        default=[],
        help="Prepare only these labels; the experiment manifest retains other rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rss_root = args.rss_root.resolve()
    output_root = args.output_root.resolve()
    if args.overwrite and args.existing != "validate":
        raise ValueError("Use either --overwrite or --existing, not both")
    existing = "overwrite" if args.overwrite else args.existing
    prepared = _prepare_all(
        rss_root,
        output_root,
        data_root=(args.data_root.resolve() if args.data_root is not None else None),
        existing=existing,
        dataset_labels=args.datasets,
    )
    if args.datasets:
        previous_path = output_root / "experiment_inputs.json"
        previous_rows: list[dict[str, Any]] = []
        if previous_path.is_file():
            previous_payload = json.loads(previous_path.read_text(encoding="utf-8"))
            previous_rows = list(previous_payload.get("subsets", []))
        replacements = {
            (subset.root_name, subset.subset_name): subset for subset in prepared
        }
        retained = [
            row
            for row in previous_rows
            if (str(row.get("root_name")), str(row.get("subset_name")))
            not in replacements
        ]
    else:
        retained = []
    manifest = {
        "schema_version": 1,
        "experiment": "pose_estimation_inputs_v1",
        "long_edge": LONG_EDGE,
        "subsets": retained + [
            {
                "dataset_label": subset.dataset_label,
                "root_name": subset.root_name,
                "subset_name": subset.subset_name,
                "image_root": str(subset.image_root),
                "pair_file": str(subset.pair_file),
                "pose_file": str(subset.pose_file),
                "downscale_only": subset.downscale_only,
                "pair_count": subset.pair_count,
            }
            for subset in prepared
        ],
    }
    manifest_path = output_root / "experiment_inputs.json"
    _write_json_atomic(manifest_path, manifest)
    total_pairs = sum(subset.pair_count for subset in prepared)
    print(
        f"Prepared {len(prepared)} subsets / {total_pairs} pairs under {output_root}"
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
