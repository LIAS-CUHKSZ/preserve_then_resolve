#!/usr/bin/env python3
"""Generate normalized NAVI pose/intrinsics CSVs for the bundled splits.

NAVI annotations store an object-to-camera transform for each image.  For a
pair of annotations ``(R1, t1)`` and ``(R2, t2)``, this script writes the
camera-1-to-camera-2 transform

    R12 = R2 R1^T,  t12 = t2 - R12 t1.

Intrinsics follow the experiment's 1024-pixel long-edge resize policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "artifacts" / "matching_estimation_results" / "NAVI_resized"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from dino_m2m.pairs import PairRecord, read_pairs  # noqa: E402


HEADER = (
    "pair_idx",
    "fx1",
    "fy1",
    "cx1",
    "cy1",
    "fx2",
    "fy2",
    "cx2",
    "cy2",
    "qw",
    "qx",
    "qy",
    "qz",
    "tx",
    "ty",
    "tz",
)
PAIR_BINDING_FILENAME = "pose_intrinsics_manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pair_identity_sha256(records: list[PairRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            f"{record.pair_index}\t{record.left_rel.as_posix()}\t"
            f"{record.right_rel.as_posix()}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _normalized_quaternion(values: Any) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError(f"Invalid wxyz quaternion: {values!r}")
    norm = float(np.linalg.norm(quaternion))
    if norm == 0.0:
        raise ValueError("Quaternion must be non-zero")
    return quaternion / norm


def _multiply_quaternions(left: np.ndarray, right: np.ndarray) -> np.ndarray:
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


def _canonical_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = _normalized_quaternion(quaternion)
    nonzero = np.flatnonzero(np.abs(quaternion) > 1e-15)
    if nonzero.size and quaternion[int(nonzero[0])] < 0.0:
        quaternion = -quaternion
    return quaternion


def _rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalized_quaternion(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _relative_pose(camera1: dict[str, Any], camera2: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    q1 = _normalized_quaternion(camera1["q"])
    q2 = _normalized_quaternion(camera2["q"])
    q12 = _canonical_quaternion(_multiply_quaternions(q2, q1 * np.array([1.0, -1.0, -1.0, -1.0])))
    rotation12 = _rotation_matrix(q12)
    translation1 = np.asarray(camera1["t"], dtype=np.float64)
    translation2 = np.asarray(camera2["t"], dtype=np.float64)
    if translation1.shape != (3,) or translation2.shape != (3,):
        raise ValueError("Camera translations must contain three values")
    translation12 = translation2 - rotation12 @ translation1
    return q12, translation12


def _scaled_intrinsics(annotation: dict[str, Any], long_edge: int) -> tuple[float, float, int, int]:
    height, width = (int(value) for value in annotation["image_size"])
    if height <= 0 or width <= 0 or long_edge <= 0:
        raise ValueError("Image dimensions and long edge must be positive")
    scale = long_edge / max(height, width)
    resized_width = int(width * scale)
    resized_height = int(height * scale)
    focal = float(annotation["camera"]["focal_length"]) * scale
    return focal, focal, resized_width // 2, resized_height // 2


class AnnotationIndex:
    def __init__(self, navi_root: Path) -> None:
        self.navi_root = navi_root
        self._scenes: dict[Path, dict[str, dict[str, Any]]] = {}

    def get(self, relative_image: Path) -> dict[str, Any]:
        annotation_path = self.navi_root / relative_image.parent.parent / "annotations.json"
        if annotation_path not in self._scenes:
            rows = json.loads(annotation_path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError(f"Expected an annotation list: {annotation_path}")
            indexed = {str(row["filename"]): row for row in rows}
            if len(indexed) != len(rows):
                raise ValueError(f"Duplicate filenames in {annotation_path}")
            self._scenes[annotation_path] = indexed
        try:
            return self._scenes[annotation_path][relative_image.name]
        except KeyError as exc:
            raise KeyError(f"No annotation for {relative_image}") from exc


def _format_float(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"Non-finite output value: {value}")
    return "0" if value == 0.0 else format(value, ".17g")


def _row(record: PairRecord, annotations: AnnotationIndex, long_edge: int) -> list[str | int]:
    left = annotations.get(record.left_rel)
    right = annotations.get(record.right_rel)
    q12, t12 = _relative_pose(left["camera"], right["camera"])
    fx1, fy1, cx1, cy1 = _scaled_intrinsics(left, long_edge)
    fx2, fy2, cx2, cy2 = _scaled_intrinsics(right, long_edge)
    return [
        record.pair_index,
        _format_float(fx1),
        _format_float(fy1),
        cx1,
        cy1,
        _format_float(fx2),
        _format_float(fy2),
        cx2,
        cy2,
        *(_format_float(value) for value in q12),
        *(_format_float(value) for value in t12),
    ]


def _write_atomic(path: Path, rows: list[list[str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)
        temporary_path = Path(output.name)
    temporary_path.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as output:
        json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
        temporary_path = Path(output.name)
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navi-root", type=Path, required=True, help="Root containing NAVI object directories")
    parser.add_argument("--splits-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--family",
        choices=("multiview", "wildset"),
        default="multiview",
        help="Pair-list family to process (default: multiview).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Destination for per-subset artifact directories (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument("--long-edge", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pair_files = sorted(args.splits_dir.glob(f"pairs_{args.family}_*.csv"))
    if not pair_files:
        raise FileNotFoundError(
            f"No pairs_{args.family}_*.csv files in {args.splits_dir}"
        )
    annotations = AnnotationIndex(args.navi_root)
    outputs: list[tuple[Path, Path, Path, list[PairRecord], list[list[str | int]]]] = []
    for pair_file in pair_files:
        subset = pair_file.stem.removeprefix("pairs_")
        output_path = args.output_root / subset / "pose_intrinsics.csv"
        binding_path = output_path.parent / PAIR_BINDING_FILENAME
        existing = [path for path in (output_path, binding_path) if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {', '.join(str(path) for path in existing)}; "
                "pass --overwrite"
            )
        records = read_pairs(pair_file)
        outputs.append(
            (
                pair_file,
                output_path,
                binding_path,
                records,
                [_row(record, annotations, args.long_edge) for record in records],
            )
        )
    for pair_file, output_path, binding_path, records, rows in outputs:
        _write_atomic(output_path, rows)
        _write_json_atomic(
            binding_path,
            {
                "schema_version": 1,
                "pair_file": pair_file.name,
                "pair_file_sha256": _sha256(pair_file),
                "pair_identity_sha256": _pair_identity_sha256(records),
                "pair_count": len(records),
                "long_edge": args.long_edge,
                "pose_intrinsics_sha256": _sha256(output_path),
            },
        )
        print(f"wrote {len(rows)} rows: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
