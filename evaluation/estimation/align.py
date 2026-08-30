"""Recompute pose errors from standardized ground-truth poses."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def _quat_wxyz_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    quaternion = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm == 0.0:
        return np.eye(3)
    w, x, y, z = quaternion / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _rotation_error_deg(estimate: np.ndarray, ground_truth: np.ndarray) -> float:
    delta = estimate @ ground_truth.T
    cosine = (float(np.trace(delta)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cosine)))))


def _translation_error_deg(estimate: np.ndarray, ground_truth: np.ndarray) -> float:
    estimate_norm = float(np.linalg.norm(estimate))
    gt_norm = float(np.linalg.norm(ground_truth))
    if estimate_norm == 0.0 or gt_norm == 0.0:
        return float("nan")
    cosine = float(np.dot(estimate, ground_truth) / (estimate_norm * gt_norm))
    error = float(np.degrees(np.arccos(max(-1.0, min(1.0, cosine)))))
    return min(error, 180.0 - error)


def align_estimation_errors(
    results: pd.DataFrame,
    pose_intrinsics: Path,
) -> pd.DataFrame:
    """Overwrite successful-row errors using poses exactly as stored in the CSV."""
    poses = pd.read_csv(pose_intrinsics)
    required = {"pair_idx", "qw", "qx", "qy", "qz", "tx", "ty", "tz"}
    missing = required - set(poses.columns)
    if missing:
        raise ValueError(f"pose_intrinsics CSV is missing columns: {sorted(missing)}")
    if poses["pair_idx"].duplicated().any():
        raise ValueError("pose_intrinsics CSV contains duplicate pair_idx values")
    ground_truth = poses.set_index("pair_idx")[
        ["qw", "qx", "qy", "qz", "tx", "ty", "tz"]
    ]
    # Historical result files may contain this column. It is deliberately
    # ignored and omitted now that all GT files use one convention.
    output = results.drop(columns=["gt_mode_used"], errors="ignore").copy()
    for index, row in output.iterrows():
        if str(row.get("status", "")).lower() != "success":
            continue
        rotation_error = float("nan")
        translation_error = float("nan")
        try:
            pair_idx = int(row["pair_idx"])
            gt = ground_truth.loc[pair_idx]
            estimated_rotation = _quat_wxyz_to_rotation(
                *(float(row[column]) for column in ("q_w", "q_x", "q_y", "q_z"))
            )
            estimated_translation = np.array(
                [float(row[column]) for column in ("t_x", "t_y", "t_z")]
            )
            stored_rotation = _quat_wxyz_to_rotation(
                float(gt["qw"]), float(gt["qx"]), float(gt["qy"]), float(gt["qz"])
            )
            stored_translation = np.array(
                [float(gt["tx"]), float(gt["ty"]), float(gt["tz"])]
            )
            rotation_error = _rotation_error_deg(estimated_rotation, stored_rotation)
            translation_error = _translation_error_deg(
                estimated_translation, stored_translation
            )
        except (KeyError, TypeError, ValueError):
            pass
        output.at[index, "rotation_error_deg"] = rotation_error
        output.at[index, "translation_error_deg"] = translation_error
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_csv", type=Path)
    parser.add_argument("--pose-intrinsics", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = args.output or args.result_csv.with_name(
        f"{args.result_csv.stem}_aligned_as_stored{args.result_csv.suffix}"
    )
    aligned = align_estimation_errors(
        pd.read_csv(args.result_csv), args.pose_intrinsics
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
