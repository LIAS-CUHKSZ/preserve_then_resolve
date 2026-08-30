"""Evaluate one or more estimator result files across the six test sets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaluation.json_utils import strict_json_dumps

from .metrics import (
    DATASETS,
    cost_stats,
    dataset_registry,
    error_auc,
    expected_pose_pair_indices,
    pose_error_vector_for_indices,
)

DEFAULT_METHOD = (
    "combined_interpolate_v3/layer19/debias_svd200/"
    "progressive_k5_GMSm2m_ThrFact_4.0_Gridsz_20_auto_mask"
)


def evaluate_methods(
    *,
    results_root: Path,
    method: str,
    result_files: dict[str, Path],
    datasets: tuple[str, ...] = tuple(DATASETS),
) -> dict[str, dict[str, dict[str, float | int | str] | None]]:
    """Compute per-dataset AUC and cost using the experiment directory layout."""
    registry = dataset_registry()
    unknown = set(datasets) - set(registry)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    report: dict[str, dict[str, dict[str, float | int | str] | None]] = {}
    for method_label, relative_result in result_files.items():
        method_report: dict[str, dict[str, float | int | str] | None] = {}
        for dataset_label in datasets:
            vectors: list[np.ndarray] = []
            subset_costs: list[dict[str, float]] = []
            missing_path: Path | None = None
            pair_count = 0
            for subset in registry[dataset_label]:
                subset_root = results_root / subset.root_name / subset.subset_name
                method_dir = subset_root / method
                result_csv = method_dir / relative_result
                if not result_csv.is_file():
                    missing_path = result_csv
                    break
                pose_csv = subset_root / "pose_intrinsics.csv"
                if not pose_csv.is_file():
                    raise FileNotFoundError(
                        f"Missing fixed-denominator pose metadata: {pose_csv}"
                    )
                expected_indices = expected_pose_pair_indices(
                    pose_csv, require_manifest=subset.require_pose_manifest
                )
                vectors.append(pose_error_vector_for_indices(result_csv, expected_indices))
                subset_costs.append(cost_stats(result_csv))
                pair_count += len(expected_indices)
            if missing_path is not None:
                method_report[dataset_label] = {
                    "status": "missing",
                    "path": str(missing_path),
                }
                continue
            auc = error_auc(np.concatenate(vectors))
            # Retain analyze_all.py's equal weighting of subset-level means.
            cost = {
                name: float(np.mean([entry[name] for entry in subset_costs]))
                for name in ("refinements", "iterations", "running_time_s")
            }
            method_report[dataset_label] = {
                "status": "ok",
                "pairs": pair_count,
                **auc,
                **cost,
            }
        report[method_label] = method_report
    return report


def _parse_result_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("result must be LABEL=RELATIVE_CSV")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("result must be LABEL=RELATIVE_CSV")
    result = Path(path)
    if result.is_absolute():
        raise argparse.ArgumentTypeError("result CSV path must be relative to the method directory")
    return label.strip(), result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report pose AUC@5/10/20 and estimator cost across paper test sets."
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument(
        "--result",
        action="append",
        type=_parse_result_spec,
        required=True,
        metavar="LABEL=RELATIVE_CSV",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=tuple(dataset_registry()),
        help="Evaluate only this dataset label; repeat as needed.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    if argv and argv[0] == "--":
        argv = argv[1:]
    args = build_parser().parse_args(argv)
    result_files = dict(args.result)
    if len(result_files) != len(args.result):
        raise SystemExit("Duplicate --result labels are not allowed")
    report = evaluate_methods(
        results_root=args.results_root,
        method=args.method,
        result_files=result_files,
        datasets=tuple(args.dataset) if args.dataset else tuple(DATASETS),
    )
    rendered = strict_json_dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
