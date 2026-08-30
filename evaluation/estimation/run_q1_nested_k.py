#!/usr/bin/env python3
"""Run the main-paper Q1 fixed-proposal nested-K experiment.

The three stages are deliberately small and explicit:

1. ``derive`` filters one uncapped progressive-K5 graph into nested E1..E5.
2. ``estimate`` runs HCM->MCM with proposals fixed to E1 for five seeds.
3. ``summarize`` computes paired Pose-AUC deltas and graph-size statistics.

No GMS, association cap, or estimator cap is used.  Every estimator cell uses
exactly 100,000 iterations; EK is used for scoring/refinement/reranking while
``proposal_max_k=1`` fixes minimal samples to E1.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from evaluation.estimation.metrics import (
    error_auc,
    expected_pose_pair_indices,
    pose_error_vector_for_indices,
    validated_pose_result_pair_ids,
)


DATASETS = ("ScanNet", "MegaDepth", "NAVI-Multi", "NAVI-Wild", "METU-CC", "METU-CS")
KS = (1, 2, 3, 4, 5)
SEEDS = (0, 1, 2, 3, 4)
THRESHOLDS = (5.0, 10.0, 20.0)
NESTED_METHOD = "protocol_b_nested"
MODE = "HCM_MC"
ITERATIONS = 100_000
PAPER_DINOV3_WEIGHTS_ID = (
    "sha256:8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"
)


@dataclass(frozen=True)
class Subset:
    dataset: str
    root_name: str
    subset_name: str
    pair_count: int


@dataclass(frozen=True)
class Task:
    subset: Subset
    k: int
    seed: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_subsets(path: Path, datasets: Sequence[str]) -> list[Subset]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requested = set(datasets)
    subsets = [
        Subset(
            dataset=str(row["dataset_label"]),
            root_name=str(row["root_name"]),
            subset_name=str(row["subset_name"]),
            pair_count=int(row["pair_count"]),
        )
        for row in payload["subsets"]
        if not requested or str(row["dataset_label"]) in requested
    ]
    if not subsets:
        raise ValueError("No subsets match --datasets")
    missing = requested - {subset.dataset for subset in subsets}
    if missing:
        raise ValueError(f"Input manifest is missing datasets: {sorted(missing)}")
    identities = [(subset.root_name, subset.subset_name) for subset in subsets]
    if len(identities) != len(set(identities)):
        raise ValueError("Input manifest contains duplicate subset identities")
    return subsets


def _subset_root(results_root: Path, subset: Subset) -> Path:
    return results_root / subset.root_name / subset.subset_name


def _source_dir(args: argparse.Namespace, subset: Subset) -> Path:
    return (
        _subset_root(args.results_root, subset)
        / args.source_method
        / f"layer{args.layer}"
        / f"debias_svd{args.debias_rank}"
        / "progressive_k5"
    )


def _nested_root(results_root: Path, subset: Subset) -> Path:
    return _subset_root(results_root, subset) / NESTED_METHOD


def _matching_names(pair_count: int) -> set[str]:
    width = max(3, len(str(pair_count)))
    return {f"matching_{index:0{width}d}.csv" for index in range(1, pair_count + 1)}


def _source_manifest(
    source_dir: Path,
    subset: Subset,
    *,
    layer: int,
    debias_rank: int,
) -> tuple[Path, dict[str, Any]]:
    path = source_dir / "association_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "pair_count": subset.pair_count,
        "model_name": "dinov3_vitl16",
        "layer": layer,
        "debias_rank": debias_rank,
        "progressive_max_k": 5,
        "association_upperbound": 0,
        "weights_id": PAPER_DINOV3_WEIGHTS_ID,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{path}: Q1 requires an uncapped K5 source: {mismatches}")
    return path, payload


def _read_associations(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    required = {"left_idx", "right_idx", "similarity", "k"}
    if not required.issubset(fields):
        raise ValueError(f"{path}: missing columns {sorted(required - set(fields))}")
    ranks: list[int] = []
    edges: set[tuple[int, int]] = set()
    for row in rows:
        rank = int(row["k"])
        similarity = float(row["similarity"])
        edge = (int(row["left_idx"]), int(row["right_idx"]))
        if rank not in KS or not np.isfinite(similarity):
            raise ValueError(f"{path}: invalid rank or similarity")
        if edge in edges:
            raise ValueError(f"{path}: duplicate association edge {edge}")
        ranks.append(rank)
        edges.add(edge)
    if ranks != sorted(ranks):
        raise ValueError(f"{path}: association rows are not ordered by first K")
    return fields, rows


def _render_csv(fields: Sequence[str], rows: Sequence[dict[str, str]]) -> str:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _write_resumable(path: Path, content: str, existing: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") == content:
            return
        if existing != "overwrite":
            raise ValueError(f"{path}: retained nested association differs")
    elif existing == "error" and path.exists():
        raise FileExistsError(path)
    _atomic_text(path, content)


def _derive_subset(args: argparse.Namespace, subset: Subset) -> dict[str, Any]:
    source_dir = _source_dir(args, subset)
    manifest_path, source_manifest = _source_manifest(
        source_dir,
        subset,
        layer=args.layer,
        debias_rank=args.debias_rank,
    )
    actual = {path.name for path in source_dir.glob("matching_*.csv")}
    expected = _matching_names(subset.pair_count)
    if actual != expected:
        raise ValueError(
            f"{source_dir}: matching rectangle differs; "
            f"missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}"
        )
    output_root = _nested_root(args.results_root, subset)
    audit_rows: list[dict[str, Any]] = []
    for name in sorted(expected):
        fields, rows = _read_associations(source_dir / name)
        pair_idx = int(Path(name).stem.split("_")[-1])
        previous: set[tuple[int, int]] = set()
        for k in KS:
            selected = [row for row in rows if int(row["k"]) <= k]
            edges = {(int(row["left_idx"]), int(row["right_idx"])) for row in selected}
            if not previous.issubset(edges):
                raise AssertionError(f"Internal E_K nesting failure for {name}/K={k}")
            target = output_root / f"progressive_k{k}" / name
            _write_resumable(target, _render_csv(fields, selected), args.existing)
            audit_rows.append(
                {
                    "dataset": subset.dataset,
                    "root_name": subset.root_name,
                    "subset_name": subset.subset_name,
                    "pair_idx": pair_idx,
                    "k": k,
                    "edge_count": len(selected),
                    "added_edges": len(edges - previous),
                }
            )
            previous = edges
    for k in KS:
        association_manifest = {
            **source_manifest,
            "progressive_max_k": k,
            "association_upperbound": 0,
            "derived_from": str(manifest_path.resolve()),
            "derived_filter": f"first_k <= {k}",
        }
        _write_resumable(
            output_root / f"progressive_k{k}" / "association_manifest.json",
            json.dumps(
                association_manifest, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            args.existing,
        )
    audit = pd.DataFrame(audit_rows)
    _atomic_text(output_root / "association_audit.csv", audit.to_csv(index=False))
    payload = {
        "schema_version": 1,
        "artifact": "q1_nested_k_associations",
        "dataset": subset.dataset,
        "root_name": subset.root_name,
        "subset_name": subset.subset_name,
        "pair_count": subset.pair_count,
        "source_dir": str(source_dir.resolve()),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": _sha256(manifest_path),
        "model_name": source_manifest.get("model_name"),
        "layer": args.layer,
        "debias_rank": args.debias_rank,
        "ks": list(KS),
        "gms": False,
        "association_cap": 0,
        "proposal_graph": "E1",
        "scoring_graph": "EK",
    }
    _atomic_text(
        output_root / "nested_k_manifest.json",
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return payload


def run_derive(args: argparse.Namespace) -> None:
    subsets = _load_subsets(args.input_manifest, args.datasets)
    for index, subset in enumerate(subsets, 1):
        _derive_subset(args, subset)
        print(f"[{index}/{len(subsets)}] derived {subset.root_name}/{subset.subset_name}")


def _result_filename(seed: int) -> str:
    return f"HCM_MC_q1_pose_seed{seed}_iter100000_proposal_k_1_q_ub_0.30.csv"


def _config_text(results_root: Path, task: Task) -> str:
    return "\n".join(
        (
            f"matching_result_root={results_root / task.subset.root_name}",
            f"datasets={task.subset.subset_name}",
            f"method={NESTED_METHOD}/progressive_k{task.k}",
            "ransac_mode=HCM_MC",
            "top_n_candidates=100",
            "pool_dedup_deg=0.0",
            "write_candidate_traces=false",
            "q_ub=0.3",
            "m2m_delta=0.01",
            "max_error_px=1.0",
            "similarity_threshold=0.0",
            "max_matching_num=0",
            "proposal_max_k=1",
            "min_iterations=100000",
            "max_iterations=100000",
            "success_prob=0.9999",
            f"seed={task.seed}",
            "tangent_sampson=false",
            "init_with_gt=false",
            "skip_existing_pairs=true",
            "allow_unbound_pose=false",
            "ransac_times=1",
            f"output_csv=q1_pose_seed{task.seed}_iter100000.csv",
            "",
        )
    )


def _result_pair_ids(path: Path, pair_count: int) -> tuple[int, ...]:
    return validated_pose_result_pair_ids(path, range(1, pair_count + 1))


def _estimate_one(args: argparse.Namespace, task: Task) -> str:
    nested_root = _nested_root(args.results_root, task.subset)
    manifest = nested_root / "nested_k_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    method_dir = nested_root / f"progressive_k{task.k}"
    config = (
        args.work_root
        / "configs"
        / task.subset.root_name
        / task.subset.subset_name
        / f"k{task.k}"
        / f"seed{task.seed}.cfg"
    )
    requested = _config_text(args.results_root, task)
    result = method_dir / _result_filename(task.seed)
    if config.is_file() and config.read_text(encoding="utf-8") != requested:
        raise ValueError(f"{config}: retained Q1 configuration differs")
    if not config.is_file():
        if result.is_file():
            raise FileNotFoundError(f"{config}: result exists without its config")
        _atomic_text(config, requested)
    log = config.with_suffix(".log")
    expected_ids = set(range(1, task.subset.pair_count + 1))
    if (
        result.is_file()
        and set(_result_pair_ids(result, task.subset.pair_count)) == expected_ids
        and log.is_file()
    ):
        return f"validated {task.subset.root_name}/{task.subset.subset_name}/k{task.k}/seed{task.seed}"
    completed = subprocess.run(
        (str(args.runner_binary), str(config)),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _atomic_text(log, completed.stdout)
    if completed.returncode:
        raise RuntimeError(
            f"Q1 estimator failed for {task}:\n{completed.stdout[-2000:]}"
        )
    if (
        not result.is_file()
        or set(_result_pair_ids(result, task.subset.pair_count)) != expected_ids
    ):
        raise RuntimeError(f"{result}: incomplete Q1 result")
    return f"ran {task.subset.root_name}/{task.subset.subset_name}/k{task.k}/seed{task.seed}"


def _run_index_rows(args: argparse.Namespace, tasks: Sequence[Task]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        nested_root = _nested_root(args.results_root, task.subset)
        rows.append(
            {
                "dataset": task.subset.dataset,
                "root_name": task.subset.root_name,
                "subset_name": task.subset.subset_name,
                "k": task.k,
                "seed": task.seed,
                "result_csv": str((nested_root / f"progressive_k{task.k}" / _result_filename(task.seed)).resolve()),
                "nested_manifest": str((nested_root / "nested_k_manifest.json").resolve()),
                "protocol": "B",
                "mode": MODE,
                "gms": False,
                "association_cap": 0,
                "estimator_cap": 0,
                "final_eligible": True,
                "q": 0.3,
                "delta": 0.01,
                "top_n_candidates": 100,
                "max_error_px": 1.0,
                "proposal_max_k": 1,
                "proposal_graph": "E1",
                "scoring_graph": "EK",
                "fixed_iterations": True,
                "min_iterations": ITERATIONS,
                "max_iterations": ITERATIONS,
                "ransac_times": 1,
            }
        )
    return rows


def run_estimate(args: argparse.Namespace) -> None:
    subsets = _load_subsets(args.input_manifest, args.datasets)
    tasks = [Task(subset, k, seed) for subset in subsets for k in KS for seed in SEEDS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_estimate_one, args, task) for task in tasks]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            print(f"[{index}/{len(futures)}] {future.result()}", flush=True)
    rows = _run_index_rows(args, tasks)
    args.work_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.work_root / "run_index.csv", index=False)


def run_summarize(args: argparse.Namespace) -> None:
    run_index = pd.read_csv(args.run_index)
    required = {"dataset", "root_name", "subset_name", "k", "seed", "result_csv", "nested_manifest"}
    missing = required - set(run_index.columns)
    if missing:
        raise ValueError(f"{args.run_index}: missing columns {sorted(missing)}")
    identity_columns = ["root_name", "subset_name", "k", "seed"]
    if run_index.duplicated(identity_columns).any():
        raise ValueError("Q1 run index contains duplicate subset/K/seed rows")
    if set(run_index["dataset"].astype(str)) != set(DATASETS):
        raise ValueError("Q1 run index must cover the canonical six datasets")
    observed = set(zip(run_index["k"].astype(int), run_index["seed"].astype(int)))
    if observed != {(k, seed) for k in KS for seed in SEEDS}:
        raise ValueError("Q1 run index does not contain the complete K x seed grid")
    for identity, rows in run_index.groupby(["root_name", "subset_name"], sort=False):
        cells = set(zip(rows["k"].astype(int), rows["seed"].astype(int)))
        if cells != {(k, seed) for k in KS for seed in SEEDS}:
            raise ValueError(f"Q1 subset {identity} has an incomplete K x seed grid")

    grouped_errors: dict[tuple[str, int, int], list[np.ndarray]] = {}
    pair_totals: dict[tuple[str, int, int], int] = {}
    for row in run_index.itertuples(index=False):
        result = Path(str(row.result_csv)).resolve()
        subset_root = args.results_root / str(row.root_name) / str(row.subset_name)
        pose = subset_root / "pose_intrinsics.csv"
        expected = expected_pose_pair_indices(pose)
        validated_pose_result_pair_ids(
            result,
            expected,
            require_complete=True,
        )
        errors = pose_error_vector_for_indices(result, expected)
        key = (str(row.dataset), int(row.k), int(row.seed))
        grouped_errors.setdefault(key, []).append(errors)
        pair_totals[key] = pair_totals.get(key, 0) + len(expected)

    seed_rows: list[dict[str, Any]] = []
    for (dataset, k, seed), vectors in sorted(grouped_errors.items()):
        aucs = error_auc(np.concatenate(vectors), THRESHOLDS)
        for threshold in THRESHOLDS:
            seed_rows.append(
                {
                    "dataset": dataset,
                    "k": k,
                    "threshold_deg": threshold,
                    "seed": seed,
                    "auc": aucs[f"auc@{threshold:g}"],
                    "pairs": pair_totals[(dataset, k, seed)],
                }
            )
    seed_frame = pd.DataFrame(seed_rows)
    baseline = seed_frame[seed_frame["k"] == 1][
        ["dataset", "threshold_deg", "seed", "auc"]
    ].rename(columns={"auc": "baseline_auc"})
    paired = seed_frame.merge(
        baseline, on=["dataset", "threshold_deg", "seed"], validate="many_to_one"
    )
    paired["delta_auc"] = paired["auc"] - paired["baseline_auc"]
    summary = (
        paired.groupby(["dataset", "k", "threshold_deg"], sort=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_seed_std=("auc", lambda values: values.std(ddof=1)),
            delta_auc=("delta_auc", "mean"),
            delta_seed_min=("delta_auc", "min"),
            delta_seed_max=("delta_auc", "max"),
            seeds=("seed", "size"),
            pairs=("pairs", "first"),
        )
        .reset_index()
    )

    audits = []
    for manifest in sorted({Path(str(value)).resolve() for value in run_index["nested_manifest"]}):
        audit = manifest.parent / "association_audit.csv"
        audits.append(pd.read_csv(audit))
    edge_counts = (
        pd.concat(audits, ignore_index=True)
        .groupby(["dataset", "k"], sort=False)
        .agg(pairs=("pair_idx", "size"), edge_median=("edge_count", "median"))
        .reset_index()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = args.output_dir / "q1_auc_per_seed.csv"
    summary_path = args.output_dir / "q1_auc_summary.csv"
    edge_path = args.output_dir / "q1_edge_counts.csv"
    seed_frame.to_csv(seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    edge_counts.to_csv(edge_path, index=False)
    payload = {
        "schema_version": 1,
        "artifact": "q1_fixed_proposal_nested_k_summary",
        "datasets": list(DATASETS),
        "ks": list(KS),
        "seeds": list(SEEDS),
        "proposal_graph": "E1",
        "scoring_graph": "EK",
        "fixed_iterations": ITERATIONS,
        "outputs": {
            "per_seed_csv": str(seed_path.resolve()),
            "summary_csv": str(summary_path.resolve()),
            "edge_counts_csv": str(edge_path.resolve()),
        },
    }
    _atomic_text(
        args.output_dir / "q1_summary.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("derive", "estimate", "summarize", "all"))
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--source-method", default="combined_interpolate_v3")
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--debias-rank", type=int, default=200)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument(
        "--runner-binary",
        type=Path,
        default=repository_root / "build/m2m_loransac/m2m_loransac_runner",
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--run-index", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--existing", choices=("validate", "overwrite"), default="validate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.results_root = args.results_root.expanduser().resolve()
    args.input_manifest = (
        args.input_manifest.expanduser().resolve()
        if args.input_manifest is not None
        else args.results_root / "experiment_inputs.json"
    )
    args.work_root = args.work_root.expanduser().resolve()
    args.runner_binary = args.runner_binary.expanduser().resolve()
    args.run_index = (
        args.run_index.expanduser().resolve()
        if args.run_index is not None
        else args.work_root / "run_index.csv"
    )
    args.output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.work_root / "summary"
    )
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.command in {"derive", "all"}:
        run_derive(args)
    if args.command in {"estimate", "all"}:
        if not args.runner_binary.is_file():
            raise FileNotFoundError(args.runner_binary)
        run_estimate(args)
    if args.command in {"summarize", "all"}:
        run_summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
