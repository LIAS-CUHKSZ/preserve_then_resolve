#!/usr/bin/env python3
"""Replay and summarize the read-only Protocol-B GT hypothesis probe.

The input is the authenticated run index from the completed controlled-E1
nested-K campaign.  Every replay keeps the original E1 proposal source,
100,000 fixed trials, HCM parameters, top-100 raw-seed pool, LO policy, and
HCM-to-MCM output path.  The runner snapshots the frozen Stage-1 pool, finishes
the normal run, and only then scores the benchmark reference pose once.  That
score is written to a separate sidecar and never enters stopping, refinement,
reranking, or the returned pose.

``run`` is restart-safe and validates every replay against its original result
row, excluding wall-clock durations only.  ``summarize`` requires the complete
leaf x K x seed rectangle and reports pair/seed inclusion rates per dataset and
an equal-dataset macro average.  It does not compute oracle pose AUC or
bootstrap intervals.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


DATASET_ORDER = (
    "ScanNet",
    "MegaDepth",
    "NAVI-Multi",
    "NAVI-Wild",
    "METU-CC",
    "METU-CS",
)
KS = (1, 2, 3, 4, 5)
SEEDS = (0, 1, 2, 3, 4)
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
REQUIRED_RUN_INDEX_COLUMNS = {
    "dataset",
    "root_name",
    "subset_name",
    "k",
    "seed",
    "result_csv",
    "nested_manifest",
    "protocol",
    "mode",
    "gms",
    "association_cap",
    "estimator_cap",
    "final_eligible",
    "q",
    "delta",
    "top_n_candidates",
    "max_error_px",
    "proposal_max_k",
    "proposal_graph",
    "scoring_graph",
    "fixed_iterations",
    "min_iterations",
    "max_iterations",
    "ransac_times",
}
TIMING_COLUMNS = {
    "running_time_s",
    "solve_ms",
    "score_ms",
    "score_us_per_eval",
    "refine_ms",
    "rank_score_ms",
    "ransac_total_ms",
}
GT_PROBE_COLUMNS = (
    "pair_idx",
    "status",
    "error_message",
    "pool_size",
    "pool_capacity",
    "pool_full",
    "gt_hcm_score",
    "pool_cutoff_hcm_score",
    "gt_edge_inliers",
    "would_enter_top_n",
)


class ProbeIntegrityError(RuntimeError):
    """Raised when a replay or summary is not bound to the controlled run."""


@dataclass(frozen=True)
class ProbeTask:
    dataset: str
    root_name: str
    subset_name: str
    k: int
    seed: int
    original_result_csv: str
    nested_manifest: str

    @property
    def key(self) -> str:
        return f"{self.root_name}/{self.subset_name}/k{self.k}/seed{self.seed}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1"}


def _require_protocol_row(row: pd.Series) -> None:
    expected: dict[str, Any] = {
        "protocol": "B",
        "mode": "HCM_MC",
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
        "min_iterations": 100000,
        "max_iterations": 100000,
        "ransac_times": 1,
    }
    for field, wanted in expected.items():
        value = row[field]
        if isinstance(wanted, bool):
            valid = _truth(value) == wanted
        elif isinstance(wanted, int):
            valid = int(value) == wanted
        elif isinstance(wanted, float):
            valid = math.isclose(float(value), wanted, rel_tol=0.0, abs_tol=1e-12)
        else:
            valid = str(value).strip().upper() == str(wanted).upper()
        if not valid:
            raise ProbeIntegrityError(
                f"Controlled protocol mismatch for {field}: {value!r} != {wanted!r}"
            )


def load_tasks(run_index: Path) -> list[ProbeTask]:
    frame = pd.read_csv(run_index)
    missing = REQUIRED_RUN_INDEX_COLUMNS.difference(frame.columns)
    if missing:
        raise ProbeIntegrityError(f"Run index is missing columns: {sorted(missing)}")
    tasks: list[ProbeTask] = []
    for _, row in frame.iterrows():
        _require_protocol_row(row)
        original = Path(str(row["result_csv"])).expanduser().resolve()
        manifest = Path(str(row["nested_manifest"])).expanduser().resolve()
        if not original.is_file() or not manifest.is_file():
            raise FileNotFoundError(
                f"Missing controlled artifact for {row['root_name']}/{row['subset_name']}: "
                f"{original} or {manifest}"
            )
        tasks.append(
            ProbeTask(
                dataset=str(row["dataset"]),
                root_name=str(row["root_name"]),
                subset_name=str(row["subset_name"]),
                k=int(row["k"]),
                seed=int(row["seed"]),
                original_result_csv=str(original),
                nested_manifest=str(manifest),
            )
        )
    keys = [task.key for task in tasks]
    if len(keys) != len(set(keys)):
        raise ProbeIntegrityError("Run index contains duplicate leaf/K/seed tasks")
    leaves = {(task.dataset, task.root_name, task.subset_name) for task in tasks}
    expected = {
        (dataset, root, subset, k, seed)
        for dataset, root, subset in leaves
        for k in KS
        for seed in SEEDS
    }
    observed = {
        (task.dataset, task.root_name, task.subset_name, task.k, task.seed)
        for task in tasks
    }
    if observed != expected or {task.dataset for task in tasks} != set(DATASET_ORDER):
        raise ProbeIntegrityError("Run index is not the complete six-dataset leaf x 5K x 5seed rectangle")
    return sorted(
        tasks,
        key=lambda task: (-task.k, task.dataset, task.subset_name, task.seed),
    )


def _task_dir(output_root: Path, task: ProbeTask) -> Path:
    return (
        output_root
        / "tasks"
        / task.root_name
        / task.subset_name
        / f"k{task.k}"
        / f"seed{task.seed}"
    )


def _result_paths(task_dir: Path) -> tuple[Path, Path]:
    stem = "HCM_MC_pose_replay_proposal_k_1_q_ub_0.30"
    primary = task_dir / f"{stem}.csv"
    sidecar = task_dir / f"{stem}_gt_hypothesis_probe.csv"
    return primary, sidecar


def _runner_config(task: ProbeTask, task_dir: Path) -> str:
    manifest = Path(task.nested_manifest)
    subset_dir = manifest.parent.parent
    if subset_dir.name != task.subset_name or subset_dir.parent.name != task.root_name:
        raise ProbeIntegrityError(
            f"Nested manifest path does not bind task {task.key}: {manifest}"
        )
    matching_result_root = subset_dir.parent
    output_csv = (task_dir / "pose_replay.csv").resolve()
    values = {
        "matching_result_root": matching_result_root,
        "datasets": task.subset_name,
        "method": f"protocol_b_nested/progressive_k{task.k}",
        "ransac_mode": "HCM_MC",
        "top_n_candidates": 100,
        "pool_dedup_deg": 0.0,
        "write_candidate_traces": "false",
        "write_gt_hypothesis_probe": "true",
        "q_ub": 0.3,
        "m2m_delta": 0.01,
        "max_error_px": 1.0,
        "similarity_threshold": 0.0,
        "max_matching_num": 0,
        "proposal_max_k": 1,
        "min_iterations": 100000,
        "max_iterations": 100000,
        "success_prob": 0.9999,
        "seed": task.seed,
        "tangent_sampson": "false",
        "init_with_gt": "false",
        "skip_existing_pairs": "true",
        "allow_unbound_pose": "false",
        "ransac_times": 1,
        "output_csv": output_csv,
    }
    return "".join(f"{key}={value}\n" for key, value in values.items())


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ProbeIntegrityError(f"CSV has no header: {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def verify_replay(
    task: ProbeTask, primary: Path, sidecar: Path
) -> dict[str, Any]:
    original_path = Path(task.original_result_csv)
    original_columns, original_rows = _read_rows(original_path)
    replay_columns, replay_rows = _read_rows(primary)
    probe_columns, probe_rows = _read_rows(sidecar)
    if original_columns != replay_columns:
        raise ProbeIntegrityError(f"Normal result schema changed for {task.key}")
    if probe_columns != list(GT_PROBE_COLUMNS):
        raise ProbeIntegrityError(f"Unexpected probe schema for {task.key}")
    invariant_columns = [
        column for column in original_columns if column not in TIMING_COLUMNS
    ]
    if len(original_rows) != len(replay_rows) or len(replay_rows) != len(probe_rows):
        raise ProbeIntegrityError(f"Replay row count changed for {task.key}")
    if not original_rows:
        raise ProbeIntegrityError(f"Replay is empty for {task.key}")
    for original, replay, probe in zip(original_rows, replay_rows, probe_rows):
        if original["pair_idx"] != replay["pair_idx"] or replay["pair_idx"] != probe["pair_idx"]:
            raise ProbeIntegrityError(f"Pair order changed for {task.key}")
        for column in invariant_columns:
            if original[column] != replay[column]:
                raise ProbeIntegrityError(
                    f"Probe changed {task.key}/pair {original['pair_idx']} column {column}: "
                    f"{original[column]!r} -> {replay[column]!r}"
                )
        if original["status"] != "success" or probe["status"] != "success":
            raise ProbeIntegrityError(
                f"Controlled-success rectangle changed status for {task.key}/pair {original['pair_idx']}"
            )
        if (
            int(probe["pool_size"]) != 100
            or int(probe["pool_capacity"]) != 100
            or probe["pool_full"] != "1"
            or probe["would_enter_top_n"] not in {"0", "1"}
            or not math.isfinite(float(probe["gt_hcm_score"]))
            or not math.isfinite(float(probe["pool_cutoff_hcm_score"]))
            or int(probe["gt_edge_inliers"]) < 0
        ):
            raise ProbeIntegrityError(
                f"Invalid frozen-pool probe for {task.key}/pair {original['pair_idx']}"
            )
        expected_inclusion = (
            float(probe["gt_hcm_score"])
            < float(probe["pool_cutoff_hcm_score"])
        )
        if expected_inclusion != (probe["would_enter_top_n"] == "1"):
            raise ProbeIntegrityError(
                f"Strict admission decision is inconsistent for {task.key}/pair {original['pair_idx']}"
            )
    return {
        "schema_version": 1,
        "artifact": "protocol_b_gt_hypothesis_probe_task_receipt",
        "created_at_utc": _utc_now(),
        "task": asdict(task),
        "rows": len(replay_rows),
        "normal_result_invariant_columns": invariant_columns,
        "timing_columns_excluded_from_identity_check": sorted(TIMING_COLUMNS),
        "original_result_sha256": _sha256(original_path),
        "replay_result": str(primary),
        "replay_result_sha256": _sha256(primary),
        "probe_sidecar": str(sidecar),
        "probe_sidecar_sha256": _sha256(sidecar),
        "verification": "PASS",
    }


def run_task(task: ProbeTask, runner: Path, output_root: Path) -> dict[str, Any]:
    task_dir = _task_dir(output_root, task)
    receipt_path = task_dir / "receipt.json"
    if receipt_path.is_file():
        stored = json.loads(receipt_path.read_text(encoding="utf-8"))
        if stored.get("verification") != "PASS" or stored.get("task") != asdict(task):
            raise ProbeIntegrityError(
                f"Existing receipt is not bound to the requested task: {receipt_path}"
            )
        primary, sidecar = _result_paths(task_dir)
        fresh = verify_replay(task, primary, sidecar)
        for field in (
            "original_result_sha256",
            "replay_result_sha256",
            "probe_sidecar_sha256",
        ):
            if stored.get(field) != fresh[field]:
                raise ProbeIntegrityError(
                    f"Existing receipt hash changed for {task.key}: {field}"
                )
        return {"task": task.key, "rows": fresh["rows"]}

    task_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(task_dir / "task.json", {"schema_version": 1, **asdict(task)})
    config_path = task_dir / "runner.cfg"
    _atomic_text(config_path, _runner_config(task, task_dir))
    primary, sidecar = _result_paths(task_dir)
    stdout_path = task_dir / "runner.stdout.txt"
    stderr_path = task_dir / "runner.stderr.txt"
    env = os.environ.copy()
    env.update(THREAD_ENV)
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            (str(runner), str(config_path)),
            stdout=stdout,
            stderr=stderr,
            env=env,
            check=False,
        )
    if completed.returncode != 0:
        tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"Runner failed for {task.key}:\n{tail}")
    if not primary.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"Runner did not publish both outputs for {task.key}")
    receipt = verify_replay(task, primary, sidecar)
    _atomic_json(receipt_path, receipt)
    return {"task": task.key, "rows": receipt["rows"]}


def run_campaign(
    run_index: Path,
    runner: Path,
    output_root: Path,
    workers: int,
) -> None:
    if not runner.is_file() or not os.access(runner, os.X_OK):
        raise FileNotFoundError(f"Runner is not executable: {runner}")
    if workers <= 0:
        raise ValueError("workers must be positive")
    tasks = load_tasks(run_index)
    output_root.mkdir(parents=True, exist_ok=True)
    campaign_path = output_root / "campaign.json"
    campaign_binding = {
        "schema_version": 1,
        "artifact": "protocol_b_gt_hypothesis_probe_campaign",
        "run_index": str(run_index.resolve()),
        "run_index_sha256": _sha256(run_index),
        "runner": str(runner.resolve()),
        "runner_sha256": _sha256(runner),
        "tasks": len(tasks),
        "thread_environment": THREAD_ENV,
    }
    if campaign_path.is_file():
        existing_campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        mismatches = {
            field: (existing_campaign.get(field), value)
            for field, value in campaign_binding.items()
            if existing_campaign.get(field) != value
        }
        if mismatches:
            raise ProbeIntegrityError(
                f"Output root is bound to a different probe campaign: {mismatches}"
            )
    else:
        _atomic_json(
            campaign_path,
            {
                **campaign_binding,
                "created_at_utc": _utc_now(),
                "workers": workers,
            },
        )
    completed_count = 0
    total_rows = 0
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    futures = {
        executor.submit(run_task, task, runner, output_root): task
        for task in tasks
    }
    try:
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed_count += 1
            total_rows += int(result["rows"])
            print(
                f"[{completed_count}/{len(tasks)}] {result['task']} "
                f"({result['rows']} rows)",
                flush=True,
            )
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    _atomic_json(
        output_root / "completion.json",
        {
            "schema_version": 1,
            "artifact": "protocol_b_gt_hypothesis_probe_completion",
            "created_at_utc": _utc_now(),
            "tasks": completed_count,
            "rows": total_rows,
            "status": "complete",
        },
    )


def _iter_complete_tasks(
    output_root: Path,
) -> Iterable[tuple[dict[str, Any], dict[str, Any], Path]]:
    for task_path in sorted((output_root / "tasks").glob("**/task.json")):
        task_payload = json.loads(task_path.read_text(encoding="utf-8"))
        task_dir = task_path.parent
        receipt_path = task_dir / "receipt.json"
        if not receipt_path.is_file():
            raise ProbeIntegrityError(f"Missing receipt: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("verification") != "PASS":
            raise ProbeIntegrityError(f"Failed receipt: {receipt_path}")
        receipt_task = receipt.get("task")
        task_fields = {
            field: task_payload.get(field)
            for field in ProbeTask.__dataclass_fields__
        }
        if receipt_task != task_fields:
            raise ProbeIntegrityError(
                f"Receipt task binding differs from task.json: {receipt_path}"
            )
        sidecar = Path(str(receipt["probe_sidecar"]))
        replay = Path(str(receipt["replay_result"]))
        original = Path(str(task_payload["original_result_csv"]))
        if _sha256(sidecar) != receipt["probe_sidecar_sha256"]:
            raise ProbeIntegrityError(f"Probe sidecar changed after verification: {sidecar}")
        if _sha256(replay) != receipt["replay_result_sha256"]:
            raise ProbeIntegrityError(f"Normal replay changed after verification: {replay}")
        if _sha256(original) != receipt["original_result_sha256"]:
            raise ProbeIntegrityError(
                f"Original controlled result changed after verification: {original}"
            )
        yield task_payload, receipt, sidecar


def summarize_campaign(output_root: Path, summary_dir: Path) -> dict[str, Any]:
    campaign_path = output_root / "campaign.json"
    completion_path = output_root / "completion.json"
    if not campaign_path.is_file() or not completion_path.is_file():
        raise ProbeIntegrityError("Probe campaign has no complete campaign/completion binding")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or completion.get("tasks") != campaign.get("tasks"):
        raise ProbeIntegrityError("Probe campaign completion is incomplete")

    run_index = Path(str(campaign.get("run_index", ""))).expanduser().resolve()
    if (
        not run_index.is_file()
        or campaign.get("run_index_sha256") != _sha256(run_index)
    ):
        raise ProbeIntegrityError("Probe campaign run-index binding is invalid")
    expected_tasks = load_tasks(run_index)
    expected_task_cells = {
        (task.dataset, task.root_name, task.subset_name, task.k, task.seed)
        for task in expected_tasks
    }
    if int(campaign.get("tasks", -1)) != len(expected_task_cells):
        raise ProbeIntegrityError("Probe campaign task count disagrees with its run index")

    long_frames: list[pd.DataFrame] = []
    task_cells: set[tuple[str, str, str, int, int]] = set()
    receipt_rows = 0
    for task, receipt, sidecar in _iter_complete_tasks(output_root):
        key = (
            str(task["dataset"]),
            str(task["root_name"]),
            str(task["subset_name"]),
            int(task["k"]),
            int(task["seed"]),
        )
        if key in task_cells:
            raise ProbeIntegrityError(f"Duplicate completed probe task: {key}")
        task_cells.add(key)
        frame = pd.read_csv(sidecar)
        if tuple(frame.columns) != GT_PROBE_COLUMNS or not frame["status"].eq("success").all():
            raise ProbeIntegrityError(f"Invalid completed sidecar: {sidecar}")
        if int(receipt.get("rows", -1)) != len(frame):
            raise ProbeIntegrityError(f"Receipt row count differs from sidecar: {sidecar}")
        receipt_rows += len(frame)
        frame.insert(0, "seed", int(task["seed"]))
        frame.insert(0, "k", int(task["k"]))
        frame.insert(0, "subset_name", str(task["subset_name"]))
        frame.insert(0, "root_name", str(task["root_name"]))
        frame.insert(0, "dataset", str(task["dataset"]))
        long_frames.append(frame)
    if task_cells != expected_task_cells:
        raise ProbeIntegrityError(
            "Completed task cells do not match the run-index leaf x K x seed rectangle"
        )
    long = pd.concat(long_frames, ignore_index=True)
    long["would_enter_top_n"] = pd.to_numeric(
        long["would_enter_top_n"], errors="raise"
    ).astype(int)
    if not long["would_enter_top_n"].isin({0, 1}).all():
        raise ProbeIntegrityError("Probe inclusion field is not binary")
    if long.duplicated(
        ["root_name", "subset_name", "k", "seed", "pair_idx"]
    ).any():
        raise ProbeIntegrityError("Probe campaign contains duplicate pair cells")
    if len(long) != receipt_rows or len(long) != int(completion["rows"]):
        raise ProbeIntegrityError("Probe long table row count disagrees with completion")

    dataset_summary = (
        long.groupby(["dataset", "k"], as_index=False, sort=False)
        .agg(
            included=("would_enter_top_n", "sum"),
            decisions=("would_enter_top_n", "size"),
        )
    )
    dataset_summary["inclusion_rate_percent"] = (
        100.0 * dataset_summary["included"] / dataset_summary["decisions"]
    )
    expected_cells = {(dataset, k) for dataset in DATASET_ORDER for k in KS}
    if set(zip(dataset_summary["dataset"], dataset_summary["k"])) != expected_cells:
        raise ProbeIntegrityError("Summary is not the complete six-dataset x 5K table")
    macro = (
        dataset_summary.groupby("k", as_index=False)
        .agg(inclusion_rate_percent=("inclusion_rate_percent", "mean"))
    )
    macro.insert(0, "dataset", "Macro")
    macro["included"] = pd.NA
    macro["decisions"] = pd.NA
    summary = pd.concat([dataset_summary, macro], ignore_index=True)
    dataset_rank = {dataset: rank for rank, dataset in enumerate((*DATASET_ORDER, "Macro"))}
    summary["_rank"] = summary["dataset"].map(dataset_rank)
    summary = summary.sort_values(["_rank", "k"]).drop(columns="_rank")

    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = summary_dir / "gt_hypothesis_probe_summary.csv"
    long_csv = summary_dir / "gt_hypothesis_probe_pair_results.csv.gz"
    summary.to_csv(summary_csv, index=False, float_format="%.12f")
    with gzip.open(long_csv, "wt", encoding="utf-8", newline="") as handle:
        long.to_csv(handle, index=False, float_format="%.17g")
    manifest = {
        "schema_version": 1,
        "artifact": "protocol_b_gt_hypothesis_probe_summary",
        "created_at_utc": _utc_now(),
        "definition": (
            "benchmark reference pose scored after the normal run against the "
            "frozen Stage-1 top-100 raw-seed HCM pool; strict lower-score admission"
        ),
        "normal_output_identity": (
            "all original result columns except wall-clock duration fields match exactly"
        ),
        "aggregation": (
            "pair/seed decisions pooled within each dataset; Macro is the equal arithmetic "
            "mean of the six dataset inclusion rates"
        ),
        "oracle_pose_auc_computed": False,
        "bootstrap_used": False,
        "tasks": len(task_cells),
        "pair_seed_k_rows": len(long),
        "summary_csv": str(summary_csv),
        "summary_csv_sha256": _sha256(summary_csv),
        "pair_results_csv_gz": str(long_csv),
        "pair_results_csv_gz_sha256": _sha256(long_csv),
        "campaign_manifest_sha256": _sha256(campaign_path),
        "campaign_completion_sha256": _sha256(completion_path),
    }
    manifest_path = summary_dir / "gt_hypothesis_probe_manifest.json"
    _atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="replay the complete read-only probe")
    run.add_argument("--run-index", type=Path, required=True)
    run.add_argument("--runner", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--workers", type=int, default=20)
    summarize = commands.add_parser("summarize", help="validate and aggregate a complete replay")
    summarize.add_argument("--output-root", type=Path, required=True)
    summarize.add_argument("--summary-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        run_campaign(
            args.run_index.expanduser().resolve(),
            args.runner.expanduser().resolve(),
            args.output_root.expanduser().resolve(),
            args.workers,
        )
    elif args.command == "summarize":
        summarize_campaign(
            args.output_root.expanduser().resolve(),
            args.summary_dir.expanduser().resolve(),
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
