from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

from evaluation.estimation import run_q1_nested_k as q1
from evaluation.visualization.plot_q1_nested_k import load_inputs, render


def _write_source(path: Path, pair_count: int = 2) -> None:
    path.mkdir(parents=True)
    (path / "association_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pair_count": pair_count,
                "progressive_max_k": 5,
                "association_upperbound": 0,
                "model_name": "dinov3_vitl16",
                "layer": 19,
                "debias_rank": 200,
                "weights_id": q1.PAPER_DINOV3_WEIGHTS_ID,
            }
        ),
        encoding="utf-8",
    )
    rows = pd.DataFrame(
        {
            "left_idx": [0, 0, 1, 1, 2],
            "right_idx": [0, 1, 1, 2, 2],
            "x1": [0, 0, 1, 1, 2],
            "y1": [0, 0, 1, 1, 2],
            "x2": [0, 1, 1, 2, 2],
            "y2": [0, 1, 1, 2, 2],
            "similarity": [0.9, 0.8, 0.7, 0.6, 0.5],
            "k": [1, 2, 3, 4, 5],
        }
    )
    for pair_idx in range(1, pair_count + 1):
        rows.to_csv(path / f"matching_{pair_idx:03d}.csv", index=False)


def test_q1_derivation_is_nested_resumable_and_source_read_only(tmp_path: Path) -> None:
    results = tmp_path / "results"
    subset = q1.Subset("ScanNet", "scan", "pairs", 2)
    source = (
        results
        / subset.root_name
        / subset.subset_name
        / "combined_interpolate_v3/layer19/debias_svd200/progressive_k5"
    )
    _write_source(source)
    before = hashlib.sha256((source / "matching_001.csv").read_bytes()).hexdigest()
    args = argparse.Namespace(
        results_root=results,
        source_method="combined_interpolate_v3",
        layer=19,
        debias_rank=200,
        existing="validate",
    )
    payload = q1._derive_subset(args, subset)
    q1._derive_subset(args, subset)
    after = hashlib.sha256((source / "matching_001.csv").read_bytes()).hexdigest()
    assert before == after
    assert payload["proposal_graph"] == "E1"
    assert payload["scoring_graph"] == "EK"
    nested = results / "scan/pairs/protocol_b_nested"
    assert [len(pd.read_csv(nested / f"progressive_k{k}/matching_001.csv")) for k in q1.KS] == list(q1.KS)

    task = q1.Task(subset, k=5, seed=3)
    config = q1._config_text(results, task)
    assert "proposal_max_k=1" in config
    assert "min_iterations=100000\nmax_iterations=100000" in config
    assert "max_matching_num=0" in config
    assert q1._result_filename(3) == (
        "HCM_MC_q1_pose_seed3_iter100000_proposal_k_1_q_ub_0.30.csv"
    )


def test_q1_plot_accepts_body_rectangle(tmp_path: Path) -> None:
    summary_rows = []
    edge_rows = []
    for dataset_index, dataset in enumerate(q1.DATASETS):
        for k in q1.KS:
            edge_rows.append(
                {"dataset": dataset, "k": k, "pairs": 10, "edge_median": 100 + 200 * k}
            )
            for threshold in q1.THRESHOLDS:
                delta = (k - 1) * (dataset_index + 1) * threshold / 100.0
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "k": k,
                        "threshold_deg": threshold,
                        "auc_mean": 20 + delta,
                        "auc_seed_std": 0.1,
                        "delta_auc": delta,
                        "delta_seed_min": delta - (0.1 if k > 1 else 0.0),
                        "delta_seed_max": delta + (0.1 if k > 1 else 0.0),
                        "seeds": 5,
                        "pairs": 10,
                    }
                )
    summary_path = tmp_path / "q1_auc_summary.csv"
    edge_path = tmp_path / "q1_edge_counts.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(edge_rows).to_csv(edge_path, index=False)
    summary, edges = load_inputs(summary_path, edge_path)
    stem = tmp_path / "q1_nested_k"
    render(summary, edges, stem, use_tex=False, dpi=72)
    assert stem.with_suffix(".pdf").stat().st_size > 0
    assert stem.with_suffix(".png").stat().st_size > 0


def test_q1_plot_rejects_duplicate_cells(tmp_path: Path) -> None:
    summary_rows = []
    edge_rows = []
    for dataset in q1.DATASETS:
        for k in q1.KS:
            edge_rows.append({"dataset": dataset, "k": k, "edge_median": 10})
            for threshold in q1.THRESHOLDS:
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "k": k,
                        "threshold_deg": threshold,
                        "delta_auc": 0.0,
                        "delta_seed_min": 0.0,
                        "delta_seed_max": 0.0,
                        "seeds": 5,
                    }
                )
    summary_rows.append(dict(summary_rows[0]))
    summary_path = tmp_path / "q1_auc_summary.csv"
    edge_path = tmp_path / "q1_edge_counts.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(edge_rows).to_csv(edge_path, index=False)
    with pytest.raises(ValueError, match="duplicate dataset/K/threshold"):
        load_inputs(summary_path, edge_path)


def test_q1_summary_rejects_duplicate_run_index_rows(tmp_path: Path) -> None:
    rows = []
    for dataset_index, dataset in enumerate(q1.DATASETS):
        for k in q1.KS:
            for seed in q1.SEEDS:
                rows.append(
                    {
                        "dataset": dataset,
                        "root_name": f"root_{dataset_index}",
                        "subset_name": "subset",
                        "k": k,
                        "seed": seed,
                        "result_csv": "unused.csv",
                        "nested_manifest": "unused.json",
                    }
                )
    rows.append(dict(rows[0]))
    run_index = tmp_path / "run_index.csv"
    pd.DataFrame(rows).to_csv(run_index, index=False)
    args = argparse.Namespace(
        run_index=run_index,
        results_root=tmp_path / "results",
        output_dir=tmp_path / "summary",
    )
    with pytest.raises(ValueError, match="duplicate subset/K/seed"):
        q1.run_summarize(args)


def test_q1_resume_rejects_pair_id_only_rows(tmp_path: Path) -> None:
    result = tmp_path / "truncated.csv"
    pd.DataFrame({"pair_idx": [1, 2]}).to_csv(result, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        q1._result_pair_ids(result, 2)


def test_q1_summary_rejects_partial_result_csv(tmp_path: Path) -> None:
    result = tmp_path / "partial.csv"
    pd.DataFrame(
        {
            "pair_idx": [1],
            "status": ["success"],
            "rotation_error_deg": [1.0],
            "translation_error_deg": [1.0],
            "running_time_s": [0.1],
            "score_us_per_eval": [2.0],
        }
    ).to_csv(result, index=False)

    rows = []
    results_root = tmp_path / "results"
    for dataset_index, dataset in enumerate(q1.DATASETS):
        root_name = f"root_{dataset_index}"
        subset_name = "subset"
        pose = results_root / root_name / subset_name / "pose_intrinsics.csv"
        pose.parent.mkdir(parents=True)
        pd.DataFrame({"pair_idx": [1, 2]}).to_csv(pose, index=False)
        for k in q1.KS:
            for seed in q1.SEEDS:
                rows.append(
                    {
                        "dataset": dataset,
                        "root_name": root_name,
                        "subset_name": subset_name,
                        "k": k,
                        "seed": seed,
                        "result_csv": result,
                        "nested_manifest": tmp_path / f"nested_{dataset_index}.json",
                    }
                )
    run_index = tmp_path / "run_index.csv"
    pd.DataFrame(rows).to_csv(run_index, index=False)
    args = argparse.Namespace(
        run_index=run_index,
        results_root=results_root,
        output_dir=tmp_path / "summary",
    )
    with pytest.raises(ValueError, match="is incomplete"):
        q1.run_summarize(args)
