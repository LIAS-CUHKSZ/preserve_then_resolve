from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

from evaluation.visualization.plot_q3_dino import (
    BASELINE,
    DATASETS,
    PROPOSED,
    SEEDS,
    THRESHOLDS,
    load_paired_deltas,
    render,
    summarize_deltas,
)


def _write_fixture(path: Path, *, missing_last: bool = False) -> None:
    rows = []
    for pipeline in (BASELINE, PROPOSED):
        for dataset_index, dataset in enumerate(DATASETS):
            for threshold in THRESHOLDS:
                for seed in SEEDS:
                    rows.append(
                        {
                            "pipeline": pipeline,
                            "variant": "fixture",
                            "association": "fixture",
                            "mode": "fixture",
                            "dataset": dataset,
                            "threshold_deg": threshold,
                            "seed": seed,
                            "auc": (
                                10.0
                                + dataset_index
                                + seed * 0.1
                                + (2.0 + seed * 0.05 if pipeline == PROPOSED else 0.0)
                            ),
                            "pairs": 10,
                            "result_files": "",
                        }
                    )
    if missing_last:
        rows.pop()
    pd.DataFrame(rows).to_csv(path, index=False)


def test_q3_deltas_are_paired_by_seed_and_render(tmp_path: Path) -> None:
    source = tmp_path / "auc_per_seed.csv"
    _write_fixture(source)
    delta = load_paired_deltas(source)
    summary = summarize_deltas(delta)
    assert len(delta) == 6 * 3 * 5
    assert len(summary) == 18
    assert (summary["mean"] > 2.0).all()
    assert (summary["sample_std"] > 0.0).all()

    stem = tmp_path / "q3_dino"
    counts = render(delta, delta, stem, use_tex=False, dpi=72)
    assert counts == {"corrected_dinov3": 18, "raw_dinov2": 18}
    assert stem.with_suffix(".pdf").stat().st_size > 0
    assert stem.with_suffix(".png").stat().st_size > 0


def test_q3_loader_rejects_incomplete_seed_rectangle(tmp_path: Path) -> None:
    source = tmp_path / "incomplete.csv"
    _write_fixture(source, missing_last=True)
    with pytest.raises(ValueError, match="incomplete Q3 rectangle"):
        load_paired_deltas(source)
