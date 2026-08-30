from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg", force=True)

from evaluation.visualization.plot_q2_estimator_comparison import (
    DATASETS,
    THRESHOLDS,
    VARIANTS,
    load_summary,
    render,
)


def _write_summary(path: Path, *, omit_last: bool = False) -> None:
    rows = []
    for variant_index, variant in enumerate(VARIANTS):
        for dataset_index, dataset in enumerate((*DATASETS, "Macro")):
            for threshold in THRESHOLDS:
                mean = 10.0 + variant_index + dataset_index + threshold / 10.0
                rows.append(
                    {
                        "pipeline": "fixture",
                        "variant": variant,
                        "association": "gms_t4_g20",
                        "mode": "fixture",
                        "dataset": dataset,
                        "threshold_deg": threshold,
                        "seeds": "0,1,2,3,4",
                        "mean": mean,
                        "seed_std": 0.2,
                        "min": mean - 0.3,
                        "max": mean + 0.3,
                    }
                )
    if omit_last:
        rows.pop()
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_q2_plot_uses_exact_body_rectangle(tmp_path: Path) -> None:
    source = tmp_path / "auc_five_seed_summary.csv"
    _write_summary(source)
    values = load_summary(source)
    output = tmp_path / "figures"
    render(values, output)
    assert (output / "q2_estimator_triplets.pdf").stat().st_size > 0
    assert (output / "q2_estimator_triplets.png").stat().st_size > 0


def test_q2_plot_rejects_incomplete_rectangle(tmp_path: Path) -> None:
    source = tmp_path / "incomplete.csv"
    _write_summary(source, omit_last=True)
    with pytest.raises(ValueError, match="Unexpected Q2 summary rectangle"):
        load_summary(source)
