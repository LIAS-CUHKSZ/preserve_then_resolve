#!/usr/bin/env python3
"""Render the two DINO panels from the main-paper Q3 comparison.

Each input is ``auc_per_seed.csv`` produced by
``run_pose_gms_selection.py summarize``.  Deltas are paired by dataset,
threshold, and random seed before the mean and sample standard deviation are
computed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ("ScanNet", "MegaDepth", "NAVI-Multi", "NAVI-Wild", "METU-CC", "METU-CS")
DATASET_LABELS = ("Scan-\nNet", "Mega-\nDepth", "NAVI-\nM", "NAVI-\nW", "METU-\nCC", "METU-\nCS")
THRESHOLDS = (5.0, 10.0, 20.0)
SEEDS = (0, 1, 2, 3, 4)
BASELINE = "mnn/CM"
PROPOSED = "gms_t4_g20/HCM_MC"
COLORS = {5.0: "#194d68", 10.0: "#2473c5", 20.0: "#72b9e6"}
MARKERS = {5.0: "o", 10.0: "s", 20.0: "D"}


def load_paired_deltas(path: Path) -> pd.DataFrame:
    """Load and validate one complete 2 x 6 x 3 x 5 Q3 rectangle."""
    frame = pd.read_csv(path)
    required = {"pipeline", "dataset", "threshold_deg", "seed", "auc", "pairs"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame = frame[
        frame["pipeline"].isin((BASELINE, PROPOSED))
        & frame["dataset"].isin(DATASETS)
    ].copy()
    frame["threshold_deg"] = pd.to_numeric(frame["threshold_deg"], errors="raise")
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["auc"] = pd.to_numeric(frame["auc"], errors="raise")
    frame["pairs"] = pd.to_numeric(frame["pairs"], errors="raise").astype(int)
    expected = {
        (pipeline, dataset, threshold, seed)
        for pipeline in (BASELINE, PROPOSED)
        for dataset in DATASETS
        for threshold in THRESHOLDS
        for seed in SEEDS
    }
    keys = list(
        frame[["pipeline", "dataset", "threshold_deg", "seed"]]
        .itertuples(index=False, name=None)
    )
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate Q3 pipeline/dataset/threshold/seed rows")
    if set(keys) != expected:
        missing_keys = sorted(expected - set(keys))
        extra_keys = sorted(set(keys) - expected)
        raise ValueError(
            f"{path}: incomplete Q3 rectangle; missing={missing_keys[:5]}, "
            f"extra={extra_keys[:5]}"
        )
    if not np.isfinite(frame["auc"].to_numpy(float)).all():
        raise ValueError(f"{path}: non-finite AUC values")

    index = ["dataset", "threshold_deg", "seed"]
    auc = frame.pivot(index=index, columns="pipeline", values="auc")
    pairs = frame.pivot(index=index, columns="pipeline", values="pairs")
    if not (pairs[BASELINE] == pairs[PROPOSED]).all():
        raise ValueError(f"{path}: paired pipelines use different pair counts")
    delta = (auc[PROPOSED] - auc[BASELINE]).rename("delta_auc_pp").reset_index()
    return delta


def summarize_deltas(delta: pd.DataFrame) -> pd.DataFrame:
    summary = (
        delta.groupby(["dataset", "threshold_deg"], sort=False)["delta_auc_pp"]
        .agg(mean="mean", sample_std=lambda values: values.std(ddof=1), seeds="size")
        .reset_index()
    )
    if not (summary["seeds"] == len(SEEDS)).all():
        raise ValueError("Every Q3 error bar must contain five paired seeds")
    return summary


def render(
    dinov3: pd.DataFrame,
    dinov2: pd.DataFrame,
    output_stem: Path,
    *,
    use_tex: bool = False,
    dpi: int = 260,
) -> dict[str, int]:
    """Render corrected-DINOv3 and raw-DINOv2 panels."""
    plt.rcParams.update(
        {
            "font.size": 7.2,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
            "text.usetex": use_tex,
        }
    )
    summaries = (summarize_deltas(dinov3), summarize_deltas(dinov2))
    titles = (r"Corrected DINOv3", r"Raw DINOv2")
    fig, axes = plt.subplots(1, 2, figsize=(4.25, 1.92), sharey=True)
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.27, top=0.79, wspace=0.07)
    x = np.arange(len(DATASETS), dtype=float)
    offsets = {5.0: -0.17, 10.0: 0.0, 20.0: 0.17}
    positive_counts: dict[str, int] = {}

    for axis, summary, title in zip(axes, summaries, titles, strict=True):
        positive = int((summary["mean"] > 0.0).sum())
        positive_counts[title.replace(" ", "_").lower()] = positive
        for threshold in THRESHOLDS:
            cells = summary[summary["threshold_deg"] == threshold].set_index("dataset")
            cells = cells.reindex(DATASETS)
            axis.errorbar(
                x + offsets[threshold],
                cells["mean"].to_numpy(float),
                yerr=cells["sample_std"].to_numpy(float),
                color=COLORS[threshold],
                marker=MARKERS[threshold],
                markersize=3.0,
                markeredgecolor="white",
                markeredgewidth=0.35,
                linestyle="none",
                elinewidth=0.7,
                capsize=1.6,
                label=rf"AUC@${threshold:g}^\circ$",
                zorder=3,
            )
        axis.axhline(0.0, color="#aaa9a3", linestyle=(0, (3, 2)), linewidth=0.7)
        axis.set_title(title, pad=6.0, fontsize=8.0)
        axis.text(
            0.02,
            0.96,
            f"{positive}/18 positive",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="#55534f",
            fontsize=6.7,
        )
        axis.set_xticks(x, DATASET_LABELS)
        axis.set_xlim(-0.55, len(DATASETS) - 0.45)
        axis.set_ylim(-1.0, 13.0)
        axis.set_yticks((0.0, 2.5, 5.0, 7.5, 10.0, 12.5))
        axis.grid(axis="both", color="#dfdfdb", linewidth=0.55)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", length=0, labelsize=6.2, pad=2.0)
        axis.tick_params(axis="y", length=2.5, labelsize=6.5)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#777570")
    axes[0].set_ylabel(r"Paired $\Delta$Pose AUC (pp)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.99),
        ncol=3,
        frameon=False,
        fontsize=7.0,
        handletextpad=0.25,
        columnspacing=0.9,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close(fig)
    return positive_counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov3", type=Path, required=True)
    parser.add_argument("--dinov2", type=Path, required=True)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("outputs/paper_figures/estimation/q3_dino_transfer"),
    )
    parser.add_argument("--tex", action="store_true", help="Use LaTeX text rendering")
    args = parser.parse_args()
    counts = render(
        load_paired_deltas(args.dinov3.resolve()),
        load_paired_deltas(args.dinov2.resolve()),
        args.output_stem.resolve(),
        use_tex=args.tex,
    )
    print(f"wrote {args.output_stem}.pdf and .png; positive cells={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
