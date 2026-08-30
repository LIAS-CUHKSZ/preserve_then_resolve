#!/usr/bin/env python3
"""Render the six-panel main-paper Q1 fixed-proposal K sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ("ScanNet", "MegaDepth", "NAVI-Multi", "NAVI-Wild", "METU-CC", "METU-CS")
KS = (1, 2, 3, 4, 5)
THRESHOLDS = (5.0, 10.0, 20.0)
STYLES = {
    5.0: ("#174a7e", "o", "-"),
    10.0: ("#2a78d6", "s", "--"),
    20.0: ("#8bbce6", "^", ":"),
}


def load_inputs(summary_path: Path, edges_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(summary_path)
    edges = pd.read_csv(edges_path)
    summary_required = {
        "dataset",
        "k",
        "threshold_deg",
        "delta_auc",
        "delta_seed_min",
        "delta_seed_max",
        "seeds",
    }
    edge_required = {"dataset", "k", "edge_median"}
    if missing := summary_required - set(summary.columns):
        raise ValueError(f"{summary_path}: missing columns {sorted(missing)}")
    if missing := edge_required - set(edges.columns):
        raise ValueError(f"{edges_path}: missing columns {sorted(missing)}")
    if summary.duplicated(["dataset", "k", "threshold_deg"]).any():
        raise ValueError("Q1 summary contains duplicate dataset/K/threshold rows")
    if edges.duplicated(["dataset", "k"]).any():
        raise ValueError("Q1 edge table contains duplicate dataset/K rows")
    expected_summary = {
        (dataset, k, threshold)
        for dataset in DATASETS
        for k in KS
        for threshold in THRESHOLDS
    }
    observed_summary = set(
        summary[["dataset", "k", "threshold_deg"]].itertuples(index=False, name=None)
    )
    expected_edges = {(dataset, k) for dataset in DATASETS for k in KS}
    observed_edges = set(edges[["dataset", "k"]].itertuples(index=False, name=None))
    if observed_summary != expected_summary:
        raise ValueError("Q1 summary is not the complete 6 x 5 x 3 rectangle")
    if observed_edges != expected_edges:
        raise ValueError("Q1 edge table is not the complete 6 x 5 rectangle")
    if not (pd.to_numeric(summary["seeds"], errors="raise") == 5).all():
        raise ValueError("Every Q1 curve point must contain five seeds")
    numeric = summary[["delta_auc", "delta_seed_min", "delta_seed_max"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Q1 summary contains non-finite deltas")
    return summary, edges


def render(
    summary: pd.DataFrame,
    edges: pd.DataFrame,
    output_stem: Path,
    *,
    use_tex: bool = False,
    dpi: int = 260,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 6.8,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
            "text.usetex": use_tex,
        }
    )
    fig, axes = plt.subplots(1, 6, figsize=(7.05, 1.78), sharey=True)
    fig.subplots_adjust(left=0.068, right=0.997, bottom=0.28, top=0.77, wspace=0.18)
    x = np.asarray(KS, dtype=float)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        dataset_rows = summary[summary["dataset"] == dataset]
        for threshold in THRESHOLDS:
            curve = dataset_rows[dataset_rows["threshold_deg"] == threshold].sort_values("k")
            color, marker, linestyle = STYLES[threshold]
            y = curve["delta_auc"].to_numpy(float)
            low = curve["delta_seed_min"].to_numpy(float)
            high = curve["delta_seed_max"].to_numpy(float)
            axis.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)
            axis.plot(
                x,
                y,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.0,
                markersize=2.6,
                markeredgecolor="white",
                markeredgewidth=0.3,
                label=rf"AUC@${threshold:g}^\circ$",
            )
        axis.axhline(0.0, color="#111111", linestyle=(0, (3, 2)), linewidth=0.7)
        axis.set_title(dataset, fontsize=7.5, pad=4.0)
        axis.set_xticks(KS)
        axis.set_xlim(0.7, 5.3)
        axis.set_ylim(-6.0, 10.5)
        axis.grid(axis="y", color="#e0e0dc", linewidth=0.55)
        axis.tick_params(axis="x", length=2.0, labelsize=6.2)
        axis.tick_params(axis="y", length=2.0, labelsize=6.2)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#777570")

        counts = (
            edges[edges["dataset"] == dataset]
            .set_index("k")
            .reindex(KS)["edge_median"]
            .to_numpy(float)
        )
        label_y = -4.15 if dataset != "MegaDepth" else 7.1
        value_y = -5.15 if dataset != "MegaDepth" else 5.9
        for k, count in zip(KS, counts, strict=True):
            axis.text(k, label_y, rf"$|E_{k}|$", ha="center", va="center", fontsize=5.3, color="#55534f")
            axis.text(k, value_y, f"{count:.0f}", ha="center", va="center", fontsize=4.9, color="#777570")
    axes[0].set_ylabel(r"$\Delta$ Pose AUC (pp)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.99),
        ncol=3,
        frameon=False,
        fontsize=7.0,
        handlelength=1.8,
        columnspacing=1.2,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(
        output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.01
    )
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--edge-counts", type=Path, required=True)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("outputs/paper_figures/estimation/q1_nested_k"),
    )
    parser.add_argument("--tex", action="store_true", help="Use LaTeX text rendering")
    args = parser.parse_args()
    summary, edges = load_inputs(args.summary.resolve(), args.edge_counts.resolve())
    render(summary, edges, args.output_stem.resolve(), use_tex=args.tex)
    print(f"wrote {args.output_stem}.pdf and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
