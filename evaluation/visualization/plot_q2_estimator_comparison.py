#!/usr/bin/env python3
"""Render the five-seed Q2 estimator comparison from the main paper.

Inputs
------
``auc_five_seed_summary.csv`` from ``run_pose_gms_selection.py summarize``.

Outputs
-------
``q2_estimator_triplets.pdf``/``.png`` in the paper-figure output directory.
The figure reports dataset-level Pose AUC as compact vertical triplets at 5,
10, and 20 degrees.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


DATASETS = ("ScanNet", "MegaDepth", "NAVI-Multi", "NAVI-Wild", "METU-CC", "METU-CS")
DATASET_TICK_LABELS = (
    "ScanNet",
    "Mega-\nDepth",
    "NAVI-\nMulti",
    "NAVI-\nWild",
    "METU-\nCC",
    "METU-\nCS",
)
THRESHOLDS = (5.0, 10.0, 20.0)
VARIANTS = ("m2m_poselib_cm", "hcm", "mcm", "full_hcm_mc")
LABELS = {
    "m2m_poselib_cm": "m-to-m + CM",
    "hcm": "HCM",
    "mcm": "MCM",
    "full_hcm_mc": "HCM$\\rightarrow$MCM",
}
COLORS = {
    "m2m_poselib_cm": "#77736d",
    "hcm": "#1baf7a",
    "mcm": "#008300",
    "full_hcm_mc": "#2a78d6",
}
MARKERS = {
    "m2m_poselib_cm": "o",
    "hcm": "s",
    "mcm": "^",
    "full_hcm_mc": "D",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e0"


def load_summary(path: Path) -> dict[tuple[str, str, float], dict[str, float]]:
    """Load and strictly validate the frozen 4 x 7 x 3 summary rectangle."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_datasets = set(DATASETS) | {"Macro"}
    expected_keys = {
        (variant, dataset, threshold)
        for variant in VARIANTS
        for dataset in expected_datasets
        for threshold in THRESHOLDS
    }
    values: dict[tuple[str, str, float], dict[str, float]] = {}
    for row in rows:
        key = (row["variant"], row["dataset"], float(row["threshold_deg"]))
        if key in values:
            raise ValueError(f"Duplicate Q2 summary cell: {key}")
        if row["seeds"] != "0,1,2,3,4":
            raise ValueError(f"Unexpected seed set for {key}: {row['seeds']}")
        values[key] = {
            field: float(row[field]) for field in ("mean", "seed_std", "min", "max")
        }
    if set(values) != expected_keys:
        missing = sorted(expected_keys - set(values))
        extra = sorted(set(values) - expected_keys)
        raise ValueError(f"Unexpected Q2 summary rectangle; missing={missing}, extra={extra}")
    for key, cell in values.items():
        if not all(np.isfinite(list(cell.values()))):
            raise ValueError(f"Non-finite Q2 summary value in {key}")
        if not cell["min"] <= cell["mean"] <= cell["max"]:
            raise ValueError(f"Mean lies outside the seed range in {key}")
    return values


def render_triplets(
    values: dict[tuple[str, str, float], dict[str, float]], output_dir: Path
) -> None:
    """Render one single-column plot of dataset-level threshold triplets."""
    fig, axis = plt.subplots(figsize=(3.45, 2.05))
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.30, top=0.97)
    dataset_x = np.arange(len(DATASETS), dtype=float)
    method_offsets = np.linspace(-0.27, 0.27, len(VARIANTS))
    threshold_alpha = (0.55, 0.78, 1.0)
    threshold_sizes = (2.35, 2.65, 2.95)

    for dataset_index, dataset in enumerate(DATASETS):
        for variant_index, variant in enumerate(VARIANTS):
            means = np.asarray(
                [values[(variant, dataset, threshold)]["mean"] for threshold in THRESHOLDS]
            )
            is_full = variant == "full_hcm_mc"
            glyph_x = dataset_x[dataset_index] + method_offsets[variant_index]
            axis.plot(
                (glyph_x, glyph_x),
                (means[0], means[-1]),
                color=COLORS[variant],
                linewidth=0.75,
                alpha=0.55,
                solid_capstyle="round",
                zorder=2,
            )
            for threshold_index, mean in enumerate(means):
                axis.plot(
                    glyph_x,
                    mean,
                    color=COLORS[variant],
                    marker=MARKERS[variant],
                    linestyle="none",
                    markersize=(
                        threshold_sizes[threshold_index] + (0.15 if is_full else 0.0)
                    ),
                    markeredgecolor="white" if is_full else COLORS[variant],
                    markeredgewidth=0.35,
                    alpha=threshold_alpha[threshold_index],
                    zorder=5 if is_full else 3,
                )

    axis.set_xlim(-0.55, len(DATASETS) - 0.45)
    axis.set_ylim(0.0, 65.0)
    axis.set_yticks((0, 20, 40, 60))
    axis.set_xticks(dataset_x, DATASET_TICK_LABELS)
    axis.set_ylabel(r"Pose AUC (\%)")
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color=GRID, linewidth=0.55)
    axis.tick_params(axis="y", colors=INK2, labelcolor=INK, labelsize=6.8)
    axis.tick_params(
        axis="x", colors=INK2, labelcolor=INK, labelsize=6.3, length=0, pad=2.0
    )
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK2)

    legend_handles = []
    for variant in VARIANTS:
        is_full = variant == "full_hcm_mc"
        legend_handles.append(
            axis.plot(
                [],
                [],
                color=COLORS[variant],
                marker=MARKERS[variant],
                linewidth=0.75,
                markersize=3.0 + (0.15 if is_full else 0.0),
                markeredgecolor="white" if is_full else COLORS[variant],
                markeredgewidth=0.35,
                label=LABELS[variant],
            )[0]
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.085),
        frameon=False,
        ncol=4,
        fontsize=6.8,
        handlelength=0.9,
        handletextpad=0.3,
        columnspacing=0.7,
        borderaxespad=0.0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "q2_estimator_triplets"
    save_options = {"bbox_inches": "tight", "pad_inches": 0.01}
    fig.savefig(stem.with_suffix(".pdf"), **save_options)
    fig.savefig(stem.with_suffix(".png"), dpi=260, **save_options)
    plt.close(fig)
    print(f"wrote {stem}.pdf and .png")


def render(
    values: dict[tuple[str, str, float], dict[str, float]],
    output_dir: Path,
) -> None:
    """Render the main-paper threshold triplets."""
    render_triplets(values, output_dir)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=repository_root / "outputs" / "paper_figures" / "estimation",
    )
    parser.add_argument("--tex", action="store_true", help="Use LaTeX text rendering")
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "legend.fontsize": 6.9,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
            "text.usetex": args.tex,
            **(
                {"text.latex.preamble": r"\usepackage{newtxtext,newtxmath}"}
                if args.tex
                else {}
            ),
        }
    )
    values = load_summary(args.input.resolve())
    render(values, args.figure_output.resolve())


if __name__ == "__main__":
    main()
