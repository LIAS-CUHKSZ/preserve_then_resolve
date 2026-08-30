"""Plot the isolated DINOv3 rank sweep and optional DINOv2 raw layer sweep.

The horizontal coordinate is the uncapped number of mutual associations.  The
vertical coordinate is the equal mean of the object-macro geometric recoveries
at strict 1, 2, and 5 cm thresholds, following the DINOv3 NAVI metric.  Every
trajectory contains the directly observed K=1..8 points; no association budget
is used to construct or interpolate a point.

Compact CSV/JSON reports and publication-ready PDF/PNG figures are written to
separate directories so numeric artifacts never become mixed with copy-ready
paper figures.

When ``--dinov2-evaluation-root`` is supplied, the established rank-200
filename is retained but its figure becomes a two-panel comparison: the
existing DINOv3 rank-200 curves on the left and all raw DINOv2 ViT-L/14 layers
on the right.  The panels share recall limits but keep independent association
count axes because patch-16 and patch-14 graph sizes are not directly aligned.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .protocol import DINOV2_MODEL_NAME, DINOV3_MODEL_NAME


BINS = ("0-40", "40-80", "80-120")
LAYERS = tuple(range(16, 25))
LAYER_SWEEP_LAYERS = LAYERS
DEBIAS_RANKS = (0, 100, 200, 300, 400, 500, 600)
DINOV2_RAW_RANKS = (0,)
MAX_K = 8
THRESHOLDS_M = (0.01, 0.02, 0.05)
RANK_COLORS = {
    0: "#000000",
    100: "#0072B2",
    200: "#009E73",
    300: "#E69F00",
    400: "#D55E00",
    500: "#CC79A7",
    600: "#7A5195",
}
# A colorblind-friendly categorical palette.  The default layer cross-section
# samples every other layer, so L17/L19/L21/L23/L24 remain easy to distinguish.
LAYER_COLORS = {
    16: "#332288",
    17: "#6A3D9A",
    18: "#0077BB",
    19: "#0072B2",
    20: "#44AA99",
    21: "#009E73",
    22: "#CCBB44",
    23: "#E69F00",
    24: "#D55E00",
}
BIN_MARKERS = {
    "0-40": "o",
    "40-80": "s",
    "80-120": "^",
}
BIN_LINESTYLES = {
    "0-40": "-",
    "40-80": (0, (2.4, 1.6)),
    "80-120": (0, (1.0, 1.5)),
}
HIGHLIGHT_LAYERS = frozenset((19, 24))
OTHER_LAYER_STYLE = (0, (4.0, 2.2))
TYPE1_FONT_SETTINGS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
    "text.usetex": False,
}


def _angle_label(angular_bin: str) -> str:
    lower, upper = angular_bin.split("-")
    return rf"{lower}--{upper}$^\circ$"


def _configure_type1_fonts() -> None:
    plt.rcParams.update(TYPE1_FONT_SETTINGS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _write_selection_report(path: Path, payload: dict[str, Any]) -> bool:
    """Write a changed selection report without invalidating equivalent consumers.

    Pose-estimation manifests bind ``best_layer.json`` by SHA256.  A purely
    visual edit to this plotting module changes ``selection_script_sha256`` but
    must not invalidate an otherwise identical, already-consumed selection.
    Preserve the existing bytes when that script hash is the only difference;
    any selection, input, or descriptor-provenance change is still written.

    Returns ``True`` when the file was created or replaced.
    """

    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = None
        if isinstance(existing, dict):
            existing_semantics = dict(existing)
            candidate_semantics = dict(payload)
            existing_semantics.pop("selection_script_sha256", None)
            candidate_semantics.pop("selection_script_sha256", None)
            if existing_semantics == candidate_semantics:
                return False

    _atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return True


def _csv_text(frame: pd.DataFrame) -> str:
    stream = io.StringIO(newline="")
    frame.to_csv(stream, index=False, lineterminator="\n")
    return stream.getvalue()


def _load_points(
    evaluation_root: Path,
    *,
    expected_model_name: str = DINOV3_MODEL_NAME,
    expected_debias_ranks: tuple[int, ...] = DEBIAS_RANKS,
) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, Any]] = []
    manifest_hashes: dict[str, str] = {}
    for angular_bin in BINS:
        bin_root = evaluation_root / f"bin_{angular_bin}"
        manifest_path = bin_root / "evaluation_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("layers") != list(LAYERS):
            raise ValueError(f"Unexpected layers in {manifest_path}")
        if manifest.get("debias_ranks") != list(expected_debias_ranks):
            raise ValueError(f"Unexpected ranks in {manifest_path}")
        # Retained DINOv3 manifests predate the explicit model field.  Continue
        # accepting those exact legacy rectangles, while requiring DINOv2 to
        # identify itself and its raw/no-correction protocol unambiguously.
        observed_model = manifest.get("model_name", DINOV3_MODEL_NAME)
        if observed_model != expected_model_name:
            raise ValueError(
                f"Expected {expected_model_name} in {manifest_path}, got {observed_model}"
            )
        if expected_model_name == DINOV2_MODEL_NAME:
            if manifest.get("patch_size") != 14:
                raise ValueError(f"DINOv2 manifest must record patch_size=14: {manifest_path}")
            if manifest.get("correction_mode") != "none":
                raise ValueError(
                    f"DINOv2 manifest must record correction_mode=none: {manifest_path}"
                )
            if manifest.get("long_edge") != 1024 or manifest.get("max_k") != MAX_K:
                raise ValueError(
                    f"DINOv2 manifest must record long_edge=1024 and max_k={MAX_K}: "
                    f"{manifest_path}"
                )
        expected_shards = 500 * len(LAYERS)
        if (
            manifest.get("pair_count") != 500
            or manifest.get("expected_shards") != expected_shards
        ):
            raise ValueError(f"Incomplete evaluation rectangle in {manifest_path}")
        if expected_model_name == DINOV2_MODEL_NAME:
            if (
                int(manifest.get("written_shards", -1))
                + int(manifest.get("resumed_shards", -1))
                != expected_shards
            ):
                raise ValueError(f"Incomplete DINOv2 shard execution in {manifest_path}")
            pair_indices = manifest.get("pair_indices")
            if (
                not isinstance(pair_indices, list)
                or len(pair_indices) != 500
                or len(set(pair_indices)) != 500
            ):
                raise ValueError(f"Invalid DINOv2 pair rectangle in {manifest_path}")
            split_audit = manifest.get("split_audit")
            if not isinstance(split_audit, dict) or split_audit.get("passed") is not True:
                raise ValueError(f"DINOv2 split audit did not pass in {manifest_path}")
        manifest_hashes[angular_bin] = _sha256(manifest_path)

        reports = bin_root / "reports"
        cdf = pd.read_csv(reports / "summary_cdf.csv")
        counts = pd.read_csv(reports / "summary_match_counts.csv")
        expected_cdf_rows = (
            len(LAYERS) * len(expected_debias_ranks) * len(THRESHOLDS_M)
        )
        if len(cdf) != expected_cdf_rows or len(counts) != len(
            LAYERS
        ) * len(expected_debias_ranks):
            raise ValueError(f"Unexpected report rectangle under {reports}")
        if cdf.isna().any().any() or counts.isna().any().any():
            raise ValueError(f"Missing report values under {reports}")
        observed_thresholds = tuple(sorted(cdf["threshold_m"].unique()))
        if len(observed_thresholds) != len(THRESHOLDS_M) or not np.allclose(
            observed_thresholds, THRESHOLDS_M, atol=0.0, rtol=0.0
        ):
            raise ValueError(f"Unexpected thresholds under {reports}: {observed_thresholds}")
        metric_columns = [f"mutual_cdf_at_{candidate_k}" for candidate_k in range(1, MAX_K + 1)]
        group_columns = ["angular_bin", "layer", "debias_rank"]
        averaged = (
            cdf.groupby(group_columns, sort=True, as_index=False)[metric_columns]
            .mean()
            .sort_values(group_columns, kind="mergesort")
        )
        support = (
            cdf.groupby(group_columns, sort=True, as_index=False)
            .agg(
                threshold_count=("threshold_m", "nunique"),
                category_count=("category_count", "min"),
                pair_count=("pair_count", "min"),
            )
        )
        averaged = averaged.merge(support, on=group_columns, validate="one_to_one")
        if not (averaged["threshold_count"] == len(THRESHOLDS_M)).all():
            raise ValueError(f"A configuration is missing a threshold under {reports}")
        merged = averaged.merge(
            counts,
            on=group_columns,
            validate="one_to_one",
            suffixes=("_cdf", "_count"),
        )
        for _, row in merged.iterrows():
            if int(row["category_count_cdf"]) != 35 or int(row["pair_count_cdf"]) != 500:
                raise ValueError("Every curve cell must contain 35 objects and 500 pairs")
            for candidate_k in range(1, MAX_K + 1):
                rows.append(
                    {
                        "angular_bin": angular_bin,
                        "layer": int(row["layer"]),
                        "debias_rank": int(row["debias_rank"]),
                        "candidate_k": candidate_k,
                        "thresholds_m": "|".join(f"{value:g}" for value in THRESHOLDS_M),
                        "object_count": int(row["category_count_cdf"]),
                        "pair_count": int(row["pair_count_cdf"]),
                        "mean_mutual_association_count": float(
                            row[f"category_macro_mean_match_count_at_{candidate_k}"]
                        ),
                        "mutual_cdf": float(row[f"mutual_cdf_at_{candidate_k}"]),
                    }
                )
    points = pd.DataFrame(rows).sort_values(
        ["angular_bin", "layer", "debias_rank", "candidate_k"],
        kind="mergesort",
    )
    expected = len(BINS) * len(LAYERS) * len(expected_debias_ranks) * MAX_K
    if len(points) != expected:
        raise ValueError(f"Expected {expected} curve points, found {len(points)}")
    for _, curve in points.groupby(
        ["angular_bin", "layer", "debias_rank"], sort=True
    ):
        curve = curve.sort_values("candidate_k")
        if list(curve["candidate_k"]) != list(range(1, MAX_K + 1)):
            raise ValueError("A curve does not contain K=1 through 8")
        if np.any(np.diff(curve["mean_mutual_association_count"]) < 0):
            raise ValueError("Mutual association count must be monotonic in K")
        if np.any(np.diff(curve["mutual_cdf"]) < -1e-12):
            raise ValueError("Mutual recovery must be monotonic in K")
    return points.reset_index(drop=True), manifest_hashes


def _rank_differences(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rank_a, rank_b in zip(DEBIAS_RANKS[:-1], DEBIAS_RANKS[1:]):
        for angular_bin in BINS:
            for layer in LAYERS:
                left = points[
                    (points["angular_bin"] == angular_bin)
                    & (points["layer"] == layer)
                    & (points["debias_rank"] == rank_a)
                ].sort_values("candidate_k")
                right = points[
                    (points["angular_bin"] == angular_bin)
                    & (points["layer"] == layer)
                    & (points["debias_rank"] == rank_b)
                ].sort_values("candidate_k")
                count_a = left["mean_mutual_association_count"].to_numpy()
                count_b = right["mean_mutual_association_count"].to_numpy()
                cdf_a = left["mutual_cdf"].to_numpy()
                cdf_b = right["mutual_cdf"].to_numpy()
                same_k_cdf_delta = 100.0 * (cdf_b - cdf_a)
                same_k_count_delta = 100.0 * (count_b - count_a) / count_a
                lower = max(float(count_a.min()), float(count_b.min()))
                upper = min(float(count_a.max()), float(count_b.max()))
                grid = np.linspace(lower, upper, 201)
                curve_delta = 100.0 * (
                    np.interp(grid, count_b, cdf_b) - np.interp(grid, count_a, cdf_a)
                )
                rows.append(
                    {
                        "angular_bin": angular_bin,
                        "layer": layer,
                        "rank_a": rank_a,
                        "rank_b": rank_b,
                        "same_k_mean_cdf_delta_pp": float(same_k_cdf_delta.mean()),
                        "same_k_max_abs_cdf_delta_pp": float(
                            np.max(np.abs(same_k_cdf_delta))
                        ),
                        "same_k_mean_count_delta_percent": float(
                            same_k_count_delta.mean()
                        ),
                        "same_k_max_abs_count_delta_percent": float(
                            np.max(np.abs(same_k_count_delta))
                        ),
                        "common_support_curve_mean_delta_pp": float(curve_delta.mean()),
                        "common_support_curve_mae_pp": float(
                            np.mean(np.abs(curve_delta))
                        ),
                        "common_support_curve_max_abs_pp": float(
                            np.max(np.abs(curve_delta))
                        ),
                        "common_support_min_associations": lower,
                        "common_support_max_associations": upper,
                    }
                )
    return pd.DataFrame(rows)


def _layer_frontier_regret(points: pd.DataFrame) -> pd.DataFrame:
    """Measure each layer against the same-rank empirical upper-left frontier."""

    rows: list[dict[str, Any]] = []
    for rank in DEBIAS_RANKS:
        rank_points = points[points["debias_rank"] == rank]
        per_bin: dict[int, list[tuple[float, float]]] = {
            layer: [] for layer in LAYERS
        }
        for angular_bin in BINS:
            cell = rank_points[rank_points["angular_bin"] == angular_bin].sort_values(
                ["mean_mutual_association_count", "mutual_cdf"],
                ascending=[True, False],
                kind="mergesort",
            )
            counts = cell["mean_mutual_association_count"].to_numpy()
            best = np.maximum.accumulate(cell["mutual_cdf"].to_numpy())
            for layer in LAYERS:
                curve = cell[cell["layer"] == layer].sort_values("candidate_k")
                indices = np.searchsorted(
                    counts,
                    curve["mean_mutual_association_count"].to_numpy(),
                    side="right",
                ) - 1
                regret = 100.0 * (best[indices] - curve["mutual_cdf"].to_numpy())
                per_bin[layer].append((float(regret.mean()), float(regret.max())))
        for layer in LAYERS:
            mean_regrets = [value[0] for value in per_bin[layer]]
            max_regrets = [value[1] for value in per_bin[layer]]
            rows.append(
                {
                    "debias_rank": rank,
                    "layer": layer,
                    "mean_regret_pp": float(np.mean(mean_regrets)),
                    "worst_bin_mean_regret_pp": float(np.max(mean_regrets)),
                    "max_point_regret_pp": float(np.max(max_regrets)),
                    "all_observed_points_on_frontier": bool(
                        np.max(max_regrets) <= 1e-12
                    ),
                }
            )
    return pd.DataFrame(rows)


def _select_best_raw_layer(
    points: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select the DINOv2 raw layer by empirical upper-left frontier regret.

    For each bin and observed layer/K point ``(x, y)``, the empirical frontier
    is the best recovery among *all DINOv2 raw points in that bin* whose
    association count is at most ``x``.  Regret is measured in percentage
    points and averaged equally across the three bins and K=1..8 observations.
    The lexicographic tie-break is fully deterministic and intentionally ends
    with the smaller one-based layer number.
    """

    required_columns = {
        "angular_bin",
        "layer",
        "debias_rank",
        "candidate_k",
        "mean_mutual_association_count",
        "mutual_cdf",
    }
    missing = required_columns.difference(points.columns)
    if missing:
        raise ValueError(f"DINOv2 point table is missing columns: {sorted(missing)}")
    if points.empty:
        raise ValueError("Cannot select a layer from an empty DINOv2 point table")
    if set(int(value) for value in points["debias_rank"].unique()) != {0}:
        raise ValueError("DINOv2 layer selection accepts raw rank 0 only")
    observed_bins = tuple(sorted(str(value) for value in points["angular_bin"].unique()))
    if observed_bins != tuple(sorted(BINS)):
        raise ValueError(f"DINOv2 layer selection requires bins {BINS}, got {observed_bins}")
    observed_layers = tuple(sorted(int(value) for value in points["layer"].unique()))
    if observed_layers != LAYERS:
        raise ValueError(f"DINOv2 layer selection requires layers {LAYERS}")

    point_rows: list[dict[str, Any]] = []
    for angular_bin in BINS:
        cell = points[points["angular_bin"] == angular_bin].sort_values(
            ["mean_mutual_association_count", "mutual_cdf", "layer", "candidate_k"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        if len(cell) != len(LAYERS) * MAX_K:
            raise ValueError(f"Incomplete DINOv2 raw rectangle in bin {angular_bin}")
        if cell[["layer", "candidate_k"]].duplicated().any():
            raise ValueError(
                f"Duplicate DINOv2 layer/K observations in bin {angular_bin}"
            )
        counts = cell["mean_mutual_association_count"].to_numpy(dtype=np.float64)
        recovery = cell["mutual_cdf"].to_numpy(dtype=np.float64)
        if not np.isfinite(counts).all() or not np.isfinite(recovery).all():
            raise ValueError("DINOv2 layer-selection inputs must be finite")
        if np.any(counts < 0.0) or np.any((recovery < 0.0) | (recovery > 1.0)):
            raise ValueError("DINOv2 layer-selection inputs are outside valid ranges")
        frontier = np.maximum.accumulate(recovery)
        for layer in LAYERS:
            curve = cell[cell["layer"] == layer].sort_values("candidate_k")
            if list(curve["candidate_k"]) != list(range(1, MAX_K + 1)):
                raise ValueError(
                    f"DINOv2 layer {layer}/{angular_bin} lacks K=1 through {MAX_K}"
                )
            curve_counts = curve["mean_mutual_association_count"].to_numpy(
                dtype=np.float64
            )
            indices = np.searchsorted(counts, curve_counts, side="right") - 1
            regrets = 100.0 * (
                frontier[indices]
                - curve["mutual_cdf"].to_numpy(dtype=np.float64)
            )
            if np.any(regrets < -1e-10):
                raise AssertionError("Empirical frontier regret cannot be negative")
            regrets = np.maximum(regrets, 0.0)
            for candidate_k, count, recall, regret in zip(
                curve["candidate_k"],
                curve_counts,
                curve["mutual_cdf"],
                regrets,
            ):
                point_rows.append(
                    {
                        "angular_bin": angular_bin,
                        "layer": layer,
                        "candidate_k": int(candidate_k),
                        "mean_mutual_association_count": float(count),
                        "mutual_cdf": float(recall),
                        "frontier_regret_pp": float(regret),
                    }
                )

    point_regret = pd.DataFrame(point_rows).sort_values(
        ["layer", "angular_bin", "candidate_k"], kind="mergesort"
    )
    layer_rows: list[dict[str, Any]] = []
    for layer in LAYERS:
        layer_points = point_regret[point_regret["layer"] == layer]
        bin_means = (
            layer_points.groupby("angular_bin", sort=True)["frontier_regret_pp"]
            .mean()
            .to_dict()
        )
        layer_rows.append(
            {
                "layer": layer,
                "mean_regret_pp": float(layer_points["frontier_regret_pp"].mean()),
                "worst_bin_mean_regret_pp": float(max(bin_means.values())),
                "max_point_regret_pp": float(layer_points["frontier_regret_pp"].max()),
                **{
                    f"bin_{angular_bin}_mean_regret_pp": float(bin_means[angular_bin])
                    for angular_bin in BINS
                },
            }
        )
    layer_regret = pd.DataFrame(layer_rows).sort_values(
        [
            "mean_regret_pp",
            "worst_bin_mean_regret_pp",
            "max_point_regret_pp",
            "layer",
        ],
        kind="mergesort",
    )
    selected = layer_regret.iloc[0]
    selection = {
        "best_layer": int(selected["layer"]),
        "selected_layer": int(selected["layer"]),
        "selected_mean_regret_pp": float(selected["mean_regret_pp"]),
        "selected_worst_bin_mean_regret_pp": float(
            selected["worst_bin_mean_regret_pp"]
        ),
        "selected_max_point_regret_pp": float(selected["max_point_regret_pp"]),
        "selection_metric": (
            "mean empirical upper-left frontier regret over 3 viewpoint bins "
            "and directly observed K=1..8 points"
        ),
        "frontier_definition": (
            "within each bin, maximum raw DINOv2 mutual recall among all "
            "layer/K points with mean mutual association count <= query point count"
        ),
        "tie_break": [
            "lowest mean_regret_pp",
            "lowest worst_bin_mean_regret_pp",
            "lowest max_point_regret_pp",
            "lowest one-based layer number",
        ],
        "bin_weighting": "equal",
        "candidate_k_weighting": "equal over K=1..8",
        "rank": 0,
        "correction_mode": "none",
    }
    return layer_regret.reset_index(drop=True), selection


def _plot_rank_sweep(points: pd.DataFrame, output_dir: Path) -> None:
    _configure_type1_fonts()
    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "legend.fontsize": 5.6,
            "savefig.dpi": 220,
        }
    )
    figure, axes = plt.subplots(3, 3, figsize=(7.15, 5.45), sharex=True, sharey=True)
    for axis, layer in zip(axes.flat, LAYERS):
        layer_frame = points[points["layer"] == layer]
        for angular_bin in BINS:
            bin_frame = layer_frame[layer_frame["angular_bin"] == angular_bin]
            for rank in DEBIAS_RANKS:
                curve = bin_frame[bin_frame["debias_rank"] == rank].sort_values(
                    "candidate_k"
                )
                x = curve["mean_mutual_association_count"].to_numpy()
                y = 100.0 * curve["mutual_cdf"].to_numpy()
                axis.plot(
                    x,
                    y,
                    color=RANK_COLORS[rank],
                    linestyle=BIN_LINESTYLES[angular_bin],
                    linewidth=1.0,
                    alpha=0.84,
                    marker="o" if rank == 0 else None,
                    markersize=2.4 if rank == 0 else 0.0,
                    markerfacecolor=RANK_COLORS[rank],
                    markeredgecolor="white",
                    markeredgewidth=0.25,
                    zorder=3 if rank == 0 else 2,
                )
        axis.set_title(f"Layer {layer}", pad=2.0)
        axis.grid(alpha=0.22)
        axis.tick_params(axis="both", labelsize=6.2)
    for axis in axes[-1, :]:
        axis.set_xlabel("Mean mutual associations")
    axes[1, 0].set_ylabel(r"Mean correspondence recall (\%)")

    rank_handles = [
        Line2D(
            [0],
            [0],
            color=RANK_COLORS[rank],
            linewidth=1.2,
            marker="o" if rank == 0 else None,
            markersize=2.6 if rank == 0 else 0.0,
            label=rf"$s={rank}$",
        )
        for rank in DEBIAS_RANKS
    ]
    bin_handles = [
        Line2D(
            [0],
            [0],
            color="#303030",
            linestyle=BIN_LINESTYLES[angular_bin],
            linewidth=1.3,
            label=_angle_label(angular_bin),
        )
        for angular_bin in BINS
    ]
    legend_items = rank_handles + bin_handles
    figure.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=len(legend_items),
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#b8b8b8",
        fontsize=6.2,
        borderpad=0.3,
        handlelength=1.7,
        columnspacing=0.65,
        handletextpad=0.3,
        borderaxespad=0.0,
    )
    figure.subplots_adjust(left=0.085, right=0.995, bottom=0.15, top=0.985,
                           hspace=0.19, wspace=0.12)
    stem = output_dir / "mknn_association_cdf_all_bins_mean_1_2_5cm"
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.01)
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


def _curve_point(
    frame: pd.DataFrame,
    angular_bin: str,
    layer: int,
    candidate_k: int,
) -> tuple[float, float]:
    point = frame[
        (frame["angular_bin"] == angular_bin)
        & (frame["layer"] == layer)
        & (frame["candidate_k"] == candidate_k)
    ]
    if len(point) != 1:
        raise ValueError(
            f"Expected one point for {angular_bin}/L{layer}/K={candidate_k}, "
            f"found {len(point)}"
        )
    row = point.iloc[0]
    return (
        float(row["mean_mutual_association_count"]),
        100.0 * float(row["mutual_cdf"]),
    )


def _curve_gain_pp(
    frame: pd.DataFrame,
    angular_bin: str,
    layer: int,
    start_k: int = 1,
    end_k: int = 5,
) -> float:
    """Return the directly observed correspondence-recall gain in pp."""
    start = _curve_point(frame, angular_bin, layer, start_k)
    end = _curve_point(frame, angular_bin, layer, end_k)
    return end[1] - start[1]


def _annotate_selected_layer_gains(
    axis: plt.Axes,
    frame: pd.DataFrame,
    layer: int,
) -> None:
    """Mark each selected-layer K=1 to K=5 gain on a compact panel."""
    curvatures = {"0-40": -0.08, "40-80": 0.0, "80-120": 0.08}
    for angular_bin in BINS:
        start = _curve_point(frame, angular_bin, layer, 1)
        end = _curve_point(frame, angular_bin, layer, 5)
        gain = _curve_gain_pp(frame, angular_bin, layer)
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "connectionstyle": f"arc3,rad={curvatures[angular_bin]}",
                "color": "black",
                "linewidth": 0.8,
                "alpha": 0.92,
                "shrinkA": 3.5,
                "shrinkB": 4.0,
            },
            zorder=6,
        )
        axis.annotate(
            rf"$+{gain:.1f}$ pp",
            xy=end,
            xycoords="data",
            xytext=(0.0, 5.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.4,
            fontweight="bold",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
            },
            zorder=7,
        )


def _annotate_rank200_selection(axis: plt.Axes, fixed: pd.DataFrame) -> None:
    """Annotate the selected L19 curve without obscuring the data trajectories."""

    gain_text_positions = {
        "0-40": (0.46, 0.955),
        "40-80": (0.45, 0.75),
        "80-120": (0.42, 0.525),
    }
    arrow_curvatures = {"0-40": 0.12, "40-80": 0.10, "80-120": 0.08}
    for angular_bin in BINS:
        start = _curve_point(fixed, angular_bin, 19, 1)
        end = _curve_point(fixed, angular_bin, 19, 5)
        gain = end[1] - start[1]
        axis.annotate(
            rf"$K=1\!\rightarrow\!5$: $+{gain:.1f}$ pp",
            xy=end,
            xycoords="data",
            xytext=gain_text_positions[angular_bin],
            textcoords="axes fraction",
            ha="center",
            va="center",
            fontsize=6.6,
            color=LAYER_COLORS[19],
            arrowprops={
                "arrowstyle": "-|>",
                "connectionstyle": f"arc3,rad={arrow_curvatures[angular_bin]}",
                "color": LAYER_COLORS[19],
                "linewidth": 0.8,
                "shrinkA": 2.0,
                "shrinkB": 4.0,
            },
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
            zorder=6,
        )

    band_labels = (
        (0.985, 0.85, r"small viewpoint ($0$--$40^\circ$)"),
        (0.985, 0.68, r"moderate viewpoint ($40$--$80^\circ$)"),
        (0.985, 0.51, r"large viewpoint ($80$--$120^\circ$)"),
    )
    for x, y, label in band_labels:
        axis.text(
            x,
            y,
            label,
            transform=axis.transAxes,
            ha="right" if x > 0.9 else "center",
            va="top",
            fontsize=6.8,
            fontstyle="italic",
            color="#3f3f3f",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
            zorder=6,
        )

    l19_k8 = _curve_point(fixed, "80-120", 19, 8)
    l24_k8 = _curve_point(fixed, "80-120", 24, 8)
    gap = l19_k8[1] - l24_k8[1]
    axis.annotate(
        "",
        xy=l19_k8,
        xytext=l24_k8,
        arrowprops={
            "arrowstyle": "<->",
            "color": "#333333",
            "linewidth": 0.85,
            "shrinkA": 4.0,
            "shrinkB": 4.0,
        },
        zorder=5,
    )
    midpoint = (
        0.5 * (l19_k8[0] + l24_k8[0]),
        0.5 * (l19_k8[1] + l24_k8[1]),
    )
    axis.annotate(
        rf"L24$\rightarrow$L19 at $K=8$: $+{gap:.2f}$ pp",
        xy=midpoint,
        xycoords="data",
        xytext=(-20.0, -30.0),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#333333",
        bbox={
            "boxstyle": "round,pad=0.14",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
        },
        zorder=6,
    )


def _plot_layer_sweep(
    points: pd.DataFrame,
    output_dir: Path,
    fixed_rank: int,
    displayed_layers: tuple[int, ...],
    highlighted_layers: frozenset[int],
) -> None:
    fixed = points[points["debias_rank"] == fixed_rank]
    _configure_type1_fonts()
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 7.0,
            "savefig.dpi": 220,
        }
    )
    figure, axis = plt.subplots(figsize=(7.15, 3.5))
    for angular_bin in BINS:
        cell = fixed[fixed["angular_bin"] == angular_bin]
        for layer in displayed_layers:
            curve = cell[cell["layer"] == layer].sort_values("candidate_k")
            x = curve["mean_mutual_association_count"].to_numpy()
            y = 100.0 * curve["mutual_cdf"].to_numpy()
            highlighted = layer in highlighted_layers
            axis.plot(
                x,
                y,
                color=LAYER_COLORS[layer],
                linestyle="-" if highlighted else OTHER_LAYER_STYLE,
                marker=BIN_MARKERS[angular_bin],
                markersize=3.2,
                markeredgecolor="white",
                markeredgewidth=0.25,
                linewidth=1.65 if highlighted else 1.15,
                alpha=0.96 if highlighted else 0.82,
                zorder=3 if highlighted else 2,
            )
    axis.set_xlabel("Mean mutual associations")
    axis.set_ylabel(r"Mean correspondence recall (\%)")
    axis.set_xlim(left=0.0)
    axis.set_ylim(8.0, 92.0)
    axis.grid(True, color="#d8d8d8", linewidth=0.55, alpha=0.75)
    if fixed_rank == 200 and {19, 24} <= set(displayed_layers):
        _annotate_rank200_selection(axis, fixed)
    layer_handles = [
        Line2D(
            [0],
            [0],
            color=LAYER_COLORS[layer],
            linestyle="-" if layer in highlighted_layers else OTHER_LAYER_STYLE,
            linewidth=1.8 if layer in highlighted_layers else 1.3,
            label=f"L{layer} (last)" if layer == max(LAYERS) else f"L{layer}",
        )
        for layer in displayed_layers
    ]
    bin_handles = [
        Line2D(
            [0],
            [0],
            color="#303030",
            linestyle="None",
            marker=BIN_MARKERS[angular_bin],
            markersize=4.0,
            label=_angle_label(angular_bin),
        )
        for angular_bin in BINS
    ]
    legend_items = layer_handles + bin_handles
    legend_columns = len(legend_items) if len(legend_items) <= 8 else 6
    if len(legend_items) > legend_columns:
        # Matplotlib fills multi-column legends down each column.  Interleave
        # row-major items so the rendered rows still read left to right.
        legend_rows = (len(legend_items) + legend_columns - 1) // legend_columns
        legend_items = [
            legend_items[row * legend_columns + column]
            for column in range(legend_columns)
            for row in range(legend_rows)
            if row * legend_columns + column < len(legend_items)
        ]

    axis.legend(
        handles=legend_items,
        loc="lower right",
        ncol=legend_columns,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#b8b8b8",
        handlelength=1.15,
        columnspacing=0.9,
        handletextpad=0.35,
        borderpad=0.35,
    )
    figure.subplots_adjust(left=0.10, right=0.995, bottom=0.17, top=0.98)
    stem = output_dir / f"layer_sweep_rank{fixed_rank}_mean_1_2_5cm"
    figure.savefig(
        stem.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.01
    )
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


def _plot_dinov3_dinov2_layer_sweep(
    dinov3_points: pd.DataFrame,
    dinov2_points: pd.DataFrame,
    output_dir: Path,
    *,
    displayed_dinov3_layers: tuple[int, ...],
    highlighted_dinov3_layers: frozenset[int],
    selected_dinov2_layer: int,
) -> None:
    """Draw the compatibility-named DINOv3/DINOv2 two-panel comparison."""

    fixed = dinov3_points[dinov3_points["debias_rank"] == 200]
    raw = dinov2_points[dinov2_points["debias_rank"] == 0]
    _configure_type1_fonts()
    plt.rcParams.update(
        {
            "font.size": 7.1,
            "axes.labelsize": 8.2,
            "legend.fontsize": 5.5,
            "savefig.dpi": 220,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.8),
        sharey=True,
        gridspec_kw={"wspace": 0.12},
    )
    left, right = axes

    for angular_bin in BINS:
        cell = fixed[fixed["angular_bin"] == angular_bin]
        for layer in displayed_dinov3_layers:
            curve = cell[cell["layer"] == layer].sort_values("candidate_k")
            highlighted = layer in highlighted_dinov3_layers
            left.plot(
                curve["mean_mutual_association_count"],
                100.0 * curve["mutual_cdf"],
                color=LAYER_COLORS[layer],
                linestyle="-" if highlighted else OTHER_LAYER_STYLE,
                marker=BIN_MARKERS[angular_bin],
                markersize=2.6,
                markeredgecolor="white",
                markeredgewidth=0.22,
                linewidth=1.45 if highlighted else 1.0,
                alpha=0.96 if highlighted else 0.80,
                zorder=3 if highlighted else 2,
            )
    left.set_title(r"DINOv3 ViT-L/16, debias $s=200$", pad=3.0)
    left.set_xlabel("Mean mutual associations")
    left.set_ylabel(r"Mean correspondence recall (\%)")
    left.set_xlim(left=0.0)
    left.grid(True, color="#d8d8d8", linewidth=0.5, alpha=0.75)

    for angular_bin in BINS:
        cell = raw[raw["angular_bin"] == angular_bin]
        for layer in LAYERS:
            curve = cell[cell["layer"] == layer].sort_values("candidate_k")
            selected = layer == selected_dinov2_layer
            right.plot(
                curve["mean_mutual_association_count"],
                100.0 * curve["mutual_cdf"],
                color=LAYER_COLORS[layer],
                linestyle="-" if selected else OTHER_LAYER_STYLE,
                marker=BIN_MARKERS[angular_bin],
                markersize=2.5,
                markeredgecolor="white",
                markeredgewidth=0.22,
                linewidth=1.7 if selected else 0.95,
                alpha=1.0 if selected else 0.76,
                zorder=4 if selected else 2,
            )
    right.set_title(r"DINOv2 ViT-L/14, raw (no debias)", pad=3.0)
    right.set_xlabel("Mean mutual associations")
    right.set_xlim(left=0.0)
    right.grid(True, color="#d8d8d8", linewidth=0.5, alpha=0.75)

    all_recall = np.concatenate(
        [
            100.0 * fixed["mutual_cdf"].to_numpy(dtype=np.float64),
            100.0 * raw["mutual_cdf"].to_numpy(dtype=np.float64),
        ]
    )
    lower = max(0.0, float(np.floor(all_recall.min() / 5.0) * 5.0 - 2.0))
    upper = min(100.0, float(np.ceil(all_recall.max() / 5.0) * 5.0 + 2.0))
    left.set_ylim(lower, upper)
    _annotate_selected_layer_gains(left, fixed, 19)
    _annotate_selected_layer_gains(right, raw, selected_dinov2_layer)

    left_handles = [
        Line2D(
            [0],
            [0],
            color=LAYER_COLORS[layer],
            linestyle="-" if layer in highlighted_dinov3_layers else OTHER_LAYER_STYLE,
            linewidth=1.6 if layer in highlighted_dinov3_layers else 1.1,
            label=(
                f"L{layer} (selected)"
                if layer == 19
                else f"L{layer}" + (" (last)" if layer == max(LAYERS) else "")
            ),
        )
        for layer in displayed_dinov3_layers
    ]
    right_handles = [
        Line2D(
            [0],
            [0],
            color=LAYER_COLORS[layer],
            linestyle="-" if layer == selected_dinov2_layer else OTHER_LAYER_STYLE,
            linewidth=1.8 if layer == selected_dinov2_layer else 1.0,
            label=(
                f"L{layer} (selected)"
                if layer == selected_dinov2_layer
                else f"L{layer}" + (" (last)" if layer == max(LAYERS) else "")
            ),
        )
        for layer in LAYERS
    ]
    bin_handles = [
        Line2D(
            [0],
            [0],
            color="#303030",
            linestyle="None",
            marker=BIN_MARKERS[angular_bin],
            markersize=3.5,
            label=_angle_label(angular_bin),
        )
        for angular_bin in BINS
    ]

    def row_major(items: list[Line2D], columns: int) -> list[Line2D]:
        rows = (len(items) + columns - 1) // columns
        return [
            items[row * columns + column]
            for column in range(columns)
            for row in range(rows)
            if row * columns + column < len(items)
        ]

    legend_style = {
        "frameon": True,
        "framealpha": 0.94,
        "facecolor": "white",
        "edgecolor": "#b8b8b8",
        "handlelength": 1.2,
        "columnspacing": 0.7,
        "handletextpad": 0.3,
        "borderpad": 0.3,
        "borderaxespad": 0.0,
    }
    left_columns = 3 if len(left_handles) + len(bin_handles) > 8 else 2
    left.legend(
        handles=row_major(left_handles + bin_handles, left_columns),
        loc="lower right",
        bbox_to_anchor=(0.985, 0.025),
        ncol=left_columns,
        **legend_style,
    )
    right.legend(
        handles=row_major(right_handles + bin_handles, 3),
        loc="lower right",
        bbox_to_anchor=(0.985, 0.025),
        ncol=3,
        **legend_style,
    )
    figure.subplots_adjust(left=0.09, right=0.995, bottom=0.16, top=0.91)
    # Preserve the established filename consumed by the paper/build scripts.
    stem = output_dir / "layer_sweep_rank200_mean_1_2_5cm"
    figure.savefig(
        stem.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.01
    )
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


def _prepare_output(path: Path, filenames: tuple[str, ...], overwrite: bool) -> Path:
    resolved = path.resolve()
    existing = [resolved / filename for filename in filenames if (resolved / filename).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {existing[0]}; pass --overwrite to regenerate"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _selection_provenance(
    evaluation_root: Path, manifest_hashes: dict[str, str]
) -> dict[str, Any]:
    manifests: dict[str, dict[str, Any]] = {}
    for angular_bin in BINS:
        path = evaluation_root / f"bin_{angular_bin}" / "evaluation_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        reports = path.parent / "reports"
        report_hashes = {
            filename: _sha256(reports / filename)
            for filename in ("summary_cdf.csv", "summary_match_counts.csv")
        }
        manifests[angular_bin] = {
            "path": str(path.resolve()),
            "sha256": manifest_hashes[angular_bin],
            "protocol_fingerprints": manifest.get("protocol_fingerprints"),
            "pair_file_sha256": manifest.get("pair_file_sha256"),
            "estimation_split_sha256": manifest.get("estimation_split_sha256"),
            "weights_id": manifest.get("weights_id"),
            "descriptor_provenance": manifest.get(
                "descriptor_provenance",
                {
                    "source_revision": "unknown",
                    "source_dirty": "unknown",
                    "backbone": {
                        "model_family": "dinov2",
                        "patch_size": manifest.get("patch_size", 14),
                        "descriptor_dim": 1024,
                        "register_tokens": 4,
                        "correction": manifest.get("correction_mode", "none"),
                    },
                },
            ),
            "dataset_snapshot_id": manifest.get("dataset_snapshot_id"),
            "descriptor_snapshot_ids": manifest.get("descriptor_snapshot_ids"),
            "numeric_report_sha256": report_hashes,
        }
    return manifests


def _uniform_descriptor_provenance(
    manifests: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    expected_backbone = {
        "model_family": "dinov2",
        "patch_size": 14,
        "descriptor_dim": 1024,
        "register_tokens": 4,
        "correction": "none",
    }
    fields = []
    for angular_bin in BINS:
        provenance = manifests[angular_bin]["descriptor_provenance"]
        if not isinstance(provenance, dict):
            raise ValueError(f"Invalid descriptor provenance for bin {angular_bin}")
        field = {
            "source_revision": provenance.get("source_revision", "unknown"),
            "source_dirty": provenance.get("source_dirty", "unknown"),
            "backbone": provenance.get("backbone"),
        }
        if field["backbone"] != expected_backbone:
            raise ValueError(
                f"Invalid DINOv2 descriptor backbone provenance for bin {angular_bin}"
            )
        if not isinstance(field["source_revision"], str) or not field["source_revision"]:
            raise ValueError(f"Invalid descriptor source revision for bin {angular_bin}")
        if field["source_dirty"] != "unknown" and not isinstance(
            field["source_dirty"], bool
        ):
            raise ValueError(f"Invalid descriptor source dirty flag for bin {angular_bin}")
        fields.append(field)
    canonical = {json.dumps(value, sort_keys=True) for value in fields}
    if len(canonical) != 1:
        raise ValueError(
            "DINOv2 bins do not share one descriptor source/model provenance"
        )
    weights_ids = {manifests[angular_bin].get("weights_id") for angular_bin in BINS}
    if len(weights_ids) != 1 or None in weights_ids or "" in weights_ids:
        raise ValueError("DINOv2 bins do not share one checkpoint identity")
    return {**fields[0], "weights_id": next(iter(weights_ids))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument(
        "--dinov2-evaluation-root",
        type=Path,
        help=(
            "Complete raw DINOv2 ViT-L/14 layer-16..24 evaluation. When supplied, "
            "the compatibility-named rank-200 figure becomes a two-panel DINOv3/DINOv2 "
            "comparison and best_layer.json is emitted."
        ),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        required=True,
        help="Directory for derived CSV/JSON reports (normally under artifacts/).",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        required=True,
        help="Directory for PDF/PNG only (normally under outputs/paper_figures/).",
    )
    parser.add_argument(
        "--fixed-rank",
        type=int,
        nargs="+",
        default=[200],
        help="One or more ranks for fixed-rank layer figures (default: 200).",
    )
    parser.add_argument(
        "--layer-sweep-layers",
        type=int,
        nargs="+",
        default=list(LAYER_SWEEP_LAYERS),
        help="Layers shown in the fixed-rank layer-sweep figure.",
    )
    parser.add_argument(
        "--highlight-layers",
        type=int,
        nargs="+",
        default=list(HIGHLIGHT_LAYERS),
        help="Layer curves drawn with a solid line in the layer-sweep figure.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace this program's existing report and figure files.",
    )
    parser.add_argument(
        "--tex",
        action="store_true",
        help="Use LaTeX/newtx text rendering for paper-copy typography.",
    )
    args = parser.parse_args()
    TYPE1_FONT_SETTINGS["text.usetex"] = args.tex
    if args.tex:
        TYPE1_FONT_SETTINGS["text.latex.preamble"] = r"\usepackage{newtxtext,newtxmath}"
    fixed_ranks = tuple(dict.fromkeys(args.fixed_rank))
    unknown_ranks = set(fixed_ranks) - set(DEBIAS_RANKS)
    if unknown_ranks:
        raise ValueError(f"fixed ranks must be selected from {DEBIAS_RANKS}")
    displayed_layers = tuple(args.layer_sweep_layers)
    highlighted_layers = frozenset(args.highlight_layers)
    if len(displayed_layers) != len(set(displayed_layers)):
        raise ValueError("layer-sweep layers must not contain duplicates")
    unknown_layers = set(displayed_layers) - set(LAYERS)
    if unknown_layers:
        raise ValueError(f"unknown layer-sweep layers: {sorted(unknown_layers)}")
    if not highlighted_layers <= set(displayed_layers):
        raise ValueError("highlight layers must be included in layer-sweep layers")
    evaluation_root = args.evaluation_root.resolve()
    report_names = (
        "curve_points.csv",
        "adjacent_rank_curve_differences.csv",
        "layer_frontier_regret.csv",
        "protocol.json",
    )
    if args.dinov2_evaluation_root is not None:
        report_names += (
            "dinov2_raw_curve_points.csv",
            "dinov2_raw_layer_frontier_regret.csv",
            "best_layer.json",
        )
    figure_names = tuple(
        [
            "mknn_association_cdf_all_bins_mean_1_2_5cm.pdf",
            "mknn_association_cdf_all_bins_mean_1_2_5cm.png",
        ]
        + [
            f"layer_sweep_rank{rank}_mean_1_2_5cm.{suffix}"
            for rank in fixed_ranks
            for suffix in ("pdf", "png")
        ]
    )
    report_output = _prepare_output(args.report_output, report_names, args.overwrite)
    figure_output = _prepare_output(args.figure_output, figure_names, args.overwrite)

    points, manifest_hashes = _load_points(evaluation_root)
    differences = _rank_differences(points)
    layer_regret = _layer_frontier_regret(points)
    _atomic_text(report_output / "curve_points.csv", _csv_text(points))
    _atomic_text(
        report_output / "adjacent_rank_curve_differences.csv", _csv_text(differences)
    )
    _atomic_text(report_output / "layer_frontier_regret.csv", _csv_text(layer_regret))
    _plot_rank_sweep(points, figure_output)
    dinov2_protocol: dict[str, Any] | None = None
    if args.dinov2_evaluation_root is not None:
        if 200 not in fixed_ranks:
            raise ValueError(
                "--dinov2-evaluation-root requires --fixed-rank to include 200 so the "
                "compatibility-named two-panel figure can be written"
            )
        dinov2_root = args.dinov2_evaluation_root.resolve()
        dinov2_points, dinov2_manifest_hashes = _load_points(
            dinov2_root,
            expected_model_name=DINOV2_MODEL_NAME,
            expected_debias_ranks=DINOV2_RAW_RANKS,
        )
        dinov2_layer_regret, selection = _select_best_raw_layer(dinov2_points)
        input_manifests = _selection_provenance(dinov2_root, dinov2_manifest_hashes)
        selection_report = {
            "schema_version": 1,
            "model_name": DINOV2_MODEL_NAME,
            "model_family": "dinov2",
            "patch_size": 14,
            "layers": list(LAYERS),
            "candidate_k": list(range(1, MAX_K + 1)),
            "thresholds_m": list(THRESHOLDS_M),
            **selection,
            "descriptor_provenance": _uniform_descriptor_provenance(input_manifests),
            "input_manifests": input_manifests,
            "selection_script_sha256": _sha256(Path(__file__).resolve()),
        }
        _atomic_text(
            report_output / "dinov2_raw_curve_points.csv", _csv_text(dinov2_points)
        )
        _atomic_text(
            report_output / "dinov2_raw_layer_frontier_regret.csv",
            _csv_text(dinov2_layer_regret),
        )
        _write_selection_report(report_output / "best_layer.json", selection_report)
        _plot_dinov3_dinov2_layer_sweep(
            points,
            dinov2_points,
            figure_output,
            displayed_dinov3_layers=displayed_layers,
            highlighted_dinov3_layers=highlighted_layers,
            selected_dinov2_layer=int(selection["selected_layer"]),
        )
        dinov2_protocol = {
            "evaluation_root": str(dinov2_root),
            "model_name": DINOV2_MODEL_NAME,
            "debias_ranks": [0],
            "correction_mode": "none",
            "source_manifest_sha256": dinov2_manifest_hashes,
            "selected_layer": int(selection["selected_layer"]),
            "selection_report": "best_layer.json",
        }
    for fixed_rank in fixed_ranks:
        if fixed_rank == 200 and dinov2_protocol is not None:
            continue
        _plot_layer_sweep(
            points,
            figure_output,
            fixed_rank,
            displayed_layers,
            highlighted_layers,
        )
    protocol = {
        "schema_version": 1,
        "evaluation_root": str(evaluation_root),
        "source_manifest_sha256": manifest_hashes,
        "script_sha256": _sha256(Path(__file__).resolve()),
        "thresholds_m": list(THRESHOLDS_M),
        "threshold_aggregation": "equal arithmetic mean after pair-to-object macro aggregation",
        "candidate_k": list(range(1, MAX_K + 1)),
        "layers": list(LAYERS),
        "debias_ranks": list(DEBIAS_RANKS),
        "fixed_rank_layer_figures": list(fixed_ranks),
        "fixed_rank_layer_figure_layers": list(displayed_layers),
        "fixed_rank_layer_figure_solid_layers": sorted(highlighted_layers),
        "association_count": "uncapped category-macro mean mutual edge count",
        "recovery": "equal mean of pair-to-object macro conditional mutual CDF at 1/2/5 cm",
        "point_semantics": "one directly observed (association count, CDF) pair per K",
        "summary_row_count": len(points),
        "rank_change_row_count": len(differences),
        "layer_frontier_regret_row_count": len(layer_regret),
        "dinov2_raw_layer_sweep": dinov2_protocol,
        "rank200_figure_layout": (
            {
                "left_panel": "retained DINOv3 rank-200 numeric curves and legend semantics",
                "right_panel": "all DINOv2 raw layers 16-24; selected layer highlighted",
                "shared_y_axis": True,
                "independent_x_axes": True,
                "legends": "independent per panel, placed in the lower-right blank region",
                "selected_layer_k1_to_k5_gain_annotations": {
                    "start_k": 1,
                    "end_k": 5,
                    "units": "percentage points",
                    "display_decimals": 1,
                    "dinov3": {
                        "layer": 19,
                        "gain_pp_by_bin": {
                            angular_bin: _curve_gain_pp(
                                points[points["debias_rank"] == 200],
                                angular_bin,
                                19,
                            )
                            for angular_bin in BINS
                        },
                    },
                    "dinov2": {
                        "layer": int(dinov2_protocol["selected_layer"]),
                        "gain_pp_by_bin": {
                            angular_bin: _curve_gain_pp(
                                dinov2_points,
                                angular_bin,
                                int(dinov2_protocol["selected_layer"]),
                            )
                            for angular_bin in BINS
                        },
                    },
                },
                "omitted_legacy_dinov3_callouts": (
                    "viewpoint-band labels and the L24-to-L19 K=8 gap are omitted "
                    "from the compact two-panel layout"
                ),
            }
            if dinov2_protocol is not None
            else None
        ),
    }
    _atomic_text(
        report_output / "protocol.json",
        json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(
        f"Wrote {len(points)} cap-free curve points, {len(differences)} rank-change "
        f"rows, {len(layer_regret)} layer-regret rows under {report_output}, and "
        f"{len(figure_names)} figure files under {figure_output}"
    )


if __name__ == "__main__":
    main()
