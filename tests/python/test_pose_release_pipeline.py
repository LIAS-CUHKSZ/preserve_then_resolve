from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from evaluation.estimation import run_pose_gms_selection as gms
from evaluation.estimation import run_pose_matching as pose_matching


def test_explicit_pipeline_parser_and_seeded_config(tmp_path: Path) -> None:
    mnn = gms._parse_pipeline("mnn:CM")
    proposed = gms._parse_pipeline("gms:4:20:HCM_MC")
    assert mnn.label == "mnn/CM"
    assert proposed.label == "gms_t4_g20/HCM_MC"
    with pytest.raises(argparse.ArgumentTypeError, match="MNN baseline"):
        gms._parse_pipeline("mnn:HCM")

    task = gms.EstimatorTask(
        subset=gms.Subset("Fixture", "root", "subset", 1),
        variant_label=proposed.variant_label,
        variant_dir=proposed.variant_dir,
        mode=proposed.mode,
        max_iterations=100000,
        seed=4,
    )
    config = gms._config_text(
        results_root=tmp_path,
        task=task,
        layer=19,
        debias_rank=200,
        method_prefix="combined_interpolate_v3",
    )
    assert "seed=4\n" in config
    assert "output_csv=pose_seed4_iter100000.csv" in config
    assert gms._result_filename("HCM_MC", 100000, 4) == (
        "HCM_MC_pose_seed4_iter100000_q_ub_0.30.csv"
    )


def test_filter_rejects_duplicate_gms_candidates_before_loading_data(
    tmp_path: Path,
) -> None:
    candidate = gms.GmsVariant(4.0, 20)
    args = argparse.Namespace(
        candidates=[candidate, candidate],
        thresholds=None,
        grids=None,
        input_manifest=tmp_path / "missing.json",
        datasets=[],
    )
    with pytest.raises(ValueError, match="Duplicate GMS candidates"):
        gms.run_filter(args)


def test_pose_driver_uses_separate_keypoint_cache_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subset = pose_matching.Subset(
        "ScanNet",
        "scannet_resized",
        "subset",
        tmp_path / "images",
        tmp_path / "pairs.csv",
        False,
    )
    keypoint_root = tmp_path / "keypoints"
    args = argparse.Namespace(
        opencv_site_packages=None,
        lightglue_source=None,
        rss_root=tmp_path,
        device="cpu",
        input_manifest=tmp_path / "inputs.json",
        groups=["scannet"],
        datasets=["ScanNet"],
        results_root=tmp_path / "results",
        basis_root=tmp_path / "basis",
        cache_root=tmp_path / "dino_cache",
        keypoint_cache_root=keypoint_root,
        max_ks=(1, 5),
        association_upperbound=2048,
        superpoint_keypoints=2048,
        stage="extract-superpoint",
        batch_size=4,
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(pose_matching, "parse_args", lambda: args)
    monkeypatch.setattr(pose_matching, "_resolve_runtime_device", lambda _: "cpu")
    monkeypatch.setattr(pose_matching, "_resolve_backbone", lambda *_: object())
    monkeypatch.setattr(pose_matching, "_load_subsets", lambda _: [subset])
    monkeypatch.setattr(pose_matching, "_selected", lambda values, *_: values)
    monkeypatch.setattr(
        pose_matching,
        "_extract_superpoint_group",
        lambda **kwargs: calls.append(kwargs),
    )

    assert pose_matching.main() == 0
    assert calls[0]["cache_root"] == keypoint_root.resolve()


def test_pose_superpoint_cache_omits_unused_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subset = pose_matching.Subset(
        "ScanNet",
        "scannet_resized",
        "subset",
        tmp_path / "images",
        tmp_path / "pairs.csv",
        False,
    )
    observed: list[bool] = []

    class FakeCache:
        def __init__(self, **_: object) -> None:
            pass

        def load_or_extract(self, _: str, *, include_descriptors: bool) -> None:
            observed.append(include_descriptors)

    monkeypatch.setattr(
        "dino_m2m.superpoint.ExternalLightGlueSuperPoint",
        lambda *_: object(),
    )
    monkeypatch.setattr("dino_m2m.superpoint.CacheBackedSuperPoint", FakeCache)
    monkeypatch.setattr(pose_matching, "_unique_images", lambda _: ["image.jpg"])

    pose_matching._extract_superpoint_group(
        group="scannet",
        subsets=[subset],
        cache_root=tmp_path / "keypoints",
        device="cpu",
        max_num_keypoints=2048,
    )
    assert observed == [False]


def _write_result(path: Path, error: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "pair_idx": [1, 2],
            "status": ["success", "success"],
            "rotation_error_deg": [error, error + 0.5],
            "translation_error_deg": [error + 0.25, error],
            "running_time_s": [0.2 + error * 0.01, 0.3 + error * 0.01],
            "score_us_per_eval": [4.0 + error * 0.1, 5.0 + error * 0.1],
        }
    ).to_csv(path, index=False)


def _write_association_manifest(
    directory: Path,
    *,
    pair_count: int,
    progressive_max_k: int,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "association_manifest.json").write_text(
        json.dumps(
            {
                "pair_count": pair_count,
                "model_name": "dinov3_vitl16",
                "layer": 19,
                "debias_rank": 200,
                "progressive_max_k": progressive_max_k,
                "association_upperbound": 2048,
                "weights_id": gms.PAPER_WEIGHTS_IDS["dinov3_vitl16"],
            }
        ),
        encoding="utf-8",
    )
    if "GMSm2m" in directory.name:
        raw_directory = directory.parent / gms.RAW_M2M_VARIANT
        _write_association_manifest(
            raw_directory,
            pair_count=pair_count,
            progressive_max_k=5,
        )
        (directory / gms.GMS_CONFIG_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "filter": "gms_m2m",
                    "threshold_factor": 4.0,
                    "grid_size": 20,
                    "auto_mask": True,
                    "with_scale": True,
                    "with_rotation": True,
                    "source_directory": str(raw_directory),
                }
            ),
            encoding="utf-8",
        )


def test_pose_result_resume_rejects_pair_id_only_rows(tmp_path: Path) -> None:
    result = tmp_path / "truncated.csv"
    pd.DataFrame({"pair_idx": [1, 2]}).to_csv(result, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        gms._result_pair_ids(result, 2)


def test_pose_result_validator_accepts_structurally_complete_skip(
    tmp_path: Path,
) -> None:
    result = tmp_path / "skipped.csv"
    pd.DataFrame(
        {
            "pair_idx": [1],
            "status": ["skipped"],
            "rotation_error_deg": [float("nan")],
            "translation_error_deg": [float("nan")],
            "running_time_s": [0.001],
            "score_us_per_eval": [float("nan")],
        }
    ).to_csv(result, index=False)
    assert gms._result_pair_ids(result, 1) == [1]


def test_five_seed_summary_writes_plot_ready_csvs(tmp_path: Path) -> None:
    results = tmp_path / "results"
    subsets = []
    pipelines = gms._pipeline_specs(("mnn:CM", "gms:4:20:HCM_MC"))
    for dataset_index, dataset in enumerate(gms.CANONICAL_DATASETS):
        root_name = f"root_{dataset_index}"
        subset_name = "subset"
        leaf = results / root_name / subset_name
        leaf.mkdir(parents=True)
        pd.DataFrame({"pair_idx": [1, 2]}).to_csv(
            leaf / "pose_intrinsics.csv", index=False
        )
        subsets.append(
            {
                "dataset_label": dataset,
                "root_name": root_name,
                "subset_name": subset_name,
                "pair_count": 2,
            }
        )
        subset = gms.Subset(dataset, root_name, subset_name, 2)
        method_root = gms._method_root(
            results, subset, 19, 200, "combined_interpolate_v3"
        )
        for pipeline in pipelines:
            _write_association_manifest(
                method_root / pipeline.variant_dir,
                pair_count=2,
                progressive_max_k=1 if pipeline.variant_label == "mnn" else 5,
            )
            for seed in range(5):
                result = method_root / pipeline.variant_dir / gms._result_filename(
                    pipeline.mode, 100000, seed
                )
                error = 4.5 + seed * 0.05
                if pipeline.label == "gms_t4_g20/HCM_MC":
                    error -= 1.0
                _write_result(result, error)

    manifest = results / "experiment_inputs.json"
    manifest.write_text(json.dumps({"subsets": subsets}), encoding="utf-8")
    output = tmp_path / "summary" / "pose_auc.json"
    args = argparse.Namespace(
        input_manifest=manifest,
        datasets=list(gms.CANONICAL_DATASETS),
        results_root=results,
        layer=19,
        debias_rank=200,
        method_prefix="combined_interpolate_v3",
        layer_selection=None,
        command="summarize",
        model_name="dinov3_vitl16",
        pipelines=["mnn:CM", "gms:4:20:HCM_MC"],
        seeds=(0, 1, 2, 3, 4),
        max_iterations=100000,
        output=output,
        per_seed_csv=None,
        summary_csv=None,
    )
    gms.run_summarize(args)

    per_seed = pd.read_csv(output.parent / "auc_per_seed.csv")
    summary = pd.read_csv(output.parent / "auc_five_seed_summary.csv")
    assert len(per_seed) == 2 * 7 * 3 * 5
    assert len(summary) == 2 * 7 * 3
    assert set(summary["variant"]) == {"mnn_cm", "full_hcm_mc"}
    assert set(summary["seeds"].astype(str)) == {"0,1,2,3,4"}
    runtime = pd.read_csv(output.parent / "runtime_five_seed_summary.csv")
    assert len(runtime) == 2 * 6 * 2
    assert set(runtime["metric"]) == {"running_time_s", "score_us_per_eval"}
    assert not (output.parent / "table1_runtime.csv").exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["seeds"] == [0, 1, 2, 3, 4]


def test_q2_summary_writes_five_seed_mean_table1(tmp_path: Path) -> None:
    results = tmp_path / "results"
    root_name = "root_scannet"
    subset_name = "subset"
    leaf = results / root_name / subset_name
    leaf.mkdir(parents=True)
    pd.DataFrame({"pair_idx": [1, 2]}).to_csv(
        leaf / "pose_intrinsics.csv", index=False
    )
    manifest = results / "experiment_inputs.json"
    manifest.write_text(
        json.dumps(
            {
                "subsets": [
                    {
                        "dataset_label": "ScanNet",
                        "root_name": root_name,
                        "subset_name": subset_name,
                        "pair_count": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    pipelines = gms._pipeline_specs(
        (
            "gms:4:20:CM",
            "gms:4:20:HCM",
            "gms:4:20:MCM",
            "gms:4:20:HCM_MC",
        )
    )
    subset = gms.Subset("ScanNet", root_name, subset_name, 2)
    method_root = gms._method_root(
        results, subset, 19, 200, "combined_interpolate_v3"
    )
    error_by_mode = {"CM": 1.0, "HCM": 2.0, "MCM": 3.0, "HCM_MC": 4.0}
    for pipeline in pipelines:
        _write_association_manifest(
            method_root / pipeline.variant_dir,
            pair_count=2,
            progressive_max_k=5,
        )
        for seed in range(5):
            result = method_root / pipeline.variant_dir / gms._result_filename(
                pipeline.mode, 100000, seed
            )
            _write_result(result, error_by_mode[pipeline.mode] + seed * 0.1)

    output = tmp_path / "summary" / "pose_auc.json"
    args = argparse.Namespace(
        input_manifest=manifest,
        datasets=["ScanNet"],
        results_root=results,
        layer=19,
        debias_rank=200,
        method_prefix="combined_interpolate_v3",
        layer_selection=None,
        command="summarize",
        model_name="dinov3_vitl16",
        pipelines=[
            "gms:4:20:CM",
            "gms:4:20:HCM",
            "gms:4:20:MCM",
            "gms:4:20:HCM_MC",
        ],
        seeds=(0, 1, 2, 3, 4),
        max_iterations=100000,
        output=output,
        per_seed_csv=None,
        summary_csv=None,
    )
    gms.run_summarize(args)

    table = pd.read_csv(output.parent / "table1_runtime.csv")
    assert table["dataset"].tolist() == ["ScanNet"]
    assert table["seeds"].astype(str).tolist() == ["0,1,2,3,4"]
    assert table.iloc[0][
        [
            "hcm_scorer_us_per_eval",
            "mcm_scorer_us_per_eval",
            "cm_runner_ms_per_pair",
            "hcm_runner_ms_per_pair",
            "mcm_runner_ms_per_pair",
            "hcm_mc_runner_ms_per_pair",
        ]
    ].astype(float).tolist() == pytest.approx([4.72, 4.82, 262, 272, 282, 292])
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["outputs"]["table1_runtime_csv"] == str(
        output.parent / "table1_runtime.csv"
    )


def test_pose_driver_defaults_to_five_seeds() -> None:
    parser = gms.build_parser()
    estimate = parser.parse_args(["estimate", "--pipelines", "mnn:CM"])
    summarize = parser.parse_args(
        [
            "summarize",
            "--pipelines",
            "mnn:CM",
            "--output",
            "summary.json",
        ]
    )
    assert estimate.seeds == (0, 1, 2, 3, 4)
    assert summarize.seeds == (0, 1, 2, 3, 4)


def test_summary_rejects_wrong_association_contract(tmp_path: Path) -> None:
    results = tmp_path / "results"
    manifest = results / "experiment_inputs.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "subsets": [
                    {
                        "dataset_label": "ScanNet",
                        "root_name": "root",
                        "subset_name": "subset",
                        "pair_count": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    subset = gms.Subset("ScanNet", "root", "subset", 2)
    directory = (
        gms._method_root(results, subset, 19, 200, "combined_interpolate_v3")
        / gms.MNN_VARIANT
    )
    _write_association_manifest(
        directory,
        pair_count=2,
        progressive_max_k=1,
    )
    payload = json.loads(
        (directory / "association_manifest.json").read_text(encoding="utf-8")
    )
    payload["association_upperbound"] = 0
    (directory / "association_manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    args = argparse.Namespace(
        command="summarize",
        input_manifest=manifest,
        datasets=["ScanNet"],
        results_root=results,
        model_name="dinov3_vitl16",
        layer=19,
        debias_rank=200,
        method_prefix="combined_interpolate_v3",
        layer_selection=None,
        pipelines=["mnn:CM"],
        seeds=(0, 1, 2, 3, 4),
        max_iterations=100000,
        output=tmp_path / "summary.json",
        per_seed_csv=None,
        summary_csv=None,
    )
    with pytest.raises(ValueError, match="association_upperbound"):
        gms.run_summarize(args)


def test_summary_rejects_wrong_gms_filter_contract(tmp_path: Path) -> None:
    results = tmp_path / "results"
    manifest = results / "experiment_inputs.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "subsets": [
                    {
                        "dataset_label": "ScanNet",
                        "root_name": "root",
                        "subset_name": "subset",
                        "pair_count": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    subset = gms.Subset("ScanNet", "root", "subset", 2)
    pipeline = gms._parse_pipeline("gms:4:20:HCM_MC")
    directory = (
        gms._method_root(results, subset, 19, 200, "combined_interpolate_v3")
        / pipeline.variant_dir
    )
    _write_association_manifest(directory, pair_count=2, progressive_max_k=5)
    filter_path = directory / gms.GMS_CONFIG_FILENAME
    filter_manifest = json.loads(filter_path.read_text(encoding="utf-8"))
    filter_manifest["with_rotation"] = False
    filter_path.write_text(json.dumps(filter_manifest), encoding="utf-8")
    args = argparse.Namespace(
        command="summarize",
        input_manifest=manifest,
        datasets=["ScanNet"],
        results_root=results,
        model_name="dinov3_vitl16",
        layer=19,
        debias_rank=200,
        method_prefix="combined_interpolate_v3",
        layer_selection=None,
        pipelines=["gms:4:20:HCM_MC"],
        seeds=(0, 1, 2, 3, 4),
        max_iterations=100000,
        output=tmp_path / "summary.json",
        per_seed_csv=None,
        summary_csv=None,
    )
    with pytest.raises(ValueError, match="main-paper GMS settings mismatch"):
        gms.run_summarize(args)


@pytest.mark.parametrize(
    ("corruption", "error_pattern"),
    (
        ("missing_pair", "is incomplete"),
        ("missing_scorer", "invalid score_us_per_eval"),
    ),
)
def test_summary_rejects_incomplete_results(
    tmp_path: Path,
    corruption: str,
    error_pattern: str,
) -> None:
    results = tmp_path / "results"
    root_name = "root"
    subset_name = "subset"
    leaf = results / root_name / subset_name
    leaf.mkdir(parents=True)
    pd.DataFrame({"pair_idx": [1, 2]}).to_csv(
        leaf / "pose_intrinsics.csv", index=False
    )
    manifest = results / "experiment_inputs.json"
    manifest.write_text(
        json.dumps(
            {
                "subsets": [
                    {
                        "dataset_label": "ScanNet",
                        "root_name": root_name,
                        "subset_name": subset_name,
                        "pair_count": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    subset = gms.Subset("ScanNet", root_name, subset_name, 2)
    directory = (
        gms._method_root(results, subset, 19, 200, "combined_interpolate_v3")
        / gms.MNN_VARIANT
    )
    _write_association_manifest(directory, pair_count=2, progressive_max_k=1)
    for seed in range(5):
        result = directory / gms._result_filename("CM", 100000, seed)
        _write_result(result, 1.0)
    corrupt = directory / gms._result_filename("CM", 100000, 0)
    frame = pd.read_csv(corrupt)
    if corruption == "missing_pair":
        frame = frame.iloc[:1]
    else:
        frame.loc[0, "score_us_per_eval"] = float("nan")
    frame.to_csv(corrupt, index=False)
    args = argparse.Namespace(
        command="summarize",
        input_manifest=manifest,
        datasets=["ScanNet"],
        results_root=results,
        model_name="dinov3_vitl16",
        layer=19,
        debias_rank=200,
        method_prefix="combined_interpolate_v3",
        layer_selection=None,
        pipelines=["mnn:CM"],
        seeds=(0, 1, 2, 3, 4),
        max_iterations=100000,
        output=tmp_path / "summary.json",
        per_seed_csv=None,
        summary_csv=None,
    )
    with pytest.raises(ValueError, match=error_pattern):
        gms.run_summarize(args)
