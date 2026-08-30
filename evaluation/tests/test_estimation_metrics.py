from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.estimation.align import align_estimation_errors, build_parser
from evaluation.estimation.metrics import (
    DATASETS,
    count_matching_csvs,
    dataset_registry,
    error_auc,
    expected_pose_pair_indices,
    pose_error_vector,
    pose_error_vector_for_indices,
)
from evaluation.json_utils import strict_json_dumps


class EstimationMetricTests(unittest.TestCase):
    def test_navi_wild_uses_canonical_result_root(self) -> None:
        self.assertEqual(
            {subset.root_name for subset in DATASETS["NAVI-Wild"]},
            {"NAVI_wild"},
        )
        self.assertTrue(
            all(subset.require_pose_manifest for subset in DATASETS["NAVI-Wild"])
        )
        self.assertEqual(
            {subset.root_name for subset in DATASETS["NAVI-Multi"]},
            {"NAVI_resized"},
        )
        self.assertEqual(set(dataset_registry()), set(DATASETS))

    def test_matching_count_ignores_pair_lists_and_legacy_association_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "matching_001.csv").touch()
            (root / "matching_002.csv").touch()
            (root / "pairs_003.csv").touch()
            (root / "pairs_wildset_0-40.csv").touch()
            self.assertEqual(count_matching_csvs(root), 2)

    def test_auc_matches_experiment_integration_rule(self) -> None:
        auc = error_auc(np.array([1.0]), thresholds=(5.0,))
        self.assertAlmostEqual(auc["auc@5"], 90.0)

    def test_missing_and_nan_pose_rows_are_failures(self) -> None:
        frame = pd.DataFrame(
            {
                "rotation_error_deg": [1.0, np.nan],
                "translation_error_deg": [2.0, 3.0],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            frame.to_csv(path, index=False)
            errors = pose_error_vector(path, total_pairs=4)
        np.testing.assert_allclose(errors, [2.0, 180.0, 180.0, 180.0])

    def test_pose_errors_align_to_fixed_pair_indices(self) -> None:
        frame = pd.DataFrame(
            {
                "pair_idx": [12, 10],
                "rotation_error_deg": [2.0, 1.0],
                "translation_error_deg": [3.0, 4.0],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            frame.to_csv(path, index=False)
            errors = pose_error_vector_for_indices(path, [10, 11, 12])
        np.testing.assert_allclose(errors, [4.0, 180.0, 3.0])

    def test_pose_manifest_fixes_denominator_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pose = root / "pose_intrinsics.csv"
            pd.DataFrame({"pair_idx": [7, 9]}).to_csv(pose, index=False)
            digest = hashlib.sha256(pose.read_bytes()).hexdigest()
            (root / "pose_intrinsics_manifest.json").write_text(
                json.dumps(
                    {
                        "pair_count": 2,
                        "pose_intrinsics_sha256": digest,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(expected_pose_pair_indices(pose), (7, 9))
            pose.write_text("pair_idx\n7\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sha256"):
                expected_pose_pair_indices(pose)

    def test_current_split_requires_pose_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pose = Path(directory) / "pose_intrinsics.csv"
            pd.DataFrame({"pair_idx": [1]}).to_csv(pose, index=False)
            with self.assertRaisesRegex(FileNotFoundError, "manifest is required"):
                expected_pose_pair_indices(pose, require_manifest=True)
            self.assertEqual(expected_pose_pair_indices(pose), (1,))

    def test_alignment_uses_pose_as_stored_and_removes_legacy_mode(self) -> None:
        root_half = np.sqrt(0.5)
        results = pd.DataFrame(
            {
                "pair_idx": [1],
                "status": ["success"],
                "q_w": [root_half],
                "q_x": [0.0],
                "q_y": [0.0],
                "q_z": [root_half],
                "t_x": [1.0],
                "t_y": [0.0],
                "t_z": [0.0],
                "gt_mode_used": ["inverse"],
            }
        )
        poses = pd.DataFrame(
            {
                "pair_idx": [1],
                "qw": [root_half],
                "qx": [0.0],
                "qy": [0.0],
                "qz": [root_half],
                "tx": [1.0],
                "ty": [0.0],
                "tz": [0.0],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            pose_path = Path(directory) / "pose_intrinsics.csv"
            poses.to_csv(pose_path, index=False)
            aligned = align_estimation_errors(results, pose_path)
        self.assertAlmostEqual(float(aligned.loc[0, "rotation_error_deg"]), 0.0)
        self.assertAlmostEqual(float(aligned.loc[0, "translation_error_deg"]), 0.0)
        self.assertNotIn("gt_mode_used", aligned.columns)

    def test_alignment_cli_requires_pose_csv(self) -> None:
        parser = build_parser()
        pose_action = next(
            action
            for action in parser._actions
            if "--pose-intrinsics" in action.option_strings
        )
        self.assertTrue(pose_action.required)
        self.assertFalse(
            any("--mode" in action.option_strings for action in parser._actions)
        )

    def test_json_serialization_replaces_nonfinite_values(self) -> None:
        rendered = strict_json_dumps(
            {"nan": float("nan"), "positive_inf": float("inf")},
            sort_keys=True,
        )
        self.assertEqual(
            json.loads(rendered), {"nan": None, "positive_inf": None}
        )
        self.assertNotIn("NaN", rendered)
        self.assertNotIn("Infinity", rendered)


if __name__ == "__main__":
    unittest.main()
