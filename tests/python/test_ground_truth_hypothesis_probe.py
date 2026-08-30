from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.estimation.ground_truth_hypothesis_probe import (
    ProbeIntegrityError,
    ProbeTask,
    run_task,
    verify_replay,
)


NORMAL_COLUMNS = (
    "pair_idx",
    "status",
    "iterations",
    "refinements",
    "q_w",
    "running_time_s",
)
PROBE_COLUMNS = (
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


def _write_csv(path: Path, columns: tuple[str, ...], row: tuple[object, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerow(row)


class GroundTruthHypothesisProbeTests(unittest.TestCase):
    def test_replay_allows_only_timing_differences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.csv"
            replay = root / "replay.csv"
            sidecar = root / "probe.csv"
            _write_csv(original, NORMAL_COLUMNS, (7, "success", 100000, 12, 1.0, 3.0))
            _write_csv(replay, NORMAL_COLUMNS, (7, "success", 100000, 12, 1.0, 4.0))
            _write_csv(
                sidecar,
                PROBE_COLUMNS,
                (7, "success", "", 100, 100, 1, 0.1, 0.2, 9, 1),
            )
            task = ProbeTask(
                dataset="ScanNet",
                root_name="scannet_resized",
                subset_name="test_1500",
                k=1,
                seed=0,
                original_result_csv=str(original),
                nested_manifest=str(root / "manifest.json"),
            )
            receipt = verify_replay(task, replay, sidecar)
            self.assertEqual(receipt["verification"], "PASS")
            self.assertEqual(receipt["rows"], 1)
            self.assertIn(
                "running_time_s",
                receipt["timing_columns_excluded_from_identity_check"],
            )

            _write_csv(replay, NORMAL_COLUMNS, (7, "success", 100000, 12, 0.9, 4.0))
            with self.assertRaisesRegex(ProbeIntegrityError, "Probe changed"):
                verify_replay(task, replay, sidecar)

    def test_pool_admission_is_strict_on_ties(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.csv"
            replay = root / "replay.csv"
            sidecar = root / "probe.csv"
            row = (3, "success", 100000, 5, 1.0, 1.0)
            _write_csv(original, NORMAL_COLUMNS, row)
            _write_csv(replay, NORMAL_COLUMNS, row)
            _write_csv(
                sidecar,
                PROBE_COLUMNS,
                (3, "success", "", 100, 100, 1, 0.2, 0.2, 4, 0),
            )
            task = ProbeTask(
                dataset="MegaDepth",
                root_name="megadepth_resized",
                subset_name="test_1500",
                k=5,
                seed=4,
                original_result_csv=str(original),
                nested_manifest=str(root / "manifest.json"),
            )
            self.assertEqual(verify_replay(task, replay, sidecar)["verification"], "PASS")

    def test_verified_receipt_is_a_read_only_restart_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.csv"
            _write_csv(
                original,
                NORMAL_COLUMNS,
                (9, "success", 100000, 8, 1.0, 2.0),
            )
            task = ProbeTask(
                dataset="ScanNet",
                root_name="scannet_resized",
                subset_name="test_1500",
                k=1,
                seed=0,
                original_result_csv=str(original),
                nested_manifest=str(root / "manifest.json"),
            )
            task_dir = (
                root
                / "tasks"
                / task.root_name
                / task.subset_name
                / "k1"
                / "seed0"
            )
            task_dir.mkdir(parents=True)
            primary = task_dir / "HCM_MC_pose_replay_proposal_k_1_q_ub_0.30.csv"
            sidecar = (
                task_dir
                / "HCM_MC_pose_replay_proposal_k_1_q_ub_0.30_gt_hypothesis_probe.csv"
            )
            _write_csv(
                primary,
                NORMAL_COLUMNS,
                (9, "success", 100000, 8, 1.0, 3.0),
            )
            _write_csv(
                sidecar,
                PROBE_COLUMNS,
                (9, "success", "", 100, 100, 1, 0.1, 0.2, 6, 1),
            )
            receipt = verify_replay(task, primary, sidecar)
            (task_dir / "receipt.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )

            result = run_task(task, Path("/runner-must-not-be-called"), root)
            self.assertEqual(result, {"task": task.key, "rows": 1})


if __name__ == "__main__":
    unittest.main()
