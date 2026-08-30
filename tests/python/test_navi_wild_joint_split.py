"""Repository-level contract checks for the versioned NAVI-Wild split."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SPLIT_ROOT = REPOSITORY_ROOT / "data" / "splits" / "navi"
BINS = {
    "0-40": (0.0, 40.0),
    "40-80": (40.0, 80.0),
    "80-120": (80.0, 120.0),
}
FAMILIES = ("estimation", "correspondence")
PAIR_HEADER = [
    "image_1",
    "image_2",
    "angular_distance_degrees",
    "image_1_height",
    "image_1_width",
    "image_2_height",
    "image_2_width",
]
PARTITION_HEADER = [
    "object_name",
    "image_path",
    "split",
    "navi_official_split",
    "image_height",
    "image_width",
    "camera_model",
    "occluded",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(family: str, angular_bin: str) -> list[dict[str, str]]:
    path = SPLIT_ROOT / family / f"pairs_wildset_{angular_bin}.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _load_generator() -> types.ModuleType:
    name = "navi_wild_joint_generator_test"
    if name in sys.modules:
        return sys.modules[name]
    path = SPLIT_ROOT / "generate.py"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def _metadata_sha256(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: value["image_path"]):
        digest.update(
            (
                f'{row["image_path"]}\t{row["image_height"]}\t'
                f'{row["image_width"]}\t{row["navi_official_split"]}\t'
                f'{row["camera_model"]}\t{row["occluded"]}\n'
            ).encode("utf-8")
        )
    return digest.hexdigest()


class NaviWildJointSplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads((SPLIT_ROOT / "manifest.json").read_text(encoding="utf-8"))
        with (SPLIT_ROOT / "image_partition.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            cls.partition_header = reader.fieldnames
            cls.partition_rows = list(reader)
        cls.owner = {row["image_path"]: row["split"] for row in cls.partition_rows}
        cls.object_for_image = {
            row["image_path"]: row["object_name"] for row in cls.partition_rows
        }
        cls.metadata_for_image = {
            row["image_path"]: row for row in cls.partition_rows
        }

    def test_manifest_and_checksums(self) -> None:
        self.assertEqual(self.manifest["split_version"], "navi-wild-joint")
        self.assertFalse(
            self.manifest["algorithm"]["selection_uses_descriptor_or_evaluation_results"]
        )
        self.assertTrue(self.manifest["audit"]["global_image_disjoint"])
        self.assertEqual(self.manifest["counts"]["objects"], 35)
        self.assertEqual(self.manifest["counts"]["source_images"], 2180)
        self.assertEqual(
            self.manifest["source_sha256"],
            "0da6831a3a0ac98cb6df5919548021253e56ab0f01a9ddec89ad5c56b4dc794a",
        )
        self.assertEqual(
            self.manifest["generator_sha256"],
            _sha256(SPLIT_ROOT / "generate.py"),
        )
        self.assertEqual(self.partition_header, PARTITION_HEADER)
        self.assertEqual(
            self.manifest["image_metadata_sha256"],
            _metadata_sha256(self.partition_rows),
        )
        self.assertEqual(
            self.manifest["algorithm"]["joint_selection_soft_objectives"],
            [
                "same_vs_cross_camera_pair_balance",
                "unordered_camera_pair_histogram_balance",
            ],
        )
        self.assertEqual(len(self.partition_rows), 2180)
        self.assertEqual(len(self.owner), 2180)
        self.assertEqual(Counter(self.owner.values()), Counter({family: 1090 for family in FAMILIES}))
        per_object: dict[str, Counter[str]] = {}
        for row in self.partition_rows:
            self.assertIn(row["split"], FAMILIES)
            self.assertEqual(row["object_name"], row["image_path"].split("/", 1)[0])
            per_object.setdefault(row["object_name"], Counter())[row["split"]] += 1
        self.assertEqual(len(per_object), 35)
        for counts in per_object.values():
            self.assertLessEqual(abs(counts["estimation"] - counts["correspondence"]), 1)
        self.assertEqual({row["occluded"] for row in self.partition_rows}, {"0"})

        for relative, record in self.manifest["files"].items():
            path = SPLIT_ROOT / relative
            self.assertEqual(path.stat().st_size, record["bytes"])
            self.assertEqual(_sha256(path), record["sha256"])

        checksum_rows = {}
        for raw in (SPLIT_ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            digest, relative = raw.split("  ", 1)
            checksum_rows[relative] = digest
        self.assertEqual(set(checksum_rows), {*self.manifest["files"], "manifest.json"})
        for relative, digest in checksum_rows.items():
            self.assertEqual(_sha256(SPLIT_ROOT / relative), digest)

    def test_wild_pair_lists_use_the_canonical_family_directories(self) -> None:
        self.assertEqual(
            {path.name for path in SPLIT_ROOT.glob("*.py")},
            {"generate.py"},
        )
        for angular_bin in BINS:
            for family in FAMILIES:
                self.assertTrue(
                    (SPLIT_ROOT / family / f"pairs_wildset_{angular_bin}.csv").is_file()
                )

    def test_pair_counts_ownership_and_angle_balance(self) -> None:
        quota_maps: dict[str, dict[str, Counter[str]]] = {
            family: {} for family in FAMILIES
        }
        five_degree_histograms: dict[tuple[str, str], Counter[tuple[str, int]]] = {}
        all_selected = {family: set() for family in FAMILIES}
        seen_edges = {family: set() for family in FAMILIES}

        for family in FAMILIES:
            cumulative_degree: Counter[str] = Counter()
            for angular_bin, (lower, upper) in BINS.items():
                pair_path = SPLIT_ROOT / family / f"pairs_wildset_{angular_bin}.csv"
                with pair_path.open(newline="", encoding="utf-8") as stream:
                    reader = csv.DictReader(stream)
                    self.assertEqual(reader.fieldnames, PAIR_HEADER)
                    rows = list(reader)
                self.assertEqual(len(rows), 500)
                object_counts: Counter[str] = Counter()
                histogram: Counter[tuple[str, int]] = Counter()
                endpoint_degree: Counter[str] = Counter()
                reversed_from_canonical_source = 0
                for row in rows:
                    left, right = row["image_1"], row["image_2"]
                    self.assertNotEqual(left, right)
                    self.assertEqual(self.owner[left], family)
                    self.assertEqual(self.owner[right], family)
                    object_name = left.split("/", 1)[0]
                    self.assertEqual(right.split("/", 1)[0], object_name)
                    self.assertEqual(self.object_for_image[left], object_name)
                    angle = float(row["angular_distance_degrees"])
                    self.assertLessEqual(lower, angle)
                    self.assertLess(angle, upper)
                    object_counts[object_name] += 1
                    histogram[(object_name, int((angle - lower) // 5))] += 1
                    undirected = tuple(sorted((left, right)))
                    self.assertNotIn(undirected, seen_edges[family])
                    seen_edges[family].add(undirected)
                    all_selected[family].update((left, right))
                    endpoint_degree.update((left, right))
                    cumulative_degree.update((left, right))
                    reversed_from_canonical_source += int(left > right)
                    for prefix, image in (("image_1", left), ("image_2", right)):
                        metadata = self.metadata_for_image[image]
                        self.assertEqual(
                            int(row[f"{prefix}_height"]),
                            int(metadata["image_height"]),
                        )
                        self.assertEqual(
                            int(row[f"{prefix}_width"]),
                            int(metadata["image_width"]),
                        )
                self.assertEqual(len(object_counts), 35)
                self.assertEqual(Counter(object_counts.values()), Counter({14: 25, 15: 10}))
                self.assertLessEqual(max(endpoint_degree.values()), 4)
                self.assertEqual(reversed_from_canonical_source, 250)
                quota_maps[family][angular_bin] = object_counts
                five_degree_histograms[(family, angular_bin)] = histogram
            self.assertLessEqual(max(cumulative_degree.values()), 6)

        self.assertFalse(all_selected["estimation"] & all_selected["correspondence"])
        for angular_bin in BINS:
            self.assertEqual(
                quota_maps["estimation"][angular_bin],
                quota_maps["correspondence"][angular_bin],
            )
            self.assertEqual(
                five_degree_histograms[("estimation", angular_bin)],
                five_degree_histograms[("correspondence", angular_bin)],
            )
        totals = Counter()
        for angular_bin in BINS:
            totals.update(quota_maps["estimation"][angular_bin])
        self.assertEqual(Counter(totals.values()), Counter({43: 30, 42: 5}))

    def test_camera_pair_balance_audit(self) -> None:
        audit = self.manifest["audit"]["camera_pair_balance"]
        self.assertEqual(
            audit["definition"],
            "unordered camera-model pair, compared per object and angular bin",
        )
        total_same_difference = 0
        total_pair_l1 = 0
        maximum_same_difference = 0
        maximum_pair_l1 = 0
        for angular_bin in BINS:
            histograms: dict[str, dict[str, Counter[tuple[str, str]]]] = {
                family: {} for family in FAMILIES
            }
            for family in FAMILIES:
                for row in _rows(family, angular_bin):
                    object_name = row["image_1"].split("/", 1)[0]
                    camera_pair = tuple(
                        sorted(
                            (
                                self.metadata_for_image[row["image_1"]]["camera_model"],
                                self.metadata_for_image[row["image_2"]]["camera_model"],
                            )
                        )
                    )
                    histograms[family].setdefault(object_name, Counter())[camera_pair] += 1

            same_totals = {family: 0 for family in FAMILIES}
            same_differences: dict[str, int] = {}
            pair_l1_values: dict[str, int] = {}
            for object_name in sorted(histograms["estimation"]):
                same_counts = {}
                for family in FAMILIES:
                    same_counts[family] = sum(
                        count
                        for (first, second), count in histograms[family][object_name].items()
                        if first == second
                    )
                    same_totals[family] += same_counts[family]
                same_differences[object_name] = abs(
                    same_counts["estimation"] - same_counts["correspondence"]
                )
                keys = (
                    histograms["estimation"][object_name].keys()
                    | histograms["correspondence"][object_name].keys()
                )
                pair_l1_values[object_name] = sum(
                    abs(
                        histograms["estimation"][object_name][key]
                        - histograms["correspondence"][object_name][key]
                    )
                    for key in keys
                )

            expected = audit["per_bin"][angular_bin]
            self.assertEqual(
                expected["estimation_same_camera_pairs"],
                same_totals["estimation"],
            )
            self.assertEqual(
                expected["correspondence_same_camera_pairs"],
                same_totals["correspondence"],
            )
            self.assertEqual(
                expected["sum_object_same_camera_absolute_difference"],
                sum(same_differences.values()),
            )
            self.assertEqual(
                expected["max_object_same_camera_absolute_difference"],
                max(same_differences.values()),
            )
            self.assertEqual(
                expected["sum_object_unordered_camera_pair_l1"],
                sum(pair_l1_values.values()),
            )
            self.assertEqual(
                expected["max_object_unordered_camera_pair_l1"],
                max(pair_l1_values.values()),
            )
            total_same_difference += sum(same_differences.values())
            total_pair_l1 += sum(pair_l1_values.values())
            maximum_same_difference = max(
                maximum_same_difference, max(same_differences.values())
            )
            maximum_pair_l1 = max(maximum_pair_l1, max(pair_l1_values.values()))

        self.assertEqual(
            audit["sum_object_bin_same_camera_absolute_difference"],
            total_same_difference,
        )
        self.assertEqual(
            audit["sum_object_bin_unordered_camera_pair_l1"], total_pair_l1
        )
        self.assertEqual(
            audit["max_object_bin_same_camera_absolute_difference"],
            maximum_same_difference,
        )
        self.assertEqual(
            audit["max_object_bin_unordered_camera_pair_l1"], maximum_pair_l1
        )
        self.assertLessEqual(total_same_difference, 10)
        self.assertLessEqual(total_pair_l1, 520)
        self.assertLessEqual(maximum_same_difference, 5)
        self.assertLessEqual(maximum_pair_l1, 14)

    def test_partition_capacity_audit(self) -> None:
        with (SPLIT_ROOT / "partition_capacity.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 35 * len(BINS))
        for row in rows:
            quota = int(row["pair_quota"])
            self.assertIn(quota, (14, 15))
            self.assertGreaterEqual(int(row["estimation_edge_capacity"]), quota + 4)
            self.assertGreaterEqual(int(row["correspondence_edge_capacity"]), quota + 4)
            self.assertEqual(
                sum(int(value) for value in row["five_degree_pair_quotas"].split(";")),
                quota,
            )

    def test_selected_edges_match_source_when_available(self) -> None:
        source_path = REPOSITORY_ROOT / "data" / "navi" / "pairs-wild_set.txt"
        if not source_path.is_file():
            self.skipTest("Raw NAVI pair graph is not distributed")
        self.assertEqual(_sha256(source_path), self.manifest["source_sha256"])
        source_oriented = {tuple(raw.split()) for raw in source_path.read_text(encoding="utf-8").splitlines()}
        source_undirected = {
            (tuple(sorted((left, right))), angle)
            for left, right, angle in source_oriented
        }
        for family in FAMILIES:
            for angular_bin in BINS:
                reversed_rows = 0
                for row in _rows(family, angular_bin):
                    self.assertIn(
                        (
                            tuple(sorted((row["image_1"], row["image_2"]))),
                            row["angular_distance_degrees"],
                        ),
                        source_undirected,
                    )
                    reversed_rows += int(
                        (
                            row["image_1"],
                            row["image_2"],
                            row["angular_distance_degrees"],
                        )
                        not in source_oriented
                    )
                self.assertEqual(reversed_rows, 250)

    def test_binary_writer_is_lf_and_preserves_existing_file_on_replace_error(self) -> None:
        generator = _load_generator()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "value.txt"
            generator._write_text(path, "first\nsecond\n", False)
            self.assertEqual(path.read_bytes(), b"first\nsecond\n")
            with self.assertRaises(FileExistsError):
                generator._write_text(path, "unexpected\n", False)
            with patch.object(generator.os, "replace", side_effect=OSError("injected")):
                with self.assertRaisesRegex(OSError, "injected"):
                    generator._write_text(path, "replacement\n", True)
            self.assertEqual(path.read_bytes(), b"first\nsecond\n")
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_full_regeneration_when_navi_is_available(self) -> None:
        source_path = REPOSITORY_ROOT / "data" / "navi" / "pairs-wild_set.txt"
        image_root = REPOSITORY_ROOT / "data" / "navi"
        if os.environ.get("NAVI_FULL_REGEN_TEST") != "1":
            self.skipTest("Set NAVI_FULL_REGEN_TEST=1 for the 25-second full check")
        if not source_path.is_file():
            self.skipTest("Raw NAVI pair graph is not distributed")
        generator = _load_generator()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "split"
            generator.generate(source_path, image_root, output)
            self.assertEqual(
                (output / "manifest.json").read_bytes(),
                (SPLIT_ROOT / "manifest.json").read_bytes(),
            )
            self.assertEqual(
                (output / "SHA256SUMS").read_bytes(),
                (SPLIT_ROOT / "SHA256SUMS").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
