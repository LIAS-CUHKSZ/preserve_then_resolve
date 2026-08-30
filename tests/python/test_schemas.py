from __future__ import annotations

import csv
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from dino_m2m.matching import write_association_csv
from dino_m2m.schemas import (
    ASSOCIATION_COLUMNS,
    load_dino_map,
    load_superpoint_cache,
    save_dino_map,
    save_superpoint_cache,
    validate_association_csv,
)


def _dino_metadata() -> dict[str, object]:
    return {
        "model_name": "dinov3_vitl16",
        "layer": 19,
        "weights_id": "sha256:abc",
        "long_edge": 1024,
        "downscale_only": False,
        "normalization_id": "rgb-imagenet-mean-std-v1",
        "resize_id": "opencv-inter-area-int-truncate-v1",
        "padding_id": "bottom-right-zero-to-patch-grid-v1",
    }


class SchemaTests(unittest.TestCase):
    def test_dino_round_trip_and_grid_crop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.dino.npz"
            source = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2)
            save_dino_map(path, source, 16, (32, 48), (31, 45), _dino_metadata())
            loaded = load_dino_map(path)
            self.assertEqual(loaded.descriptor_map.shape, (2, 3, 2))
            self.assertEqual(loaded.proc_hw, (32, 48))
            self.assertEqual(loaded.orig_hw, (31, 45))

    def test_dino_provenance_is_required_and_validated_when_expected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.dino.npz"
            metadata = _dino_metadata()
            save_dino_map(
                path,
                np.ones((2, 3, 4), np.float32),
                16,
                (32, 48),
                (31, 45),
                metadata,
            )
            loaded = load_dino_map(path, expected_metadata={"layer": 19})
            self.assertEqual(loaded.metadata["weights_id"], "sha256:abc")
            with self.assertRaisesRegex(ValueError, "Stale DINO cache"):
                load_dino_map(path, expected_metadata={"layer": 20})

    def test_legacy_dino_cache_warns_and_remains_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            np.savez(path, descriptor_map=np.ones((2, 3, 4), np.float32))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = load_dino_map(path)
            self.assertEqual(loaded.proc_hw, (32, 48))
            self.assertGreaterEqual(len(caught), 3)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with self.assertRaisesRegex(ValueError, "Unverifiable DINO cache"):
                    load_dino_map(
                        path, expected_metadata={"model_name": "dinov3_vitl16"}
                    )

    def test_superpoint_metadata_and_stale_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.spkp.npz"
            keypoints = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
            save_superpoint_cache(
                path,
                keypoints,
                long_edge=1024,
                max_num_keypoints=2048,
                downscale_only=False,
                weights_id="upstream-default",
                orig_hw=(800, 600),
                proc_hw=(1024, 768),
                source_path="image.jpg",
            )
            loaded = load_superpoint_cache(path, {"long_edge": 1024})
            np.testing.assert_array_equal(loaded.keypoints, keypoints)
            self.assertEqual(loaded.metadata["proc_hw"], (1024, 768))
            with self.assertRaisesRegex(ValueError, "Stale SuperPoint cache"):
                load_superpoint_cache(path, {"long_edge": 512})

    def test_superpoint_missing_metadata_requires_explicit_legacy_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.spkp.npz"
            np.savez(path, keypoints=np.array([[1.0, 2.0]], np.float32))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with self.assertRaisesRegex(ValueError, "Unverifiable SuperPoint cache"):
                    load_superpoint_cache(path, {"downscale_only": False})
                loaded = load_superpoint_cache(
                    path,
                    {"downscale_only": False},
                    allow_missing_metadata=True,
                )
            self.assertEqual(loaded.keypoints.shape, (1, 2))

    def test_association_header_empty_and_duplicate_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matching_001.csv"
            write_association_csv(path, [])
            self.assertEqual(validate_association_csv(path), 0)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(ASSOCIATION_COLUMNS)
                row = [0, 0, 1, 2, 3, 4, 0.9, 1]
                writer.writerow(row)
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "Duplicate association"):
                validate_association_csv(path)


if __name__ == "__main__":
    unittest.main()
