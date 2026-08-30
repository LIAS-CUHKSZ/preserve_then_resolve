from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from dino_m2m.superpoint import (
    CacheBackedSuperPoint,
    SuperPointConfig,
    _state_dict_sha256,
)


class SuperPointCacheTests(unittest.TestCase):
    def test_default_state_identity_changes_with_tensor_contents(self) -> None:
        class FakeTensor:
            def __init__(self, values):
                self.values = np.asarray(values, np.float32)

            def detach(self):
                return self

            def cpu(self):
                return self

            def contiguous(self):
                return self

            def numpy(self):
                return self.values

        self.assertNotEqual(
            _state_dict_sha256({"weight": FakeTensor([1.0, 2.0])}),
            _state_dict_sha256({"weight": FakeTensor([1.0, 3.0])}),
        )

    def test_cache_only_default_weights_require_expected_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "superpoint-weights-id"):
                CacheBackedSuperPoint(root, root, SuperPointConfig())
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                CacheBackedSuperPoint(
                    root,
                    root,
                    SuperPointConfig(expected_weights_id="upstream-default"),
                )

    def test_custom_weight_identity_uses_checkpoint_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a" / "weights.pth"
            second = root / "b" / "weights.pth"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            self.assertNotEqual(
                SuperPointConfig(weights=first).weights_id,
                SuperPointConfig(weights=second).weights_id,
            )

    def test_legacy_filename_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            cache_root = root / "cache"
            image_root.mkdir()
            cache_root.mkdir()
            Image.new("RGB", (16, 16)).save(image_root / "frame.jpg")
            config = SuperPointConfig(expected_weights_id="test-default-weights")
            strict_cache = CacheBackedSuperPoint(image_root, cache_root, config)
            legacy_path = strict_cache.legacy_cache_path(Path("frame.jpg"))
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(legacy_path, keypoints=np.array([[8.0, 8.0]], np.float32))

            with self.assertRaisesRegex(FileNotFoundError, "allow-legacy-keypoint-cache"):
                strict_cache.load_or_extract(Path("frame.jpg"))

            opted_in = CacheBackedSuperPoint(
                image_root, cache_root, config, allow_legacy_cache=True
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loaded = opted_in.load_or_extract(Path("frame.jpg"))
            np.testing.assert_array_equal(loaded.keypoints, [[8.0, 8.0]])


if __name__ == "__main__":
    unittest.main()
