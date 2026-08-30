from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dino_m2m.config import apply_dataset_override, deep_merge, get_value, load_config


class ConfigTests(unittest.TestCase):
    def test_recursive_merge_and_dataset_override(self) -> None:
        base = {
            "matching": {"max_k": [5], "association_upperbound": 2048},
            "resize": {"downscale_only": False},
            "dataset_overrides": {"METU-CC": {"resize": {"downscale_only": True}}},
        }
        merged = deep_merge(base, {"matching": {"max_k": [3]}})
        self.assertEqual(get_value(merged, "matching.max_k"), [3])
        self.assertEqual(get_value(merged, "matching.association_upperbound"), 2048)
        metu = apply_dataset_override(merged, "METU-CC")
        self.assertTrue(get_value(metu, "resize.downscale_only"))
        self.assertFalse(get_value(merged, "resize.downscale_only"))
        with self.assertRaisesRegex(ValueError, "Unknown dataset override"):
            apply_dataset_override(merged, "typo-dataset")

    def test_yaml_paths_are_relative_to_config(self) -> None:
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML is not installed in the lightweight test environment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text(
                "paths:\n  image_root: data/images\n  dinov2_source: third_party/dinov2\n"
                "matching:\n  max_k: [5]\n",
                encoding="utf-8",
            )
            config = load_config(config_path)
            self.assertEqual(config["paths"]["image_root"], root / "data/images")
            self.assertEqual(
                config["paths"]["dinov2_source"], root / "third_party/dinov2"
            )
            self.assertEqual(config["matching"]["max_k"], [5])


if __name__ == "__main__":
    unittest.main()
