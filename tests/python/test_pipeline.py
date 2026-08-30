from __future__ import annotations

import json
import tempfile
import unittest
import weakref
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dino_m2m.matching import ImageFeatures
from dino_m2m.pipeline import MatchOptions, run_matching
from dino_m2m.schemas import load_dino_map, save_dino_map, validate_association_csv
from dino_m2m.superpoint import SuperPointConfig


class PipelineFailureTests(unittest.TestCase):
    def test_dinov2_cache_source_revision_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dino_root = root / "dino"
            dino_root.mkdir()
            cache = dino_root / "frame.dino.npz"
            metadata = {
                "model_name": "dinov2_vitl14_reg",
                "layer": 24,
                "weights_id": "sha256:test",
                "long_edge": 1024,
                "downscale_only": False,
                "normalization_id": "rgb-imagenet-mean-std-v1",
                "resize_id": "opencv-inter-area-int-truncate-v1",
                "padding_id": "bottom-right-zero-to-patch-grid-v1",
                "model_family": "dinov2",
                "descriptor_dim": 1024,
                "register_tokens": 4,
                "correction": "none",
                "source_revision": "old-revision",
                "source_dirty": False,
            }
            save_dino_map(
                cache,
                np.zeros((2, 3, 1024), np.float32),
                14,
                (28, 42),
                (28, 42),
                metadata,
            )
            expected = dict(metadata)
            expected["source_revision"] = "current-revision"
            with self.assertRaisesRegex(ValueError, "source_revision"):
                load_dino_map(cache, expected_metadata=expected)

    def test_dinov2_rejects_nonzero_correction_before_io(self) -> None:
        options = MatchOptions(
            pair_file=Path("missing-pairs.txt"),
            image_root=Path("missing-images"),
            dino_root=Path("missing-dino"),
            keypoint_cache_root=Path("missing-keypoints"),
            output_root=Path("missing-output"),
            weights=Path("missing-weights.pth"),
            model_name="dinov2_vitl14_reg",
            layer=24,
            svd_components=(200,),
        )
        with self.assertRaisesRegex(ValueError, "does not support"):
            run_matching(options)

    def test_dinov2_patch_size_is_profile_driven(self) -> None:
        options = MatchOptions(
            pair_file=Path("missing-pairs.txt"),
            image_root=Path("missing-images"),
            dino_root=Path("missing-dino"),
            keypoint_cache_root=Path("missing-keypoints"),
            output_root=Path("missing-output"),
            weights=Path("missing-weights.pth"),
            model_name="dinov2_vitl14_reg",
            layer=24,
            patch_size=16,
            svd_components=(0,),
        )
        with self.assertRaisesRegex(ValueError, "patch size 14"):
            run_matching(options)

    def test_later_variant_failure_preserves_completed_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            dino_root = root / "dino"
            keypoint_root = root / "keypoints"
            output_root = root / "output"
            for path in (image_root, dino_root, keypoint_root):
                path.mkdir()
            pair_file = root / "pairs.txt"
            pair_file.write_text("left.jpg right.jpg\n", encoding="utf-8")
            weights = root / "weights.pth"
            weights.write_bytes(b"checkpoint")
            features = ImageFeatures(
                keypoints=np.array([[1.0, 2.0]], np.float32),
                descriptors=np.ones((1, 2), np.float32),
                image_size=(16, 16),
            )
            successful = (
                np.array([0], np.int64),
                np.array([0], np.int64),
                np.array([0.9], np.float32),
                np.array([1], np.int64),
            )
            options = MatchOptions(
                pair_file=pair_file,
                image_root=image_root,
                dino_root=dino_root,
                keypoint_cache_root=keypoint_root,
                output_root=output_root,
                dino_weights=weights,
                svd_components=(0,),
                max_ks=(1, 2),
                superpoint=SuperPointConfig(expected_weights_id="test-default-weights"),
            )
            with (
                patch("dino_m2m.pipeline.resolve_device", return_value="cpu"),
                patch("dino_m2m.pipeline._load_features", return_value=features),
                patch(
                    "dino_m2m.pipeline.progressive_mutual_knn",
                    side_effect=(successful, RuntimeError("second variant failed")),
                ),
            ):
                summary = run_matching(options)

            first = output_root / "debias_svd0" / "progressive_k1" / "matching_001.csv"
            second = output_root / "debias_svd0" / "progressive_k2" / "matching_001.csv"
            self.assertEqual(validate_association_csv(first), 1)
            self.assertFalse(second.exists())
            self.assertEqual(summary.output_count, 1)
            self.assertEqual(summary.failure_count, 1)
            association_manifest = json.loads(
                (first.parent / "association_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(association_manifest["pair_count"], 1)
            self.assertEqual(association_manifest["progressive_max_k"], 1)

            # Simulate an interrupted/legacy caller leaving a file at the exact
            # path recorded as incomplete.  Resume must not treat it as success.
            from dino_m2m.matching import write_association_csv

            write_association_csv(second, [])

            with (
                patch("dino_m2m.pipeline.resolve_device", return_value="cpu"),
                patch("dino_m2m.pipeline._load_features", return_value=features),
                patch(
                    "dino_m2m.pipeline.progressive_mutual_knn",
                    return_value=successful,
                ) as resumed_match,
            ):
                resumed = run_matching(replace(options, existing="skip"))
            self.assertEqual(resumed_match.call_count, 1)
            self.assertEqual(resumed.output_count, 1)
            self.assertEqual(resumed.failure_count, 0)
            self.assertEqual(validate_association_csv(first), 1)
            self.assertEqual(validate_association_csv(second), 1)
            self.assertEqual(
                json.loads(resumed.failure_manifest.read_text(encoding="utf-8")), []
            )

            pair_file.write_text("left.jpg another.jpg\n", encoding="utf-8")
            with (
                patch("dino_m2m.pipeline.resolve_device", return_value="cpu"),
                self.assertRaisesRegex(ValueError, "Association manifest"),
            ):
                run_matching(replace(options, existing="skip"))

    def test_feature_cache_evicts_after_last_pending_pair_and_loads_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            dino_root = root / "dino"
            keypoint_root = root / "keypoints"
            output_root = root / "output"
            for path in (image_root, dino_root, keypoint_root):
                path.mkdir()
            pair_file = root / "pairs.txt"
            pair_file.write_text(
                "a.jpg b.jpg\na.jpg c.jpg\nd.jpg d.jpg\n", encoding="utf-8"
            )
            weights = root / "weights.pth"
            weights.write_bytes(b"checkpoint")
            calls: list[Path] = []
            references: dict[Path, weakref.ReferenceType[ImageFeatures]] = {}

            def load_feature(image_rel: Path, *_args: object) -> ImageFeatures:
                # b is dead before c loads; a and c are dead before d loads.
                if image_rel == Path("c.jpg"):
                    self.assertIsNone(references[Path("b.jpg")]())
                    self.assertIsNotNone(references[Path("a.jpg")]())
                if image_rel == Path("d.jpg"):
                    self.assertIsNone(references[Path("a.jpg")]())
                    self.assertIsNone(references[Path("c.jpg")]())
                calls.append(image_rel)
                feature = ImageFeatures(
                    keypoints=np.array([[1.0, 2.0]], np.float32),
                    descriptors=np.ones((1, 2), np.float32),
                    image_size=(16, 16),
                )
                references[image_rel] = weakref.ref(feature)
                return feature

            successful = (
                np.array([0], np.int64),
                np.array([0], np.int64),
                np.array([0.9], np.float32),
                np.array([1], np.int64),
            )
            options = MatchOptions(
                pair_file=pair_file,
                image_root=image_root,
                dino_root=dino_root,
                keypoint_cache_root=keypoint_root,
                output_root=output_root,
                dino_weights=weights,
                svd_components=(0,),
                max_ks=(1,),
                superpoint=SuperPointConfig(expected_weights_id="test-default-weights"),
            )
            with (
                patch("dino_m2m.pipeline.resolve_device", return_value="cpu"),
                patch("dino_m2m.pipeline._load_features", side_effect=load_feature),
                patch(
                    "dino_m2m.pipeline.progressive_mutual_knn",
                    return_value=successful,
                ),
            ):
                summary = run_matching(options)

            self.assertEqual(
                calls,
                [Path(name) for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg")],
            )
            self.assertEqual(summary.failure_count, 0)
            self.assertEqual(summary.output_count, 3)


if __name__ == "__main__":
    unittest.main()
