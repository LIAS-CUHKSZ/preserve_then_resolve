from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dino_m2m.matching import (
    ImageFeatures,
    build_association_rows,
    interpolate_descriptors,
    interpolate_resized_descriptors,
)


class InterpolationTests(unittest.TestCase):
    def test_patch_centers_and_boundary_filter_match_legacy_behavior(self) -> None:
        descriptor_map = np.array(
            [
                [[1.0, 0.0], [0.0, 2.0]],
                [[1.0, 1.0], [2.0, 2.0]],
            ],
            dtype=np.float32,
        )
        keypoints = np.array([[8, 8], [24, 8], [0, 0]], dtype=np.float32)
        kept, descriptors = interpolate_descriptors(keypoints, descriptor_map, 16)
        np.testing.assert_array_equal(kept, keypoints[:2])
        np.testing.assert_allclose(descriptors, [[1, 0], [0, 1]], atol=1e-7)

    def test_naflex_half_pixel_mapping_retains_source_coordinates(self) -> None:
        descriptor_map = np.zeros((2, 3, 2), dtype=np.float32)
        descriptor_map[0, 0] = [3.0, 0.0]
        descriptor_map[1, 2] = [0.0, 4.0]
        # For a 2x resize, these source coordinates map exactly to the two
        # selected patch centers: (x + .5) * 2 / 16 - .5.
        keypoints = np.array([[3.5, 3.5], [19.5, 11.5]], dtype=np.float32)
        kept, descriptors = interpolate_resized_descriptors(
            keypoints,
            descriptor_map,
            16,
            source_hw=(16, 24),
            resized_hw=(32, 48),
        )
        np.testing.assert_array_equal(kept, keypoints)
        np.testing.assert_allclose(descriptors, [[1, 0], [0, 1]], atol=1e-7)

    def test_naflex_mapping_rejects_grid_geometry_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "differs from descriptor grid"):
            interpolate_resized_descriptors(
                np.array([[1.0, 1.0]], dtype=np.float32),
                np.zeros((2, 3, 2), dtype=np.float32),
                16,
                source_hw=(16, 24),
                resized_hw=(31, 48),
            )

    def test_compact_ids_preserve_association_order(self) -> None:
        left = ImageFeatures(np.array([[0, 0], [1, 1], [2, 2]], np.float32), None, (0, 0))
        right = ImageFeatures(np.array([[3, 3], [4, 4], [5, 5]], np.float32), None, (0, 0))
        rows = build_association_rows(
            left,
            right,
            np.array([2, 0]),
            np.array([1, 2]),
            np.array([0.8, 0.7], np.float32),
            np.array([1, 2]),
        )
        self.assertEqual([row[:2] for row in rows], [[1, 0], [0, 1]])
        self.assertEqual([row[-1] for row in rows], [1, 2])


class ProgressiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ImportError:
            cls.torch = None
        else:
            cls.torch = torch

    def test_first_k_then_similarity_order(self) -> None:
        if self.torch is None:
            self.skipTest("PyTorch is not installed in the lightweight test environment")
        from dino_m2m.matching import progressive_mutual_knn

        left = self.torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        right = self.torch.tensor([[1.0, 0.0], [0.2, 0.98]])
        left_idx, right_idx, scores, ks = progressive_mutual_knn(left, right, 2, 0)
        self.assertTrue(np.all(ks[:-1] <= ks[1:]))
        for first_k in np.unique(ks):
            within = scores[ks == first_k]
            self.assertTrue(np.all(within[:-1] >= within[1:]))
        self.assertEqual(set(zip(left_idx.tolist(), right_idx.tolist())), {
            (0, 0), (0, 1), (1, 0), (1, 1)
        })

    def test_basis_metadata_is_preserved_and_validated(self) -> None:
        if self.torch is None:
            self.skipTest("PyTorch is not installed in the lightweight test environment")
        from dino_m2m.matching import load_basis_payload

        metadata = {
            "model_name": "dinov3_vitl16",
            "layer": 19,
            "weights_id": "sha256:abc",
            "normalization_id": "rgb-imagenet-mean-std-v1",
            "patch_size": 16,
            "image_height": 32,
            "image_width": 48,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "basis.pt"
            self.torch.save(
                {
                    "basis": self.torch.eye(4),
                    "max_rank": 4,
                    "meta": metadata,
                },
                path,
            )
            loaded = load_basis_payload(path, self.torch.device("cpu"), metadata)
            self.assertEqual(loaded["meta"], metadata)
            self.assertEqual(loaded["max_rank"], 4)
            with self.assertRaisesRegex(ValueError, "Stale debiasing basis"):
                load_basis_payload(
                    path,
                    self.torch.device("cpu"),
                    {**metadata, "layer": 20},
                )

    def test_basis_rank_is_selected_by_prefix_columns(self) -> None:
        from dino_m2m.matching import basis_for_dim

        full = np.arange(20, dtype=np.float32).reshape(4, 5)
        selected = basis_for_dim({"basis": full, "max_rank": 5}, 2, 4)
        np.testing.assert_array_equal(selected, full[:, :2])
        with self.assertRaisesRegex(ValueError, "exceeds stored max_rank"):
            basis_for_dim({"basis": full, "max_rank": 5}, 6, 4)

    def test_basis_loader_rejects_noncanonical_payload(self) -> None:
        from dino_m2m.matching import load_basis_payload

        class FakeTorch:
            float32 = object()

            @staticmethod
            def load(path, map_location):
                return {"basis_by_dim": {500: object()}, "meta": {}}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "basis.pt"
            path.touch()
            with (
                patch("dino_m2m.matching.require_torch", return_value=(FakeTorch, None)),
                self.assertRaisesRegex(ValueError, "Unsupported debiasing basis payload"),
            ):
                load_basis_payload(path, "cpu")

    def test_random_case_matches_original_incremental_k_loop(self) -> None:
        if self.torch is None:
            self.skipTest("PyTorch is not installed in the lightweight test environment")
        from dino_m2m.matching import progressive_mutual_knn

        torch = self.torch
        functional = torch.nn.functional
        generator = torch.Generator().manual_seed(42)
        left = torch.randn(17, 8, generator=generator)
        right = torch.randn(19, 8, generator=generator)

        def original(max_k: int, upperbound: int):
            normalized_left = functional.normalize(left, p=2, dim=1)
            normalized_right = functional.normalize(right, p=2, dim=1)
            similarity = normalized_left @ normalized_right.T
            left_ranked = torch.topk(similarity, k=max_k, dim=1).indices
            right_ranked = torch.topk(similarity, k=max_k, dim=0).indices
            first_seen = {}
            for k in range(1, max_k + 1):
                left_mask = torch.zeros_like(similarity, dtype=torch.bool)
                right_mask = torch.zeros_like(similarity, dtype=torch.bool)
                left_mask.scatter_(1, left_ranked[:, :k], True)
                right_mask.scatter_(0, right_ranked[:k, :], True)
                for i, j in (left_mask & right_mask).nonzero().numpy():
                    first_seen.setdefault((int(i), int(j)), k)
                if upperbound and len(first_seen) >= upperbound:
                    break
            packed = np.array([(i, j, k) for (i, j), k in first_seen.items()], dtype=np.int64)
            scores = similarity[packed[:, 0], packed[:, 1]].numpy().astype(np.float32)
            order = np.lexsort((-scores, packed[:, 2]))[:upperbound or None]
            return packed[order], scores[order]

        for max_k in (1, 5):
            with self.subTest(max_k=max_k):
                expected_pairs, expected_scores = original(max_k, 20)
                left_idx, right_idx, scores, ks = progressive_mutual_knn(
                    left, right, max_k, 20
                )
                actual_pairs = np.column_stack((left_idx, right_idx, ks))
                np.testing.assert_array_equal(actual_pairs, expected_pairs)
                np.testing.assert_array_equal(scores, expected_scores)
                if max_k == 1:
                    np.testing.assert_array_equal(ks, np.ones_like(ks))


if __name__ == "__main__":
    unittest.main()
