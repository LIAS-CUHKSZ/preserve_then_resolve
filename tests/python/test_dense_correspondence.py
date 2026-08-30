from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from PIL import Image

from dino_m2m.dino import backbone_provenance, dino_artifact_metadata
from dino_m2m.provenance import checkpoint_identity
from dino_m2m.schemas import save_dino_map
from evaluation.dense_correspondence.evaluate import (
    DensePair,
    DenseEvaluationOptions,
    _DescriptorCache,
    _descriptor_extraction_provenance,
    _descriptor_snapshot_id,
    _resume_may_repair,
    _validate_options,
    audit_split_disjointness,
    evaluate_dense_correspondence,
)
from evaluation.dense_correspondence.geometry import (
    build_patch_geometry,
    candidate_correspondence_errors,
    decode_navi_inverse_depth,
    minimum_object_correspondence_errors,
)
from evaluation.dense_correspondence.protocol import (
    DINOV2_MODEL_NAME,
    DINOV2_PROTOCOL_SPECIFICATION,
    PROTOCOL_SPECIFICATION,
    RAW_SHARD_SCHEMA_VERSION,
    _complete_indices,
    atomic_save_shard,
    canonical_json,
    make_protocol_payload,
    protocol_fingerprint,
    validate_shard,
)
from evaluation.dense_correspondence.ranking import (
    compute_bidirectional_topk,
    mutual_entry_ranks,
)
from evaluation.dense_correspondence.summarize import (
    DenseSummaryOptions,
    compute_direction_cdf,
    summarize_dense_correspondence,
)
from evaluation.dense_correspondence.plot_extended_rank_layer_curves import (
    BINS as LAYER_SWEEP_BINS,
    LAYERS as LAYER_SWEEP_LAYERS,
    MAX_K as LAYER_SWEEP_MAX_K,
    _annotate_selected_layer_gains,
    _curve_gain_pp,
    _curve_point,
    _load_points,
    _selection_provenance,
    _select_best_raw_layer,
    _uniform_descriptor_provenance,
    _write_selection_report,
)


def _annotation(height: int, width: int) -> dict:
    return {
        "image_size": [height, width],
        "camera": {
            "q": [1.0, 0.0, 0.0, 0.0],
            "t": [0.0, 0.0, 0.0],
            "focal_length": 100.0,
        },
    }


def _raw_shard_payload(
    *,
    pair_index: int = 1,
    layer: int = 16,
    object_name: str = "test_object",
    successful: bool = True,
    fingerprint: str | None = None,
) -> dict[str, np.ndarray]:
    max_k = 4
    debias_ranks = np.array([0, 10], dtype=np.int32)
    source = np.array([0, 1], dtype=np.int64)
    target_row = np.array([0, 1, 2, 3], dtype=np.int64)
    target = np.tile(target_row, (2, 2, 1))
    cosine = np.tile(
        np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32), (2, 2, 1)
    )
    target_object = np.ones((2, 2, max_k), dtype=np.bool_)
    target_depth = np.ones((2, 2, max_k), dtype=np.bool_)
    if successful:
        error_one_rank = np.array(
            [[0.0, 0.03, 0.06, 0.10], [0.03, 0.015, 0.06, 0.10]],
            dtype=np.float64,
        )
    else:
        error_one_rank = np.full((2, max_k), 0.10, dtype=np.float64)
    errors = np.tile(error_one_rank, (2, 1, 1))
    entry_one_rank = np.array([[1, 2, 3, 4], [2, 4, 3, 4]], dtype=np.int16)
    entry = np.tile(entry_one_rank, (2, 1, 1))
    minimum = np.array([0.0, 0.015], dtype=np.float64)
    embedded_protocol = make_protocol_payload(
        angular_bin="0-40",
        pair_file_sha256="sha256:synthetic-correspondence-pairs",
        estimation_split_sha256="sha256:synthetic-estimation-pairs",
        model_name="dinov3_vitl16",
        weights_id="sha256:synthetic-weights",
        layer=layer,
        debias_ranks=debias_ranks.tolist(),
        basis_identities={"32x32": "sha256:synthetic-basis"},
        dataset_snapshot_id="sha256:synthetic-dataset-snapshot",
        descriptor_snapshot_id=f"sha256:synthetic-layer-{layer}-descriptors",
        long_edge=1024,
        patch_size=16,
        max_k=max_k,
    )
    protocol_json = canonical_json(embedded_protocol)
    fingerprint = fingerprint or protocol_fingerprint(embedded_protocol)
    payload: dict[str, np.ndarray] = {
        "schema_version": np.array(RAW_SHARD_SCHEMA_VERSION, dtype=np.int32),
        "protocol_fingerprint": np.array(fingerprint),
        "protocol_json": np.array(protocol_json),
        "angular_bin": np.array("0-40"),
        "pair_index": np.array(pair_index, dtype=np.int64),
        "layer": np.array(layer, dtype=np.int32),
        "debias_ranks": debias_ranks,
        "max_k": np.array(max_k, dtype=np.int32),
        "angle_degrees": np.array(15.0, dtype=np.float64),
        "object_name": np.array(object_name),
        "image_a": np.array(f"{object_name}/wild_set/images/{pair_index:03d}.jpg"),
        "image_b": np.array(f"{object_name}/wild_set/images/{pair_index + 10:03d}.jpg"),
        "image_a_grid_hw": np.array([2, 2], dtype=np.int32),
        "image_b_grid_hw": np.array([2, 2], dtype=np.int32),
        "image_a_resized_hw": np.array([32, 32], dtype=np.int32),
        "image_b_resized_hw": np.array([32, 32], dtype=np.int32),
        "image_a_object_patch_index": np.arange(4, dtype=np.int64),
        "image_b_object_patch_index": np.arange(4, dtype=np.int64),
        "mutual_match_count_at_k": np.array(
            [[2, 4, 6, 8], [1, 3, 5, 7]], dtype=np.int64
        ),
    }
    for direction in ("a_to_b", "b_to_a"):
        payload.update(
            {
                f"{direction}_source_patch_index": source,
                f"{direction}_source_min_object_error_m": minimum,
                f"{direction}_candidate_target_patch_index": target,
                f"{direction}_candidate_cosine": cosine,
                f"{direction}_candidate_target_is_object": target_object,
                f"{direction}_candidate_target_has_depth": target_depth,
                f"{direction}_candidate_error_m": errors,
                f"{direction}_candidate_mutual_entry_k": entry,
            }
        )
    return payload


def _evaluation_manifest(
    payloads: list[dict[str, np.ndarray]],
) -> dict[str, object]:
    if not payloads:
        raise ValueError("Synthetic manifest needs at least one shard payload")
    by_pair: dict[int, dict[str, np.ndarray]] = {}
    fingerprints: dict[int, str] = {}
    for payload in payloads:
        pair_index = int(np.asarray(payload["pair_index"]).item())
        layer = int(np.asarray(payload["layer"]).item())
        by_pair.setdefault(pair_index, payload)
        fingerprint = str(np.asarray(payload["protocol_fingerprint"]).item())
        previous = fingerprints.setdefault(layer, fingerprint)
        if previous != fingerprint:
            raise ValueError("Synthetic layer payloads need one protocol fingerprint")
    pair_indices = sorted(by_pair)
    layers = sorted(fingerprints)
    pair_identities = []
    for pair_index in pair_indices:
        payload = by_pair[pair_index]
        pair_identities.append(
            {
                "pair_index": pair_index,
                "object_name": str(np.asarray(payload["object_name"]).item()),
                "image_a": str(np.asarray(payload["image_a"]).item()),
                "image_b": str(np.asarray(payload["image_b"]).item()),
                "angle_degrees": float(np.asarray(payload["angle_degrees"]).item()),
            }
        )
    first = payloads[0]
    protocol = json.loads(str(np.asarray(first["protocol_json"]).item()))
    return {
        "angular_bin": str(np.asarray(first["angular_bin"]).item()),
        "pair_count": len(pair_indices),
        "pair_indices": pair_indices,
        "pair_identities": pair_identities,
        "layers": layers,
        "debias_ranks": [int(value) for value in first["debias_ranks"]],
        "max_k": int(np.asarray(first["max_k"]).item()),
        "expected_shards": len(pair_indices) * len(layers),
        "written_shards": len(payloads),
        "resumed_shards": 0,
        "recomputed_shards": 0,
        "protocol_fingerprints": {
            f"layer{layer}": fingerprints[layer] for layer in layers
        },
        "dataset_snapshot_id": protocol["dataset_snapshot_id"],
        "split_audit": {"passed": True},
    }


def _write_evaluation_manifest(
    root: Path, payloads: list[dict[str, np.ndarray]]
) -> dict[str, object]:
    manifest = _evaluation_manifest(payloads)
    path = root / "bin_0-40" / "evaluation_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


class DenseGeometryTests(unittest.TestCase):
    def test_inverse_depth_decode_is_float64_metres_and_zero_is_invalid(self) -> None:
        decoded = decode_navi_inverse_depth(
            np.array([[0, 65535], [32768, 1]], dtype=np.uint16)
        )
        self.assertEqual(decoded.dtype, np.float64)
        self.assertTrue(np.isnan(decoded[0, 0]))
        self.assertAlmostEqual(decoded[0, 1], 0.01)
        self.assertAlmostEqual(decoded[1, 1], 655.35)

    def test_patch_geometry_excludes_partial_padding_patches(self) -> None:
        depth = np.full((5, 7), 65535, dtype=np.uint16)
        mask = np.ones((5, 7), dtype=np.uint8)
        geometry = build_patch_geometry(
            raw_depth=depth,
            object_mask=mask,
            annotation=_annotation(5, 7),
            grid_hw=(3, 4),
            resized_hw=(5, 7),
            patch_size=2,
        )
        np.testing.assert_array_equal(
            geometry.complete_indices, np.array([0, 1, 2, 4, 5, 6])
        )
        np.testing.assert_array_equal(
            geometry.valid_object_indices, geometry.complete_indices
        )

    def test_complete_indices_use_dynamic_dinov2_patch14_geometry(self) -> None:
        complete = _complete_indices(
            np.array([3, 4], dtype=np.int32),
            np.array([29, 43], dtype=np.int32),
            patch_size=14,
        )
        # The padded grid is 3x4, but only the upper-left 2x3 patch centers
        # belong to complete native DINOv2 patches.
        np.testing.assert_array_equal(
            complete, np.array([0, 1, 2, 4, 5, 6], dtype=np.int64)
        )

    def test_patch_center_uses_pixel_index_resize_inverse_once(self) -> None:
        depth = np.zeros((32, 32), dtype=np.uint16)
        mask = np.zeros((32, 32), dtype=np.uint8)
        depth[16, 16] = 65535
        mask[16, 16] = 1
        geometry = build_patch_geometry(
            raw_depth=depth,
            object_mask=mask,
            annotation=_annotation(32, 32),
            grid_hw=(1, 1),
            resized_hw=(16, 16),
            patch_size=16,
        )
        np.testing.assert_array_equal(
            geometry.valid_object_indices, np.array([0], dtype=np.int64)
        )
        self.assertAlmostEqual(geometry.xyz_m[0, 2], 0.01)

    def test_background_candidate_keeps_raw_error_but_is_not_object(self) -> None:
        depth = np.full((2, 4), 65535, dtype=np.uint16)
        source_mask = np.ones((2, 4), dtype=np.uint8)
        destination_mask = source_mask.copy()
        destination_mask[:, 2:] = 0
        annotation = _annotation(2, 4)
        source = build_patch_geometry(
            raw_depth=depth,
            object_mask=source_mask,
            annotation=annotation,
            grid_hw=(1, 2),
            resized_hw=(2, 4),
            patch_size=2,
        )
        destination = build_patch_geometry(
            raw_depth=depth,
            object_mask=destination_mask,
            annotation=annotation,
            grid_hw=(1, 2),
            resized_hw=(2, 4),
            patch_size=2,
        )
        target_object, target_depth, error = candidate_correspondence_errors(
            source=source,
            destination=destination,
            source_indices=np.array([0], dtype=np.int64),
            destination_indices=np.array([1], dtype=np.int64),
            source_camera=annotation["camera"],
            destination_camera=annotation["camera"],
        )
        self.assertFalse(target_object[0])
        self.assertTrue(target_depth[0])
        self.assertTrue(np.isfinite(error[0]))

    def test_identity_minimum_object_error_is_zero(self) -> None:
        depth = np.full((4, 4), 65535, dtype=np.uint16)
        mask = np.ones((4, 4), dtype=np.uint8)
        annotation = _annotation(4, 4)
        geometry = build_patch_geometry(
            raw_depth=depth,
            object_mask=mask,
            annotation=annotation,
            grid_hw=(2, 2),
            resized_hw=(4, 4),
            patch_size=2,
        )
        errors = minimum_object_correspondence_errors(
            source=geometry,
            destination=geometry,
            source_indices=geometry.valid_object_indices,
            source_camera=annotation["camera"],
            destination_camera=annotation["camera"],
        )
        np.testing.assert_allclose(errors, 0.0, atol=1e-15)


class DenseRankingTests(unittest.TestCase):
    def test_mutual_entry_rank_is_maximum_of_two_directional_ranks(self) -> None:
        forward = np.array([[1, 0], [0, 1]], dtype=np.int64)
        reverse = np.array([[0, 1], [0, 1]], dtype=np.int64)
        entry, counts = mutual_entry_ranks(forward, reverse)
        self.assertEqual(entry[0, 0], 1)
        self.assertEqual(entry[0, 1], 2)
        np.testing.assert_array_equal(counts, np.array([1, 4], dtype=np.int64))

    def test_identity_descriptors_enter_mutual_knn_at_one(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        descriptors = torch.eye(5, dtype=torch.float32)
        result = compute_bidirectional_topk(descriptors, descriptors, max_k=3)
        np.testing.assert_array_equal(result.a_to_b_indices[:, 0], np.arange(5))
        np.testing.assert_array_equal(result.a_to_b_mutual_entry_k[:, 0], 1)
        self.assertEqual(result.mutual_match_count_at_k[0], 5)


class DenseSummaryTests(unittest.TestCase):
    def test_dinov2_protocol_is_raw_only_and_patch14(self) -> None:
        common = {
            "angular_bin": "0-40",
            "pair_file_sha256": "sha256:pairs",
            "estimation_split_sha256": "sha256:estimation",
            "model_name": DINOV2_MODEL_NAME,
            "weights_id": "sha256:weights",
            "layer": 16,
            "dataset_snapshot_id": "sha256:dataset",
            "descriptor_snapshot_id": "sha256:descriptors",
            "long_edge": 1024,
            "max_k": 8,
        }
        payload = make_protocol_payload(
            **common,
            debias_ranks=(0,),
            basis_identities={},
            patch_size=14,
        )
        self.assertEqual(payload["protocol"], DINOV2_PROTOCOL_SPECIFICATION)
        self.assertEqual(payload["protocol"]["positional_bias_correction"], "none")
        self.assertEqual(payload["debias_ranks"], [0])
        with self.assertRaisesRegex(ValueError, "raw-only"):
            make_protocol_payload(
                **common,
                debias_ranks=(0, 100),
                basis_identities={},
                patch_size=14,
            )
        with self.assertRaisesRegex(ValueError, "patch_size=14"):
            make_protocol_payload(
                **common,
                debias_ranks=(0,),
                basis_identities={},
                patch_size=16,
            )

    def test_directional_and_mutual_cdf_use_strict_threshold_and_gt_denominator(self) -> None:
        metrics = compute_direction_cdf(
            candidate_error_m=np.array(
                [[0.0, 0.03, 0.06, 0.1], [0.03, 0.015, 0.06, 0.1]],
                dtype=np.float64,
            ),
            candidate_target_is_object=np.ones((2, 4), dtype=np.bool_),
            candidate_target_has_depth=np.ones((2, 4), dtype=np.bool_),
            candidate_mutual_entry_k=np.array(
                [[1, 2, 3, 4], [2, 4, 3, 4]], dtype=np.int16
            ),
            source_min_object_error_m=np.array([0.0, 0.015], dtype=np.float64),
            threshold_m=0.02,
            max_k=4,
        )
        self.assertEqual(metrics["gt_coverage"], 1.0)
        self.assertEqual(metrics["directional_cdf_at_1"], 0.5)
        self.assertEqual(metrics["directional_cdf_at_2"], 1.0)
        self.assertEqual(metrics["mutual_cdf_at_2"], 0.5)
        self.assertEqual(metrics["mutual_cdf_at_4"], 1.0)

        equality = compute_direction_cdf(
            candidate_error_m=np.array([[0.02]], dtype=np.float64),
            candidate_target_is_object=np.ones((1, 1), dtype=np.bool_),
            candidate_target_has_depth=np.ones((1, 1), dtype=np.bool_),
            candidate_mutual_entry_k=np.ones((1, 1), dtype=np.int16),
            source_min_object_error_m=np.array([0.02], dtype=np.float64),
            threshold_m=0.02,
            max_k=1,
        )
        self.assertEqual(equality["n_gt"], 0)
        self.assertTrue(math.isnan(float(equality["directional_cdf_at_1"])))

    def test_summary_reports_equal_mean_across_default_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _raw_shard_payload()
            shard = root / "bin_0-40/shards/layer16/pair_000001.npz"
            atomic_save_shard(shard, payload)
            _write_evaluation_manifest(root, [payload])
            output = root / "bin_0-40/reports"
            summarize_dense_correspondence(
                DenseSummaryOptions(
                    shard_root=root,
                    angular_bin="0-40",
                    output_dir=output,
                    thresholds_m=(0.01, 0.02, 0.05),
                    max_k=4,
                )
            )
            with (output / "summary_cdf_threshold_mean.csv").open(newline="") as stream:
                row = next(
                    item
                    for item in csv.DictReader(stream)
                    if item["debias_rank"] == "0"
                )
            self.assertEqual(row["thresholds_m"], "0.01|0.02|0.05")
            self.assertEqual(row["threshold_count"], "3")
            self.assertAlmostEqual(float(row["directional_cdf_at_1"]), 5.0 / 6.0)
            self.assertAlmostEqual(float(row["mutual_cdf_at_1"]), 2.0 / 3.0)
            self.assertTrue((output / "summary_match_counts.csv").is_file())

    def test_ineligible_pair_excludes_conditional_cdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _raw_shard_payload()
            payload["b_to_a_source_min_object_error_m"] = np.full(
                2, np.inf, dtype=np.float64
            )
            payload["b_to_a_candidate_target_has_depth"] = np.zeros(
                (2, 2, 4), dtype=np.bool_
            )
            payload["b_to_a_candidate_error_m"] = np.full(
                (2, 2, 4), np.inf, dtype=np.float64
            )
            shard = root / "bin_0-40/shards/layer16/pair_000001.npz"
            atomic_save_shard(shard, payload)
            _write_evaluation_manifest(root, [payload])
            output = root / "bin_0-40/reports"
            summarize_dense_correspondence(
                DenseSummaryOptions(
                    shard_root=root,
                    angular_bin="0-40",
                    output_dir=output,
                    thresholds_m=(0.02,),
                    max_k=4,
                )
            )
            with (output / "pair_cdf.csv").open(newline="") as stream:
                pair = next(
                    row
                    for row in csv.DictReader(stream)
                    if row["debias_rank"] == "0"
                )
            self.assertEqual(pair["cdf_pair_eligible"], "0")
            self.assertTrue(math.isnan(float(pair["directional_cdf_at_1"])))
            with (output / "summary_cdf.csv").open(newline="") as stream:
                summary = next(
                    row
                    for row in csv.DictReader(stream)
                    if row["debias_rank"] == "0"
                )
            self.assertEqual(summary["cdf_eligible_category_count"], "0")
            self.assertTrue(math.isnan(float(summary["directional_cdf_at_1"])))

    def test_shard_validation_checks_current_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.npz"
            atomic_save_shard(path, _raw_shard_payload())
            data = validate_shard(
                path,
                expected_fingerprint=str(
                    np.asarray(_raw_shard_payload()["protocol_fingerprint"]).item()
                ),
                expected_bin="0-40",
                expected_pair_index=1,
                expected_layer=16,
                expected_debias_ranks=(0, 10),
                expected_max_k=4,
            )
            self.assertEqual(data["a_to_b_candidate_error_m"].dtype, np.float64)
            invalid = _raw_shard_payload()
            invalid["a_to_b_candidate_mutual_entry_k"][0, 0, 0] = 0
            atomic_save_shard(path, invalid)
            with self.assertRaisesRegex(ValueError, "mutual-entry"):
                validate_shard(path)

            invalid_object = _raw_shard_payload()
            invalid_object["a_to_b_candidate_target_is_object"][0, 0, 0] = False
            atomic_save_shard(path, invalid_object)
            with self.assertRaisesRegex(ValueError, "object flags"):
                validate_shard(path)

    def test_shard_validation_rejects_missing_or_incompatible_protocol_spec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            missing = _raw_shard_payload()
            missing_protocol = json.loads(
                str(np.asarray(missing["protocol_json"]).item())
            )
            missing_protocol.pop("protocol")
            missing["protocol_json"] = np.array(canonical_json(missing_protocol))
            missing["protocol_fingerprint"] = np.array(
                protocol_fingerprint(missing_protocol)
            )
            missing_path = root / "missing_protocol.npz"
            atomic_save_shard(missing_path, missing)
            with self.assertRaisesRegex(ValueError, "missing keys"):
                validate_shard(missing_path)

            incompatible = _raw_shard_payload()
            incompatible_protocol = json.loads(
                str(np.asarray(incompatible["protocol_json"]).item())
            )
            incompatible_protocol["protocol"]["patch_geometry_sampling"] = (
                "legacy-patch-sampling"
            )
            incompatible["protocol_json"] = np.array(
                canonical_json(incompatible_protocol)
            )
            incompatible["protocol_fingerprint"] = np.array(
                protocol_fingerprint(incompatible_protocol)
            )
            incompatible_path = root / "incompatible_protocol.npz"
            atomic_save_shard(incompatible_path, incompatible)
            with self.assertRaisesRegex(ValueError, "specification is incompatible"):
                validate_shard(incompatible_path)

            self.assertNotEqual(
                PROTOCOL_SPECIFICATION["patch_geometry_sampling"],
                "legacy-patch-sampling",
            )

    def test_resume_refuses_to_repair_schema_v1_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schema_v1.npz"
            payload = _raw_shard_payload()
            payload["schema_version"] = np.array(1, dtype=np.int32)
            atomic_save_shard(path, payload)
            before = path.read_bytes()
            fingerprint = str(
                np.asarray(payload["protocol_fingerprint"]).reshape(()).item()
            )
            with self.assertRaisesRegex(RuntimeError, "schema-v1"):
                _resume_may_repair(path, fingerprint)
            self.assertEqual(path.read_bytes(), before)

    def test_summary_rejects_extra_missing_and_identity_mismatched_shards(self) -> None:
        def summarize(root: Path) -> None:
            summarize_dense_correspondence(
                DenseSummaryOptions(
                    shard_root=root,
                    angular_bin="0-40",
                    output_dir=root / "bin_0-40" / "reports",
                    thresholds_m=(0.02,),
                    max_k=4,
                )
            )

        with self.subTest("extra shard"), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_dir = root / "bin_0-40" / "shards" / "layer16"
            included = _raw_shard_payload(pair_index=1)
            extra = _raw_shard_payload(pair_index=2)
            atomic_save_shard(shard_dir / "pair_000001.npz", included)
            atomic_save_shard(shard_dir / "pair_000002.npz", extra)
            _write_evaluation_manifest(root, [included])
            with self.assertRaisesRegex(ValueError, "outside the current manifest"):
                summarize(root)

        with self.subTest("missing shard"), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_dir = root / "bin_0-40" / "shards" / "layer16"
            present = _raw_shard_payload(pair_index=1)
            missing = _raw_shard_payload(pair_index=2)
            atomic_save_shard(shard_dir / "pair_000001.npz", present)
            _write_evaluation_manifest(root, [present, missing])
            with self.assertRaisesRegex(FileNotFoundError, "expected shards are missing"):
                summarize(root)

        with self.subTest(
            "pair identity mismatch"
        ), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_dir = root / "bin_0-40" / "shards" / "layer16"
            payload = _raw_shard_payload(pair_index=1)
            atomic_save_shard(shard_dir / "pair_000001.npz", payload)
            manifest = _write_evaluation_manifest(root, [payload])
            pair_identities = manifest["pair_identities"]
            assert isinstance(pair_identities, list)
            pair_identities[0]["image_a"] = "wrong_object/wild_set/images/999.jpg"
            manifest_path = root / "bin_0-40" / "evaluation_manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "(?i)identity"):
                summarize(root)

    def test_summary_rejects_cross_layer_shared_protocol_mixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layer16 = _raw_shard_payload(pair_index=1, layer=16)
            layer17 = _raw_shard_payload(pair_index=1, layer=17)
            protocol17 = json.loads(
                str(np.asarray(layer17["protocol_json"]).item())
            )
            protocol17["weights_id"] = "sha256:different-weights"
            layer17["protocol_json"] = np.array(canonical_json(protocol17))
            layer17["protocol_fingerprint"] = np.array(
                protocol_fingerprint(protocol17)
            )
            for layer, payload in ((16, layer16), (17, layer17)):
                atomic_save_shard(
                    root
                    / f"bin_0-40/shards/layer{layer}/pair_000001.npz",
                    payload,
                )
            _write_evaluation_manifest(root, [layer16, layer17])
            with self.assertRaisesRegex(ValueError, "do not share one"):
                summarize_dense_correspondence(
                    DenseSummaryOptions(
                        shard_root=root,
                        angular_bin="0-40",
                        output_dir=root / "bin_0-40/reports",
                        thresholds_m=(0.02,),
                        max_k=4,
                    )
                )

    def test_category_macro_does_not_weight_category_by_pair_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_dir = root / "bin_0-40" / "shards" / "layer16"
            payloads = [
                _raw_shard_payload(pair_index=1, object_name="easy", successful=True),
                _raw_shard_payload(pair_index=2, object_name="hard", successful=False),
                _raw_shard_payload(pair_index=3, object_name="hard", successful=False),
                _raw_shard_payload(pair_index=4, object_name="hard", successful=False),
            ]
            for payload in payloads:
                pair_index = int(payload["pair_index"])
                atomic_save_shard(shard_dir / f"pair_{pair_index:06d}.npz", payload)
            _write_evaluation_manifest(root, payloads)
            output = root / "bin_0-40" / "reports"
            result = summarize_dense_correspondence(
                DenseSummaryOptions(
                    shard_root=root,
                    angular_bin="0-40",
                    output_dir=output,
                    thresholds_m=(0.02,),
                    max_k=4,
                )
            )
            self.assertEqual(result.direction_row_count, 16)
            self.assertEqual(result.pair_row_count, 8)
            self.assertEqual(result.category_row_count, 4)
            with (output / "summary_cdf.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            rank_zero = next(row for row in rows if row["debias_rank"] == "0")
            self.assertAlmostEqual(float(rank_zero["directional_cdf_at_2"]), 0.5)
            self.assertTrue((output / "summary_match_counts.csv").is_file())


class DenseDinoV2LayerSelectionTests(unittest.TestCase):
    @staticmethod
    def _points(recall_by_layer: dict[int, float]) -> pd.DataFrame:
        rows = []
        for angular_bin in LAYER_SWEEP_BINS:
            for layer in LAYER_SWEEP_LAYERS:
                for candidate_k in range(1, LAYER_SWEEP_MAX_K + 1):
                    rows.append(
                        {
                            "angular_bin": angular_bin,
                            "layer": layer,
                            "debias_rank": 0,
                            "candidate_k": candidate_k,
                            "mean_mutual_association_count": 100.0 * candidate_k,
                            "mutual_cdf": recall_by_layer[layer],
                        }
                    )
        return pd.DataFrame(rows)

    def test_frontier_regret_formula_is_percentage_point_mean_over_bins_and_k(self) -> None:
        recall = {layer: 0.7 for layer in LAYER_SWEEP_LAYERS}
        recall[16] = 0.8
        regret, selection = _select_best_raw_layer(self._points(recall))
        by_layer = regret.set_index("layer")
        self.assertAlmostEqual(float(by_layer.loc[16, "mean_regret_pp"]), 0.0)
        self.assertAlmostEqual(float(by_layer.loc[17, "mean_regret_pp"]), 10.0)
        self.assertAlmostEqual(
            float(by_layer.loc[17, "worst_bin_mean_regret_pp"]), 10.0
        )
        self.assertEqual(selection["selected_layer"], 16)
        self.assertEqual(selection["best_layer"], 16)

    def test_selected_layer_gain_is_direct_k1_to_k5_percentage_points(self) -> None:
        points = self._points({layer: 0.5 for layer in LAYER_SWEEP_LAYERS})
        mask = (
            (points["angular_bin"] == "0-40")
            & (points["layer"] == 16)
            & (points["candidate_k"] == 5)
        )
        points.loc[mask, "mutual_cdf"] = 0.845
        self.assertAlmostEqual(_curve_gain_pp(points, "0-40", 16), 34.5)

    def test_gain_labels_are_centered_directly_above_k5_points(self) -> None:
        points = self._points({layer: 0.5 for layer in LAYER_SWEEP_LAYERS})
        axis = MagicMock()
        _annotate_selected_layer_gains(axis, points, 16)

        label_calls = [
            call
            for call in axis.annotate.call_args_list
            if call.args and call.args[0]
        ]
        self.assertEqual(len(label_calls), len(LAYER_SWEEP_BINS))
        for angular_bin, call in zip(LAYER_SWEEP_BINS, label_calls, strict=True):
            self.assertEqual(
                call.kwargs["xy"], _curve_point(points, angular_bin, 16, 5)
            )
            self.assertEqual(call.kwargs["xytext"], (0.0, 5.0))
            self.assertEqual(call.kwargs["ha"], "center")
            self.assertEqual(call.kwargs["va"], "bottom")

    def test_equivalent_selection_report_preserves_bound_file_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "best_layer.json"
            original = {
                "schema_version": 1,
                "selected_layer": 24,
                "selection_script_sha256": "old-script",
            }
            original_bytes = json.dumps(original, indent=4).encode("utf-8")
            path.write_bytes(original_bytes)

            updated = dict(original, selection_script_sha256="new-script")
            self.assertFalse(_write_selection_report(path, updated))
            self.assertEqual(path.read_bytes(), original_bytes)

            updated["selected_layer"] = 23
            self.assertTrue(_write_selection_report(path, updated))
            self.assertEqual(json.loads(path.read_text())["selected_layer"], 23)

    def test_frontier_regret_exact_tie_selects_lowest_layer(self) -> None:
        tied = {layer: 0.5 for layer in LAYER_SWEEP_LAYERS}
        regret, selection = _select_best_raw_layer(self._points(tied))
        self.assertTrue(np.allclose(regret["mean_regret_pp"], 0.0))
        self.assertEqual(selection["selected_layer"], min(LAYER_SWEEP_LAYERS))
        self.assertEqual(selection["best_layer"], min(LAYER_SWEEP_LAYERS))
        self.assertEqual(
            selection["tie_break"][-1], "lowest one-based layer number"
        )

    def test_frontier_regret_duplicate_association_counts_use_all_equal_x_points(self) -> None:
        points = self._points({layer: 0.5 for layer in LAYER_SWEEP_LAYERS})
        points.loc[points["layer"] == 16, "mutual_cdf"] = 0.8
        regret, selection = _select_best_raw_layer(points)
        by_layer = regret.set_index("layer")
        self.assertAlmostEqual(float(by_layer.loc[16, "mean_regret_pp"]), 0.0)
        self.assertAlmostEqual(float(by_layer.loc[17, "mean_regret_pp"]), 30.0)
        self.assertEqual(selection["best_layer"], 16)

    def test_frontier_regret_rejects_duplicate_layer_k_rows(self) -> None:
        points = self._points({layer: 0.5 for layer in LAYER_SWEEP_LAYERS})
        malformed = points.copy()
        last_bin_start = len(malformed) - len(LAYER_SWEEP_LAYERS) * LAYER_SWEEP_MAX_K
        malformed.iloc[-1] = malformed.iloc[last_bin_start]
        with self.assertRaisesRegex(ValueError, "Duplicate DINOv2 layer/K"):
            _select_best_raw_layer(malformed)


class DenseEvaluatorIntegrationTests(unittest.TestCase):
    def test_dinov2_descriptor_manifest_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "extraction manifest"):
                _descriptor_extraction_provenance(
                    Path(directory),
                    (16,),
                    model_name=DINOV2_MODEL_NAME,
                    weights_id="sha256:weights",
                    patch_size=14,
                )

    def test_descriptor_manifest_provenance_must_be_uniform_across_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = backbone_provenance(DINOV2_MODEL_NAME, correction="none")
            for layer in (16, 17):
                path = root / f"layer{layer}" / "extraction_manifest.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "model_name": DINOV2_MODEL_NAME,
                            "layer": layer,
                            "weights_id": "sha256:weights",
                            **profile,
                            "source_revision": "abc123",
                            "source_dirty": False,
                        }
                    ),
                    encoding="utf-8",
                )
            provenance = _descriptor_extraction_provenance(
                root,
                (16, 17),
                model_name=DINOV2_MODEL_NAME,
                weights_id="sha256:weights",
                patch_size=14,
            )
            self.assertEqual(provenance["source_revision"], "abc123")
            self.assertEqual(provenance["backbone"], profile)
            layer17 = root / "layer17" / "extraction_manifest.json"
            changed = json.loads(layer17.read_text(encoding="utf-8"))
            changed["source_revision"] = "different"
            layer17.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one source_revision"):
                _descriptor_extraction_provenance(
                    root,
                    (16, 17),
                    model_name=DINOV2_MODEL_NAME,
                    weights_id="sha256:weights",
                    patch_size=14,
                )

    def test_dinov2_descriptor_snapshot_binds_extraction_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layer_root = root / "layer16"
            image_a = Path("object/wild_set/images/a.jpg")
            image_b = Path("object/wild_set/images/b.jpg")
            for image in (image_a, image_b):
                path = (layer_root / image).with_suffix(".dino.npz")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"descriptor")
            manifest = layer_root / "extraction_manifest.json"
            manifest.write_text('{"source_revision":"first"}', encoding="utf-8")
            pair = DensePair(
                pair_index=1,
                image_a=image_a,
                image_b=image_b,
                angle_degrees=10.0,
                object_name="object",
                image_a_hw=(32, 32),
                image_b_hw=(32, 32),
            )
            first = _descriptor_snapshot_id(
                root, 16, (pair,), bind_extraction_manifest=True
            )
            manifest.write_text('{"source_revision":"other"}', encoding="utf-8")
            second = _descriptor_snapshot_id(
                root, 16, (pair,), bind_extraction_manifest=True
            )
            self.assertNotEqual(first, second)

    def test_dinov2_descriptor_array_dimension_must_match_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Path("frame.jpg")
            metadata = {
                **dino_artifact_metadata(
                    model_name=DINOV2_MODEL_NAME,
                    layer=16,
                    weights_id="sha256:weights",
                    long_edge=1024,
                    downscale_only=False,
                ),
                **{
                    key: value
                    for key, value in backbone_provenance(
                        DINOV2_MODEL_NAME, correction="none"
                    ).items()
                    if key != "patch_size"
                },
                "source_revision": "revision",
                "source_dirty": False,
            }
            cache_path = (root / "layer16" / image).with_suffix(".dino.npz")
            save_dino_map(
                cache_path,
                np.zeros((2, 2, 768), dtype=np.float32),
                14,
                (28, 28),
                (28, 28),
                metadata,
            )
            expected = {**metadata, "patch_size": 14}
            cache = _DescriptorCache(
                dino_root=root,
                layer=16,
                expected_metadata=expected,
                device="cpu",
                capacity=1,
            )
            with self.assertRaisesRegex(ValueError, "dimension"):
                cache.get(image)

    def test_dinov2_plot_rejects_incomplete_execution_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "bin_0-40" / "evaluation_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "model_name": DINOV2_MODEL_NAME,
                        "layers": list(LAYER_SWEEP_LAYERS),
                        "debias_ranks": [0],
                        "patch_size": 14,
                        "correction_mode": "none",
                        "long_edge": 1024,
                        "max_k": LAYER_SWEEP_MAX_K,
                        "pair_count": 500,
                        "expected_shards": 4500,
                        "written_shards": 4499,
                        "resumed_shards": 0,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Incomplete DINOv2 shard"):
                _load_points(
                    root,
                    expected_model_name=DINOV2_MODEL_NAME,
                    expected_debias_ranks=(0,),
                )

    def test_layer_selection_provenance_hashes_numeric_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_hashes = {}
            for angular_bin in LAYER_SWEEP_BINS:
                bin_root = root / f"bin_{angular_bin}"
                reports = bin_root / "reports"
                reports.mkdir(parents=True)
                manifest = bin_root / "evaluation_manifest.json"
                manifest.write_text(
                    json.dumps(
                        {
                            "weights_id": "sha256:weights",
                            "descriptor_provenance": {
                                "source_revision": "revision",
                                "source_dirty": False,
                                "backbone": backbone_provenance(
                                    DINOV2_MODEL_NAME, correction="none"
                                ),
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                (reports / "summary_cdf.csv").write_text(
                    "value\n1\n", encoding="utf-8"
                )
                (reports / "summary_match_counts.csv").write_text(
                    "value\n2\n", encoding="utf-8"
                )
                manifest_hashes[angular_bin] = "manifest-hash"
            provenance = _selection_provenance(root, manifest_hashes)
            for record in provenance.values():
                self.assertEqual(
                    set(record["numeric_report_sha256"]),
                    {"summary_cdf.csv", "summary_match_counts.csv"},
                )
                self.assertTrue(
                    all(
                        len(value) == 64
                        for value in record["numeric_report_sha256"].values()
                    )
                )
            uniform = _uniform_descriptor_provenance(provenance)
            self.assertEqual(uniform["weights_id"], "sha256:weights")

            provenance["80-120"]["weights_id"] = "sha256:other"
            with self.assertRaisesRegex(ValueError, "checkpoint identity"):
                _uniform_descriptor_provenance(provenance)

            provenance["80-120"]["weights_id"] = "sha256:weights"
            provenance["80-120"]["descriptor_provenance"]["backbone"][
                "descriptor_dim"
            ] = 768
            with self.assertRaisesRegex(ValueError, "backbone provenance"):
                _uniform_descriptor_provenance(provenance)

    def test_dinov2_evaluator_resolves_patch14_and_rejects_debias_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            dino_root = root / "descriptors"
            image_root.mkdir()
            dino_root.mkdir()
            pair_file = root / "pairs.csv"
            estimation_file = root / "estimation.csv"
            weights = root / "weights.pth"
            pair_file.write_text("synthetic", encoding="utf-8")
            estimation_file.write_text("synthetic", encoding="utf-8")
            weights.write_bytes(b"synthetic")
            common = {
                "angular_bin": "0-40",
                "pair_file": pair_file,
                "estimation_pair_file": estimation_file,
                "image_root": image_root,
                "dino_root": dino_root,
                "basis_root": None,
                "output_root": root / "output",
                "weights": weights,
                "model_name": DINOV2_MODEL_NAME,
                "layers": tuple(LAYER_SWEEP_LAYERS),
                "ranks": (0,),
            }
            layers, ranks, patch_size = _validate_options(
                DenseEvaluationOptions(**common)
            )
            self.assertEqual(layers, tuple(LAYER_SWEEP_LAYERS))
            self.assertEqual(ranks, (0,))
            self.assertEqual(patch_size, 14)
            with self.assertRaisesRegex(ValueError, "raw-only"):
                _validate_options(
                    DenseEvaluationOptions(**{**common, "ranks": (0, 100)})
                )
            with self.assertRaisesRegex(ValueError, "patch_size=14"):
                _validate_options(
                    DenseEvaluationOptions(**{**common, "patch_size": 16})
                )

    def test_split_audit_uses_all_canonical_estimation_bins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            estimation_dir = root / "estimation"
            estimation_dir.mkdir()
            header = (
                "image_1,image_2,angular_distance_degrees,image_1_height,"
                "image_1_width,image_2_height,image_2_width\n"
            )
            overlap = "test_object/wild_set/images/000.jpg"
            correspondence = root / "pairs_wildset_0-40.csv"
            correspondence.write_text(
                header
                + f"{overlap},test_object/wild_set/images/001.jpg,10,32,32,32,32\n",
                encoding="utf-8",
            )
            rows = {
                "0-40": (
                    "test_object/wild_set/images/010.jpg",
                    "test_object/wild_set/images/011.jpg",
                    10,
                ),
                "40-80": (overlap, "test_object/wild_set/images/012.jpg", 50),
                "80-120": (
                    "test_object/wild_set/images/013.jpg",
                    "test_object/wild_set/images/014.jpg",
                    90,
                ),
            }
            for angular_bin, (left, right, angle) in rows.items():
                (estimation_dir / f"pairs_wildset_{angular_bin}.csv").write_text(
                    header + f"{left},{right},{angle},32,32,32,32\n",
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(ValueError, "anywhere in the estimation"):
                audit_split_disjointness(
                    correspondence,
                    estimation_dir / "pairs_wildset_0-40.csv",
                    "0-40",
                )

    def test_rank_zero_identity_pair_writes_resumes_and_repairs_current_shard(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "navi"
            scene = image_root / "test_object" / "wild_set"
            for child in ("images", "depth", "masks"):
                (scene / child).mkdir(parents=True)
            camera = {
                "q": [1.0, 0.0, 0.0, 0.0],
                "t": [0.0, 0.0, 0.0],
                "focal_length": 1000.0,
            }
            annotations = []
            for index in range(4):
                annotations.append(
                    {
                        "filename": f"{index:03d}.jpg",
                        "image_size": [16, 1024],
                        "camera": camera,
                    }
                )
                if index < 2:
                    Image.new("RGB", (1024, 16), color=(index, 0, 0)).save(
                        scene / "images" / f"{index:03d}.jpg"
                    )
                    Image.fromarray(
                        np.full((16, 1024), 65535, dtype=np.uint16)
                    ).save(scene / "depth" / f"{index:03d}.png")
                    Image.fromarray(np.ones((16, 1024), dtype=np.uint8)).save(
                        scene / "masks" / f"{index:03d}.png"
                    )
            (scene / "annotations.json").write_text(
                json.dumps(annotations), encoding="utf-8"
            )
            header = (
                "image_1,image_2,angular_distance_degrees,image_1_height,"
                "image_1_width,image_2_height,image_2_width\n"
            )
            pairs = root / "pairs_wildset_0-40.csv"
            pairs.write_text(
                header
                + "test_object/wild_set/images/000.jpg,"
                "test_object/wild_set/images/001.jpg,0,16,1024,16,1024\n",
                encoding="utf-8",
            )
            estimation = root / "estimation_wildset_0-40.csv"
            estimation.write_text(
                header
                + "test_object/wild_set/images/002.jpg,"
                "test_object/wild_set/images/003.jpg,0,16,1024,16,1024\n",
                encoding="utf-8",
            )
            weights = root / "weights.pth"
            weights.write_bytes(b"synthetic-checkpoint")
            weights_id = checkpoint_identity(weights)
            metadata = dino_artifact_metadata(
                model_name="dinov3_vitl16",
                layer=16,
                weights_id=weights_id,
                long_edge=1024,
                downscale_only=False,
            )
            descriptors = np.eye(64, dtype=np.float32).reshape(1, 64, 64)
            dino_root = root / "dino"
            for index in range(2):
                save_dino_map(
                    dino_root
                    / "layer16"
                    / "test_object"
                    / "wild_set"
                    / "images"
                    / f"{index:03d}.dino.npz",
                    descriptors,
                    16,
                    (16, 1024),
                    (16, 1024),
                    metadata,
                )
            output_root = root / "rank_evaluation"
            options = DenseEvaluationOptions(
                angular_bin="0-40",
                pair_file=pairs,
                estimation_pair_file=estimation,
                image_root=image_root,
                dino_root=dino_root,
                basis_root=None,
                output_root=output_root,
                weights=weights,
                layers=(16,),
                ranks=(0,),
                max_k=4,
                device="cpu",
                summarize=False,
            )
            first = evaluate_dense_correspondence(options)
            self.assertEqual(first.written_shards, 1)
            shard = output_root / "bin_0-40/shards/layer16/pair_000001.npz"
            data = validate_shard(shard)
            np.testing.assert_array_equal(
                data["a_to_b_candidate_target_patch_index"][0, :, 0], np.arange(64)
            )
            np.testing.assert_allclose(
                data["a_to_b_candidate_error_m"][0, :, 0], 0.0, atol=1e-15
            )
            np.testing.assert_array_equal(
                data["a_to_b_candidate_mutual_entry_k"][0, :, 0], 1
            )
            resumed = evaluate_dense_correspondence(options)
            self.assertEqual(resumed.resumed_shards, 1)
            shard.write_bytes(b"corrupt")
            repaired = evaluate_dense_correspondence(options)
            self.assertEqual(repaired.recomputed_shards, 1)
            validate_shard(shard)


if __name__ == "__main__":
    unittest.main()
