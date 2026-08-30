from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from dino_m2m.dino import (
    BACKBONE_PROFILES,
    DINO_NORMALIZATION_ID,
    DINO_PADDING_ID,
    DINO_RESIZE_ID,
    ExtractionOptions,
    _basis_metadata,
    _load_state_dict_checked,
    _save_basis,
    backbone_provenance,
    extract_intermediate_maps,
    extract_dino,
    fit_bias_for_pair_files,
    model_indices_for_layers,
    pad_hw_to_patch_grid,
    required_basis_sizes,
    source_checkout_provenance,
    validate_correction_ranks,
)
from dino_m2m.provenance import checkpoint_identity
from dino_m2m.schemas import load_dino_map, save_dino_map


class DinoProvenanceTests(unittest.TestCase):
    def test_dinov2_profile_and_one_based_layer_mapping(self) -> None:
        profile = BACKBONE_PROFILES["dinov2_vitl14_reg"]
        self.assertEqual(
            (
                profile.family,
                profile.patch_size,
                profile.depth,
                profile.descriptor_dim,
                profile.register_tokens,
            ),
            ("dinov2", 14, 24, 1024, 4),
        )
        self.assertEqual(
            model_indices_for_layers("dinov2_vitl14_reg", (16, 24)),
            (15, 23),
        )
        self.assertEqual(pad_hw_to_patch_grid(29, 43, 14), (42, 56))

    def test_dinov2_rank_zero_is_the_only_correction_mode(self) -> None:
        self.assertEqual(
            validate_correction_ranks("dinov2_vitl14_reg", (0,), "auto"),
            ("none", (0,)),
        )
        with self.assertRaisesRegex(ValueError, "does not support"):
            validate_correction_ranks("dinov2_vitl14_reg", (200,), "auto")
        self.assertEqual(
            validate_correction_ranks("dinov3_vitl16", (0, 200), "auto"),
            ("positional-debias", (0, 200)),
        )

    def test_backbone_provenance_reports_no_dinov2_correction(self) -> None:
        self.assertEqual(
            backbone_provenance("dinov2_vitl14_reg", correction="none"),
            {
                "model_family": "dinov2",
                "patch_size": 14,
                "descriptor_dim": 1024,
                "register_tokens": 4,
                "correction": "none",
            },
        )

    def test_source_revision_is_unknown_for_non_git_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_meta = source_checkout_provenance(root)
            self.assertEqual(
                source_meta,
                {"source_revision": "unknown", "source_dirty": "unknown"},
            )
            path = root / "descriptor.dino.npz"
            metadata = {
                "model_name": "dinov2_vitl14_reg",
                "layer": 24,
                "weights_id": "sha256:test",
                "long_edge": 1024,
                "downscale_only": False,
                "normalization_id": DINO_NORMALIZATION_ID,
                "resize_id": DINO_RESIZE_ID,
                "padding_id": DINO_PADDING_ID,
                "model_family": "dinov2",
                "descriptor_dim": 1024,
                "register_tokens": 4,
                "correction": "none",
                **source_meta,
            }
            save_dino_map(
                path,
                np.zeros((2, 3, 1024), np.float32),
                14,
                (28, 42),
                (28, 42),
                metadata,
            )
            loaded = load_dino_map(path, expected_metadata=metadata)
            self.assertEqual(loaded.metadata["source_dirty"], "unknown")

    def test_intermediate_token_fallback_strips_cls_and_registers(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed in the lightweight test environment")

        class FakeModel:
            kwargs = None

            def get_intermediate_layers(self, batch, **kwargs):
                self.kwargs = kwargs
                tokens = torch.arange(11 * 1024, dtype=torch.float32).reshape(
                    1, 11, 1024
                )
                return (tokens, tokens + 1)

        model = FakeModel()
        batch = torch.zeros(1, 3, 28, 42)
        layer16, layer24 = extract_intermediate_maps(
            model,
            batch,
            model_name="dinov2_vitl14_reg",
            layers=(16, 24),
        )
        self.assertEqual(model.kwargs, {"n": (15, 23), "reshape": True, "norm": True})
        self.assertEqual(tuple(layer16.shape), (1, 1024, 2, 3))
        self.assertEqual(tuple(layer24.shape), (1, 1024, 2, 3))
        self.assertEqual(float(layer16[0, 0, 0, 0]), float(5 * 1024))

    def test_compact_basis_payload_stores_only_max_rank_matrix(self) -> None:
        class FakeBasis:
            shape = (1024, 500)

            def cpu(self):
                return self

        class FakeTorch:
            payload = None

            @classmethod
            def save(cls, payload, path):
                cls.payload = payload

        with tempfile.TemporaryDirectory() as directory:
            _save_basis(
                output_path=Path(directory) / "basis.pt",
                basis=FakeBasis(),
                meta={},
                torch=FakeTorch,
                save_json=False,
            )
        self.assertEqual(set(FakeTorch.payload), {"basis", "max_rank", "meta"})
        self.assertEqual(FakeTorch.payload["max_rank"], 500)

    @staticmethod
    def _dimension_csv(path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "image_1",
                    "image_2",
                    "angular_distance_degrees",
                    "image_1_height",
                    "image_1_width",
                    "image_2_height",
                    "image_2_width",
                )
            )
            writer.writerow(("landscape.jpg", "portrait.jpg", 20, 3024, 4032, 4032, 3024))
            writer.writerow(("other.jpg", "landscape.jpg", 30, 3072, 4080, 3024, 4032))

    def test_required_basis_sizes_follow_resize_and_padding_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pair_file = Path(directory) / "pairs_test.csv"
            self._dimension_csv(pair_file)
            self.assertEqual(
                required_basis_sizes(
                    (pair_file,), long_edge=1024, downscale_only=False
                ),
                ((768, 1024), (784, 1024), (1024, 768)),
            )

    def test_basis_metadata_uses_patch_padded_dimensions(self) -> None:
        metadata = _basis_metadata(
            model_name="dinov3_vitl16",
            layer=19,
            image_height=1024,
            image_width=771,
            weights_id="weights-id",
        )
        self.assertEqual(metadata["image_height"], 1024)
        self.assertEqual(metadata["image_width"], 784)
        self.assertEqual(metadata["padded_hw"], (1024, 784))

    def test_fit_bias_for_pairs_loads_model_once_for_all_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair_file = root / "pairs_test.csv"
            self._dimension_csv(pair_file)
            source = root / "dinov3"
            source.mkdir()
            weights = root / "weights.pth"
            weights.write_bytes(b"checkpoint")
            output_root = root / "basis"
            fake_basis = object()
            with (
                patch("dino_m2m.dino.require_torch", return_value=(object(), None)),
                patch("dino_m2m.dino.resolve_device", return_value="cpu"),
                patch("dino_m2m.dino._load_model", return_value=object()) as load_model,
                patch(
                    "dino_m2m.dino.build_positional_basis",
                    return_value={19: fake_basis},
                ) as build_basis,
                patch("dino_m2m.dino.checkpoint_identity", return_value="weights-id"),
                patch("dino_m2m.dino._save_basis") as save_basis,
            ):
                outputs = fit_bias_for_pair_files(
                    pair_files=(pair_file,),
                    dinov3_source=source,
                    weights=weights,
                    output_root=output_root,
                    filename_template="basis_{height}x{width}.pt",
                    model_name="dinov3_vitl16",
                    layers=(19,),
                    long_edge=1024,
                    downscale_only=False,
                    components=(200,),
                    device_arg="cpu",
                )
            self.assertEqual(load_model.call_count, 1)
            self.assertEqual(build_basis.call_count, 3)
            self.assertEqual(save_basis.call_count, 3)
            self.assertEqual(
                outputs,
                [
                    output_root / "layer19" / "basis_768x1024.pt",
                    output_root / "layer19" / "basis_784x1024.pt",
                    output_root / "layer19" / "basis_1024x768.pt",
                ],
            )

    def test_fit_bias_for_pairs_batches_multiple_layers_and_uses_layer_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair_file = root / "pairs_test.csv"
            self._dimension_csv(pair_file)
            source = root / "dinov3"
            source.mkdir()
            weights = root / "weights.pth"
            weights.write_bytes(b"checkpoint")
            output_root = root / "basis"

            def fake_build(_model, **kwargs):
                return {layer: object() for layer in kwargs["layers"]}

            with (
                patch("dino_m2m.dino.require_torch", return_value=(object(), None)),
                patch("dino_m2m.dino.resolve_device", return_value="cpu"),
                patch("dino_m2m.dino._load_model", return_value=object()) as load_model,
                patch(
                    "dino_m2m.dino.build_positional_basis", side_effect=fake_build
                ) as build_basis,
                patch("dino_m2m.dino.checkpoint_identity", return_value="weights-id"),
                patch("dino_m2m.dino._save_basis") as save_basis,
            ):
                outputs = fit_bias_for_pair_files(
                    pair_files=(pair_file,),
                    dinov3_source=source,
                    weights=weights,
                    output_root=output_root,
                    filename_template="basis_{height}x{width}.pt",
                    model_name="dinov3_vitl16",
                    layers=(18, 19),
                    long_edge=1024,
                    downscale_only=False,
                    components=(400,),
                    device_arg="cpu",
                )
            self.assertEqual(load_model.call_count, 1)
            self.assertEqual(build_basis.call_count, 3)
            self.assertEqual(save_basis.call_count, 6)
            self.assertTrue(
                all(
                    call.kwargs["layers"] == (18, 19)
                    for call in build_basis.call_args_list
                )
            )
            self.assertEqual(
                outputs,
                [
                    output_root / f"layer{layer}" / f"basis_{height}x{width}.pt"
                    for layer in (18, 19)
                    for height, width in ((768, 1024), (784, 1024), (1024, 768))
                ],
            )

    def test_skip_validates_existing_basis_rank_and_metadata(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed in the lightweight test environment")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair_file = root / "pairs_test.csv"
            self._dimension_csv(pair_file)
            source = root / "dinov3"
            source.mkdir()
            weights = root / "weights.pth"
            weights.write_bytes(b"checkpoint")
            output_root = root / "basis"
            (output_root / "layer19").mkdir(parents=True)
            weights_id = checkpoint_identity(weights)
            paths_and_meta = []
            for height, width in required_basis_sizes(
                (pair_file,), long_edge=1024, downscale_only=False
            ):
                path = output_root / "layer19" / f"basis_{height}x{width}.pt"
                meta = _basis_metadata(
                    model_name="dinov3_vitl16",
                    layer=19,
                    image_height=height,
                    image_width=width,
                    weights_id=weights_id,
                )
                meta.update({"long_edge": 1024, "downscale_only": False})
                torch.save(
                    {"basis": torch.eye(2), "max_rank": 2, "meta": meta}, path
                )
                paths_and_meta.append((path, meta))

            with patch("dino_m2m.dino._load_model") as load_model:
                outputs = fit_bias_for_pair_files(
                    pair_files=(pair_file,),
                    dinov3_source=source,
                    weights=weights,
                    output_root=output_root,
                    filename_template="basis_{height}x{width}.pt",
                    model_name="dinov3_vitl16",
                    layers=(19,),
                    long_edge=1024,
                    downscale_only=False,
                    components=(1, 2),
                    device_arg="cpu",
                    existing="skip",
                )
            self.assertEqual(outputs, [])
            load_model.assert_not_called()

            insufficient_path, meta = paths_and_meta[0]
            torch.save(
                {"basis": torch.ones((2, 1)), "max_rank": 1, "meta": meta},
                insufficient_path,
            )
            with self.assertRaisesRegex(ValueError, "requires at least 2"):
                fit_bias_for_pair_files(
                    pair_files=(pair_file,),
                    dinov3_source=source,
                    weights=weights,
                    output_root=output_root,
                    filename_template="basis_{height}x{width}.pt",
                    model_name="dinov3_vitl16",
                    layers=(19,),
                    long_edge=1024,
                    downscale_only=False,
                    components=(2,),
                    device_arg="cpu",
                    existing="skip",
                )

    def test_checkpoint_identity_depends_on_contents_not_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a" / "weights.pth"
            second = root / "b" / "weights.pth"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            self.assertNotEqual(checkpoint_identity(first), checkpoint_identity(second))

    def test_incompatible_checkpoint_keys_are_reported(self) -> None:
        class FakeModel:
            def load_state_dict(self, state, strict=False):
                self.state = state
                self.strict = strict
                return SimpleNamespace(
                    missing_keys=["blocks.0.weight"], unexpected_keys=["wrong.weight"]
                )

        with self.assertRaisesRegex(RuntimeError, "1 missing keys"):
            _load_state_dict_checked(FakeModel(), {"x": object()}, Path("weights.pth"))

    def test_extract_dino_does_not_skip_corrupt_existing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            output_root = root / "features"
            source_root = root / "dinov3"
            image_root.mkdir()
            output_root.mkdir()
            source_root.mkdir()
            Image.new("RGB", (24, 24)).save(image_root / "frame.jpg")
            weights = root / "weights.pth"
            weights.write_bytes(b"checkpoint")
            output = output_root / "frame.dino.npz"
            np.savez(
                output,
                schema_version=np.int32(2),
                descriptor_map=np.ones((2, 2), np.float32),
                patch_size=np.int32(16),
                proc_hw=np.array([32, 32], np.int32),
                orig_hw=np.array([24, 24], np.int32),
                model_name=np.str_("dinov3_vitl16"),
                layer=np.int32(19),
                weights_id=np.str_(checkpoint_identity(weights)),
                long_edge=np.int32(1024),
                downscale_only=np.bool_(False),
                normalization_id=np.str_(DINO_NORMALIZATION_ID),
                resize_id=np.str_(DINO_RESIZE_ID),
                padding_id=np.str_(DINO_PADDING_ID),
            )
            options = ExtractionOptions(
                input_root=image_root,
                output_root=output_root,
                dinov3_source=source_root,
                weights=weights,
            )
            with self.assertRaisesRegex(ValueError, "descriptor_map"):
                extract_dino(options)


if __name__ == "__main__":
    unittest.main()
