"""Model-aware DINO layer extraction and positional-bias basis estimation."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .matching import load_basis_payload, require_torch, resolve_device
from .pairs import read_pairs, unique_images
from .provenance import checkpoint_identity
from .resize import resize_pil_long_edge, resized_hw_long_edge
from .schemas import (
    DINO_MODEL_PROVENANCE_KEYS,
    DINO_PROVENANCE_KEYS,
    load_dino_map,
    save_dino_map,
)


# Backward-compatible constant for callers that predate model-aware profiles.
PATCH_SIZE = 16
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINO_NORMALIZATION_ID = "rgb-imagenet-mean-std-v1"
DINO_RESIZE_ID = "opencv-inter-area-int-truncate-v1"
DINO_PADDING_ID = "bottom-right-zero-to-patch-grid-v1"
DINO_EXTRACTION_MANIFEST_FILENAME = "extraction_manifest.json"


@dataclass(frozen=True)
class BackboneProfile:
    """Static architecture and correction contract for one supported backbone."""

    model_name: str
    family: str
    patch_size: int
    depth: int
    descriptor_dim: int
    register_tokens: int
    correction_policy: str
    strict_checkpoint: bool = False


BACKBONE_PROFILES: dict[str, BackboneProfile] = {
    "dinov3_vitl16": BackboneProfile(
        "dinov3_vitl16", "dinov3", 16, 24, 1024, 4, "positional-debias"
    ),
    "dinov2_vitl14_reg": BackboneProfile(
        "dinov2_vitl14_reg",
        "dinov2",
        14,
        24,
        1024,
        4,
        "none",
        strict_checkpoint=True,
    ),
}
MODEL_LAYERS = {name: profile.depth for name, profile in BACKBONE_PROFILES.items()}
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PAIR_SIZE_COLUMNS = (
    "image_1_height",
    "image_1_width",
    "image_2_height",
    "image_2_width",
)


@dataclass(frozen=True)
class ExtractionOptions:
    input_root: Path
    output_root: Path
    # ``dinov3_source`` remains the third positional field for API compatibility.
    dinov3_source: Path | None = None
    weights: Path | None = None
    layers: tuple[int, ...] = (19,)
    model_name: str = "dinov3_vitl16"
    pair_file: Path | None = None
    image_list: Path | None = None
    batch_size: int = 4
    long_edge: int = 1024
    downscale_only: bool = False
    device: str = "auto"
    overwrite: bool = False
    compile_model: bool = False
    # Generic source name used by new DINOv2/v3 callers.
    source: Path | None = None

    def resolved_source(self) -> Path:
        return resolve_model_source(self.source, self.dinov3_source)

    def resolved_weights(self) -> Path:
        if self.weights is None:
            raise ValueError("DINO weights are required")
        return Path(self.weights)


def get_backbone_profile(model_name: str) -> BackboneProfile:
    try:
        return BACKBONE_PROFILES[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(BACKBONE_PROFILES))
        raise ValueError(
            f"Unsupported DINO model {model_name!r}; supported models: {supported}"
        ) from exc


def resolve_model_source(source: Path | None, legacy_source: Path | None) -> Path:
    """Resolve the generic source path while retaining the DINOv3 field alias."""
    if source is None and legacy_source is None:
        raise ValueError("DINO source checkout is required")
    if source is not None and legacy_source is not None:
        generic = Path(source).expanduser().resolve()
        legacy = Path(legacy_source).expanduser().resolve()
        if generic != legacy:
            raise ValueError(
                f"Conflicting DINO source paths: source={generic}, "
                f"dinov3_source={legacy}"
            )
        return generic
    return Path(source if source is not None else legacy_source).expanduser().resolve()


def validate_correction_ranks(
    model_name: str,
    ranks: Iterable[int],
    correction: str = "auto",
) -> tuple[str, tuple[int, ...]]:
    """Resolve correction mode and enforce the backbone-specific rank contract."""
    profile = get_backbone_profile(model_name)
    normalized_ranks = tuple(sorted(set(int(rank) for rank in ranks)))
    if not normalized_ranks or min(normalized_ranks) < 0:
        raise ValueError("correction ranks must contain non-negative integers")
    correction = correction.strip().lower().replace("_", "-")
    if correction not in {"auto", "none", "positional-debias"}:
        raise ValueError(
            "correction must be one of: auto, none, positional-debias"
        )
    if correction == "auto":
        correction = "positional-debias" if any(normalized_ranks) else "none"
    if correction == "none" and any(normalized_ranks):
        raise ValueError("correction='none' requires rank 0 only")
    if correction == "positional-debias" and not any(normalized_ranks):
        raise ValueError("correction='positional-debias' requires a positive rank")
    if profile.correction_policy == "none" and correction != "none":
        raise ValueError(
            f"{model_name} does not support positional-bias correction; use rank 0 "
            "with correction='none'"
        )
    return correction, normalized_ranks


def model_indices_for_layers(
    model_name: str, layers: Iterable[int]
) -> tuple[int, ...]:
    """Map public one-based block outputs to model zero-based block indices."""
    profile = get_backbone_profile(model_name)
    ordered = tuple(int(layer) for layer in layers)
    if not ordered or any(layer < 1 or layer > profile.depth for layer in ordered):
        raise ValueError(
            f"Layers must lie in 1..{profile.depth} for {model_name}"
        )
    return tuple(layer - 1 for layer in ordered)


def _validate_options(options: ExtractionOptions) -> ExtractionOptions:
    get_backbone_profile(options.model_name)
    model_indices_for_layers(options.model_name, options.layers)
    if options.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if options.long_edge < 0:
        raise ValueError("long_edge must be non-negative")
    if not Path(options.input_root).is_dir():
        raise NotADirectoryError(f"Input image root does not exist: {options.input_root}")
    source = options.resolved_source()
    weights = options.resolved_weights()
    if not source.is_dir():
        raise NotADirectoryError(f"DINO source checkout does not exist: {source}")
    if not weights.is_file():
        raise FileNotFoundError(f"DINO weights do not exist: {weights}")
    if options.pair_file is not None and options.image_list is not None:
        raise ValueError("Use at most one of pair_file and image_list")
    return options


def _resolve_image(token: Path, input_root: Path) -> Path:
    if token.is_absolute():
        path = token.resolve()
    else:
        path = (input_root / token).resolve()
    try:
        path.relative_to(input_root)
    except ValueError as exc:
        raise ValueError(f"Image path escapes input root: {token}") from exc
    if path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image suffix: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    return path


def select_images(options: ExtractionOptions) -> list[Path]:
    input_root = Path(options.input_root).resolve()
    if options.pair_file is not None:
        tokens = unique_images(read_pairs(options.pair_file))
    elif options.image_list is not None:
        list_path = Path(options.image_list)
        if not list_path.is_file():
            raise FileNotFoundError(f"Image list does not exist: {list_path}")
        tokens = [
            Path(line.strip())
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        return sorted(
            path.resolve()
            for path in input_root.rglob("*")
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES
        )
    return sorted({_resolve_image(token, input_root) for token in tokens})


def layer_output_roots(output_root: Path, layers: tuple[int, ...]) -> dict[int, Path]:
    """Use the exact requested root for one layer and named children for many."""
    if len(layers) == 1:
        return {layers[0]: Path(output_root)}
    return {layer: Path(output_root) / f"layer{layer}" for layer in layers}


def dino_artifact_metadata(
    *,
    model_name: str,
    layer: int,
    weights_id: str,
    long_edge: int,
    downscale_only: bool,
) -> dict[str, Any]:
    """Return the provenance fields shared by descriptor and basis artifacts."""
    return {
        "model_name": model_name,
        "layer": layer,
        "weights_id": weights_id,
        "long_edge": long_edge,
        "downscale_only": downscale_only,
        "normalization_id": DINO_NORMALIZATION_ID,
        "resize_id": DINO_RESIZE_ID,
        "padding_id": DINO_PADDING_ID,
    }


def backbone_provenance(
    model_name: str,
    *,
    correction: str | None = None,
) -> dict[str, Any]:
    """Return architecture/correction provenance for reports and new artifacts."""
    profile = get_backbone_profile(model_name)
    resolved_correction = profile.correction_policy if correction is None else correction
    return {
        "model_family": profile.family,
        "patch_size": profile.patch_size,
        "descriptor_dim": profile.descriptor_dim,
        "register_tokens": profile.register_tokens,
        "correction": resolved_correction,
    }


def source_checkout_provenance(source: Path | None) -> dict[str, Any]:
    """Identify a local model source checkout without invoking a shell."""
    unknown = {"source_revision": "unknown", "source_dirty": "unknown"}
    if source is None:
        return unknown
    source = Path(source).expanduser().resolve()
    try:
        revision = subprocess.run(
            ("git", "-C", str(source), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "-C", str(source), "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return unknown
    if not revision:
        return unknown
    return {"source_revision": revision, "source_dirty": bool(status.strip())}


def dino_cache_validation_metadata(
    metadata: Mapping[str, Any],
    *,
    patch_size: int | None = None,
    require_model_provenance: bool = False,
) -> dict[str, Any]:
    """Select metadata exposed by the current descriptor-cache reader.

    New caches store optional model-profile fields. Existing DINOv3 caches remain
    valid unless callers explicitly require those fields.
    """
    expected = {key: metadata[key] for key in DINO_PROVENANCE_KEYS}
    if require_model_provenance:
        expected.update(
            {key: metadata[key] for key in DINO_MODEL_PROVENANCE_KEYS}
        )
    if patch_size is not None:
        expected["patch_size"] = int(patch_size)
    return expected


def _checkpoint_state_dict(checkpoint: Any, path: Path) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            f"DINO checkpoint {path} must contain a state-dict mapping, got "
            f"{type(checkpoint).__name__}"
        )
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
        raise ValueError(f"DINO checkpoint {path} has no valid `model` state dict")
    return state


def _load_state_dict_checked(
    model: Any,
    state: Mapping[str, Any],
    path: Path,
    *,
    strict: bool = False,
) -> None:
    try:
        incompatible = model.load_state_dict(state, strict=strict)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Failed to load DINO checkpoint {path}: {exc}") from exc
    missing = list(getattr(incompatible, "missing_keys", ()))
    unexpected = list(getattr(incompatible, "unexpected_keys", ()))
    if missing or unexpected:
        sample_missing = ", ".join(missing[:5]) or "none"
        sample_unexpected = ", ".join(unexpected[:5]) or "none"
        raise RuntimeError(
            f"DINO checkpoint {path} is incompatible with the requested model: "
            f"{len(missing)} missing keys ({sample_missing}); {len(unexpected)} unexpected "
            f"keys ({sample_unexpected})"
        )


def _load_model(options: ExtractionOptions, device: Any):
    torch, _ = require_torch()
    profile = get_backbone_profile(options.model_name)
    model = torch.hub.load(
        repo_or_dir=str(options.resolved_source()),
        model=options.model_name,
        source="local",
        pretrained=False,
    )
    checkpoint_path = options.resolved_weights().expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = _checkpoint_state_dict(checkpoint, checkpoint_path)
    _load_state_dict_checked(
        model, state, checkpoint_path, strict=profile.strict_checkpoint
    )
    model.eval().to(device)
    if options.compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("compile_model was requested, but this PyTorch has no torch.compile")
        model = torch.compile(model)
    return model


def _image_tensor(
    path: Path,
    long_edge: int,
    downscale_only: bool,
    patch_size: int = PATCH_SIZE,
):
    torch, _ = require_torch()
    with Image.open(path) as image:
        image = resize_pil_long_edge(
            image.convert("RGB"), long_edge, downscale_only=downscale_only
        )
        width, height = image.size
        array = np.asarray(image, dtype=np.float32) / 255.0
    padded_height, padded_width = pad_hw_to_patch_grid(height, width, patch_size)
    padded = np.zeros((padded_height, padded_width, 3), dtype=np.float32)
    padded[:height, :width] = array
    tensor = torch.from_numpy(padded).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std, (height, width), (padded_height, padded_width)


def _output_path(path: Path, input_root: Path, output_root: Path) -> Path:
    return (output_root / path.relative_to(input_root)).with_suffix(".dino.npz")


def _write_extraction_manifests(
    output_roots: Mapping[int, Path],
    metadata_by_layer: Mapping[int, Mapping[str, Any]],
    *,
    patch_size: int,
) -> None:
    for layer, output_root in output_roots.items():
        path = Path(output_root) / DINO_EXTRACTION_MANIFEST_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "artifact": "dino-descriptor-extraction",
            **metadata_by_layer[layer],
            "patch_size": patch_size,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        try:
            os.replace(temp_path, path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise


def extract_intermediate_maps(
    model: Any,
    batch: Any,
    *,
    model_name: str,
    layers: Iterable[int],
) -> tuple[Any, ...]:
    """Extract patch maps, defensively stripping CLS/register tokens if needed."""
    profile = get_backbone_profile(model_name)
    layers = tuple(layers)
    outputs = tuple(
        model.get_intermediate_layers(
            batch,
            n=model_indices_for_layers(model_name, layers),
            reshape=True,
            norm=True,
        )
    )
    if len(outputs) != len(layers):
        raise RuntimeError(
            f"{model_name} returned {len(outputs)} layer outputs for {len(layers)} requests"
        )
    grid_h = int(batch.shape[-2]) // profile.patch_size
    grid_w = int(batch.shape[-1]) // profile.patch_size
    normalized: list[Any] = []
    for layer, output in zip(layers, outputs):
        # Official DINOv2/v3 implementations already strip special tokens when
        # reshape=True. Supporting token tensors makes the contract testable and
        # guards compatible third-party forks.
        if output.ndim == 3:
            token_count = int(output.shape[1])
            patch_tokens = grid_h * grid_w
            if token_count == patch_tokens + 1 + profile.register_tokens:
                output = output[:, 1 + profile.register_tokens :, :]
            elif token_count != patch_tokens:
                raise ValueError(
                    f"Unexpected token count {token_count} for {model_name} layer {layer}; "
                    f"expected {patch_tokens} patches plus CLS/{profile.register_tokens} registers"
                )
            output = output.reshape(output.shape[0], grid_h, grid_w, output.shape[2])
            output = output.permute(0, 3, 1, 2)
        if output.ndim != 4:
            raise ValueError(
                f"Unexpected {model_name} layer {layer} output shape {tuple(output.shape)}"
            )
        expected = (int(batch.shape[0]), profile.descriptor_dim, grid_h, grid_w)
        if tuple(output.shape) != expected:
            raise ValueError(
                f"Unexpected {model_name} layer {layer} output shape {tuple(output.shape)}; "
                f"expected {expected}"
            )
        normalized.append(output)
    return tuple(normalized)


def extract_dino(options: ExtractionOptions) -> int:
    """Extract the selected DINO layers and return the number of files written."""
    options = _validate_options(options)
    input_root = Path(options.input_root).resolve()
    profile = get_backbone_profile(options.model_name)
    layers = tuple(dict.fromkeys(options.layers))
    output_roots = layer_output_roots(options.output_root, layers)
    images = select_images(options)
    if not images:
        raise RuntimeError(f"No input images selected under {input_root}")
    weights_id = checkpoint_identity(options.resolved_weights())
    source_provenance = source_checkout_provenance(options.resolved_source())
    profile_provenance = backbone_provenance(options.model_name, correction="none")
    # patch_size has a dedicated, schema-reserved NPZ field.
    stored_profile_provenance = {
        key: value for key, value in profile_provenance.items() if key != "patch_size"
    }
    expected_by_layer = {
        layer: {
            **dino_artifact_metadata(
                model_name=options.model_name,
                layer=layer,
                weights_id=weights_id,
                long_edge=options.long_edge,
                downscale_only=options.downscale_only,
            ),
            **stored_profile_provenance,
            **source_provenance,
        }
        for layer in layers
    }
    pending: list[Path] = []
    for image_path in images:
        needs_extraction = options.overwrite
        if not options.overwrite:
            for layer, root in output_roots.items():
                output_path = _output_path(image_path, input_root, root)
                if not output_path.is_file():
                    needs_extraction = True
                    continue
                try:
                    load_dino_map(
                        output_path,
                        expected_metadata=dino_cache_validation_metadata(
                            expected_by_layer[layer],
                            patch_size=profile.patch_size,
                            require_model_provenance=profile.family == "dinov2",
                        ),
                    )
                except Exception as exc:
                    raise ValueError(
                        f"Existing DINO cache is incompatible with this extraction: "
                        f"{output_path}. Pass --overwrite to regenerate it. {exc}"
                    ) from exc
        if needs_extraction:
            pending.append(image_path)
    if not pending:
        # DINOv2 resume validation above requires the complete model/source
        # provenance on every cache, so repairing a missing/interrupted manifest
        # cannot misattribute legacy descriptors.
        if profile.family == "dinov2":
            _write_extraction_manifests(
                output_roots, expected_by_layer, patch_size=profile.patch_size
            )
        return 0

    torch, _ = require_torch()
    device = resolve_device(options.device)
    model = _load_model(options, device)
    written = 0
    for start in range(0, len(pending), options.batch_size):
        prepared = [
            (
                path,
                *_image_tensor(
                    path,
                    options.long_edge,
                    options.downscale_only,
                    profile.patch_size,
                ),
            )
            for path in pending[start : start + options.batch_size]
        ]
        size_groups: dict[tuple[int, int], list[tuple[Any, ...]]] = defaultdict(list)
        for sample in prepared:
            size_groups[sample[3]].append(sample)
        for proc_hw, group in size_groups.items():
            batch = torch.stack([sample[1] for sample in group]).to(device)
            with torch.inference_mode():
                features_by_layer = extract_intermediate_maps(
                    model,
                    batch,
                    model_name=options.model_name,
                    layers=sorted(layers),
                )
            for layer, features in zip(sorted(layers), features_by_layer):
                for sample, feature in zip(group, features):
                    image_path, _, orig_hw, _ = sample
                    output_path = _output_path(image_path, input_root, output_roots[layer])
                    if output_path.is_file() and not options.overwrite:
                        continue
                    descriptor_map = (
                        feature.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
                    )
                    save_dino_map(
                        output_path,
                        descriptor_map,
                        profile.patch_size,
                        proc_hw,
                        orig_hw,
                        expected_by_layer[layer],
                    )
                    written += 1
    _write_extraction_manifests(
        output_roots, expected_by_layer, patch_size=profile.patch_size
    )
    return written


def pad_hw_to_patch_grid(
    height: int, width: int, patch_size: int = PATCH_SIZE
) -> tuple[int, int]:
    if min(height, width, patch_size) <= 0:
        raise ValueError("Image dimensions and patch_size must be positive")
    return (
        (height + patch_size - 1) // patch_size * patch_size,
        (width + patch_size - 1) // patch_size * patch_size,
    )


def build_positional_basis(
    model: Any,
    *,
    image_height: int,
    image_width: int,
    layers: tuple[int, ...],
    max_components: int,
    device: Any,
    model_name: str = "dinov3_vitl16",
) -> dict[int, Any]:
    """Reproduce INSID3-style zero-image positional basis estimation."""
    profile = get_backbone_profile(model_name)
    if profile.correction_policy == "none":
        raise ValueError(
            f"{model_name} does not support positional-bias correction; no basis is needed"
        )
    model_indices_for_layers(model_name, layers)
    torch, functional = require_torch()
    padded_height, padded_width = pad_hw_to_patch_grid(
        image_height, image_width, profile.patch_size
    )
    zero = torch.zeros(1, 3, padded_height, padded_width, device=device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    zero = (zero - mean) / std
    with torch.inference_mode():
        feature_maps = model.get_intermediate_layers(
            zero,
            n=model_indices_for_layers(model_name, layers),
            reshape=True,
        )
    result: dict[int, Any] = {}
    for layer, feature_map in zip(layers, feature_maps):
        feature_map = functional.normalize(feature_map, p=2, dim=1)
        channels_first = feature_map.permute(1, 0, 2, 3).reshape(feature_map.shape[1], -1)
        channels_first = channels_first - channels_first.mean(dim=1, keepdim=True)
        left_vectors, _, _ = torch.linalg.svd(channels_first, full_matrices=False)
        if max_components > left_vectors.shape[1]:
            raise ValueError(
                f"Requested {max_components} basis vectors, maximum is {left_vectors.shape[1]}"
            )
        result[layer] = left_vectors[:, :max_components].contiguous()
    return result


def required_basis_sizes(
    pair_files: Iterable[Path],
    *,
    long_edge: int,
    downscale_only: bool,
    patch_size: int = PATCH_SIZE,
) -> tuple[tuple[int, int], ...]:
    """Return long-edge-resized, patch-padded ``(H, W)`` required by pair CSVs."""
    if long_edge < 0:
        raise ValueError("long_edge must be non-negative")
    sizes: set[tuple[int, int]] = set()
    image_sizes: dict[str, tuple[int, int]] = {}
    files = tuple(Path(path) for path in pair_files)
    if not files:
        raise ValueError("At least one pair-list CSV is required")
    for pair_file in files:
        if not pair_file.is_file():
            raise FileNotFoundError(pair_file)
        with pair_file.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            required = {"image_1", "image_2", *PAIR_SIZE_COLUMNS}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{pair_file}: missing CSV columns: {sorted(missing)}")
            for line_number, row in enumerate(reader, 2):
                for side in (1, 2):
                    image = row[f"image_{side}"].strip()
                    try:
                        height = int(row[f"image_{side}_height"])
                        width = int(row[f"image_{side}_width"])
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"{pair_file}:{line_number}: invalid image_{side} dimensions"
                        ) from exc
                    if not image or min(height, width) <= 0:
                        raise ValueError(
                            f"{pair_file}:{line_number}: invalid image_{side} path or dimensions"
                        )
                    original = height, width
                    previous = image_sizes.setdefault(image, original)
                    if previous != original:
                        raise ValueError(
                            f"Conflicting dimensions for {image}: {previous} versus {original}"
                        )
                    resized_height, resized_width, _ = resized_hw_long_edge(
                        height, width, long_edge, downscale_only=downscale_only
                    )
                    sizes.add(
                        pad_hw_to_patch_grid(
                            resized_height, resized_width, patch_size
                        )
                    )
    return tuple(sorted(sizes))


def _save_basis(
    *,
    output_path: Path,
    basis: Any,
    meta: dict[str, Any],
    torch: Any,
    save_json: bool,
) -> None:
    max_rank = int(basis.shape[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "basis": basis.cpu(),
            "max_rank": max_rank,
            "meta": meta,
        },
        output_path,
    )
    if save_json:
        output_path.with_suffix(".json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )


def _basis_metadata(
    *,
    model_name: str,
    layer: int,
    image_height: int,
    image_width: int,
    weights_id: str,
) -> dict[str, Any]:
    profile = get_backbone_profile(model_name)
    image_height, image_width = pad_hw_to_patch_grid(
        image_height, image_width, profile.patch_size
    )
    return {
        "model_name": model_name,
        "layer": layer,
        "layer_index": layer - 1,
        "image_height": image_height,
        "image_width": image_width,
        "padded_hw": (image_height, image_width),
        "patch_size": profile.patch_size,
        "weights_id": weights_id,
        "normalization_id": DINO_NORMALIZATION_ID,
        "procedure": (
            "pad_zero_image_to_patch_grid->normalize_zero_image->dino_layer->"
            "L2_channel_norm->center_channels->SVD(U[:, :k])"
        ),
    }


def fit_bias(
    *,
    dinov3_source: Path | None = None,
    weights: Path,
    output: Path,
    model_name: str,
    layers: tuple[int, ...],
    image_height: int,
    image_width: int,
    components: tuple[int, ...],
    device_arg: str,
    save_json: bool = False,
    source: Path | None = None,
) -> list[Path]:
    layers = tuple(sorted(set(layers)))
    if not layers:
        raise ValueError("At least one DINO layer is required")
    if not components or min(components) <= 0:
        raise ValueError("All basis component counts must be positive")
    validate_correction_ranks(model_name, components)
    resolved_source = resolve_model_source(source, dinov3_source)
    options = ExtractionOptions(
        input_root=Path("."),
        output_root=Path("."),
        source=resolved_source,
        weights=weights,
        model_name=model_name,
        layers=layers,
    )
    # Validate only the model and external dependencies; no image root is needed.
    model_indices_for_layers(model_name, layers)
    if not resolved_source.is_dir() or not Path(weights).is_file():
        raise FileNotFoundError("DINO source checkout and weights are required")
    profile = get_backbone_profile(model_name)
    image_height, image_width = pad_hw_to_patch_grid(
        image_height, image_width, profile.patch_size
    )
    torch, _ = require_torch()
    device = resolve_device(device_arg)
    model = _load_model(options, device)
    bases = build_positional_basis(
        model,
        image_height=image_height,
        image_width=image_width,
        layers=layers,
        max_components=max(components),
        device=device,
        model_name=model_name,
    )
    weights_id = checkpoint_identity(weights)
    outputs: list[Path] = []
    for layer, basis in bases.items():
        output_string = str(output)
        if len(layers) > 1 and "{layer}" not in output_string:
            raise ValueError("Multi-layer bias output must include `{layer}`")
        output_path = Path(output_string.format(layer=layer))
        meta = _basis_metadata(
            model_name=model_name,
            layer=layer,
            image_height=image_height,
            image_width=image_width,
            weights_id=weights_id,
        )
        _save_basis(
            output_path=output_path,
            basis=basis,
            meta=meta,
            torch=torch,
            save_json=save_json,
        )
        outputs.append(output_path)
    return outputs


def fit_bias_for_pair_files(
    *,
    pair_files: Iterable[Path],
    dinov3_source: Path | None = None,
    weights: Path,
    output_root: Path,
    filename_template: str,
    model_name: str,
    layers: tuple[int, ...],
    long_edge: int,
    downscale_only: bool,
    components: tuple[int, ...],
    device_arg: str,
    existing: str = "error",
    save_json: bool = False,
    source: Path | None = None,
) -> list[Path]:
    """Fit every positional basis required by dimension-annotated pair-list CSVs."""
    profile = get_backbone_profile(model_name)
    components = tuple(sorted(set(components)))
    if not components or min(components) <= 0:
        raise ValueError("All basis component counts must be positive")
    validate_correction_ranks(model_name, components)
    layers = tuple(sorted(set(layers)))
    if not layers:
        raise ValueError("At least one DINO layer is required")
    model_indices_for_layers(model_name, layers)
    if existing not in {"error", "skip", "overwrite"}:
        raise ValueError("existing must be one of: error, skip, overwrite")
    resolved_source = resolve_model_source(source, dinov3_source)
    if not resolved_source.is_dir() or not Path(weights).is_file():
        raise FileNotFoundError("DINO source checkout and weights are required")
    pair_files = tuple(Path(path) for path in pair_files)
    sizes = required_basis_sizes(
        pair_files,
        long_edge=long_edge,
        downscale_only=downscale_only,
        patch_size=profile.patch_size,
    )

    outputs_by_layer_size: dict[tuple[int, tuple[int, int]], Path] = {}
    for layer in layers:
        for height, width in sizes:
            try:
                filename = filename_template.format(height=height, width=width)
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    "basis filename template may use only `{height}` and `{width}`"
                ) from exc
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("basis filename template must stay under output_root/layerN")
            output_path = Path(output_root) / f"layer{layer}" / relative
            if output_path in outputs_by_layer_size.values():
                raise ValueError(
                    f"basis filename template is not unique by layer and size: {output_path}"
                )
            outputs_by_layer_size[(layer, (height, width))] = output_path

    conflicts = [path for path in outputs_by_layer_size.values() if path.exists()]
    if conflicts and existing == "error":
        raise FileExistsError(
            f"Basis already exists: {conflicts[0]}. Use --existing skip or overwrite."
        )
    requested_rank = max(components)
    weights_id: str | None = None
    if existing == "skip":
        weights_id = checkpoint_identity(weights)
        for (layer, (height, width)), path in outputs_by_layer_size.items():
            if not path.exists():
                continue
            expected_meta = _basis_metadata(
                model_name=model_name,
                layer=layer,
                image_height=height,
                image_width=width,
                weights_id=weights_id,
            )
            expected_meta.update(
                {"long_edge": long_edge, "downscale_only": downscale_only}
            )
            try:
                payload = load_basis_payload(
                    path, "cpu", expected_metadata=expected_meta
                )
            except Exception as exc:
                raise ValueError(
                    f"Existing basis is incompatible with this run: {path}. "
                    f"Use --existing overwrite to regenerate it. {exc}"
                ) from exc
            stored_rank = int(payload["max_rank"])
            if stored_rank < requested_rank:
                raise ValueError(
                    f"Existing basis {path} has max_rank={stored_rank}, but this run "
                    f"requires at least {requested_rank}. Use --existing overwrite "
                    "to regenerate it."
                )
    pending = {
        layer_size: path
        for layer_size, path in outputs_by_layer_size.items()
        if existing != "skip" or not path.exists()
    }
    if not pending:
        return []

    options = ExtractionOptions(
        input_root=Path("."),
        output_root=Path("."),
        source=resolved_source,
        weights=weights,
        model_name=model_name,
        layers=layers,
    )
    torch, _ = require_torch()
    device = resolve_device(device_arg)
    model = _load_model(options, device)
    if weights_id is None:
        weights_id = checkpoint_identity(weights)
    outputs_by_key: dict[tuple[int, tuple[int, int]], Path] = {}
    for height, width in sizes:
        pending_layers = tuple(
            layer for layer in layers if (layer, (height, width)) in pending
        )
        if not pending_layers:
            continue
        bases = build_positional_basis(
            model,
            image_height=height,
            image_width=width,
            layers=pending_layers,
            max_components=max(components),
            device=device,
            model_name=model_name,
        )
        for layer in pending_layers:
            output_path = pending[(layer, (height, width))]
            meta = _basis_metadata(
                model_name=model_name,
                layer=layer,
                image_height=height,
                image_width=width,
                weights_id=weights_id,
            )
            meta.update(
                {
                    "long_edge": long_edge,
                    "downscale_only": downscale_only,
                    "source_pair_files": [str(path) for path in pair_files],
                }
            )
            _save_basis(
                output_path=output_path,
                basis=bases[layer],
                meta=meta,
                torch=torch,
                save_json=save_json,
            )
            outputs_by_key[(layer, (height, width))] = output_path
    return [outputs_by_key[key] for key in sorted(outputs_by_key)]
