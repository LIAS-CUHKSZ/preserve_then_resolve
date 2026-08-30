"""File-oriented orchestration for progressive DINO many-to-many matching."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dino import (
    backbone_provenance,
    dino_artifact_metadata,
    get_backbone_profile,
    model_indices_for_layers,
    source_checkout_provenance,
    validate_correction_ranks,
)
from .matching import (
    ImageFeatures,
    basis_for_dim,
    build_association_rows,
    debias_descriptors,
    interpolate_descriptors,
    load_basis_payload,
    progressive_mutual_knn,
    require_torch,
    resolve_device,
    write_association_csv,
)
from .pairs import PairRecord, matching_filename, read_pairs
from .provenance import checkpoint_identity
from .schemas import load_dino_map
from .superpoint import (
    CacheBackedSuperPoint,
    ExternalLightGlueSuperPoint,
    SuperPointConfig,
)


@dataclass(frozen=True)
class MatchOptions:
    pair_file: Path
    image_root: Path
    dino_root: Path
    keypoint_cache_root: Path
    output_root: Path
    dino_weights: Path | None = None
    model_name: str = "dinov3_vitl16"
    layer: int = 19
    patch_size: int | None = None
    basis_root: Path | None = None
    basis_filename_template: str = "dinov3_vitl16_{height}x{width}_basis.pt"
    svd_components: tuple[int, ...] = (500,)
    max_ks: tuple[int, ...] = (5,)
    association_upperbound: int = 2048
    device: str = "auto"
    max_pairs: int | None = None
    existing: str = "overwrite"
    compute_missing_keypoints: bool = False
    keypoint_cache_overwrite: bool = False
    allow_legacy_keypoint_cache: bool = False
    superpoint: SuperPointConfig = SuperPointConfig()
    # Canonical model-agnostic names; dino_weights remains a compatibility alias.
    weights: Path | None = None
    source: Path | None = None
    correction: str = "auto"

    def resolved_weights(self) -> Path:
        if self.weights is None and self.dino_weights is None:
            raise ValueError(
                "DINO weights are required to validate descriptor provenance"
            )
        if self.weights is not None and self.dino_weights is not None:
            canonical = Path(self.weights).expanduser().resolve()
            legacy = Path(self.dino_weights).expanduser().resolve()
            if canonical != legacy:
                raise ValueError(
                    f"Conflicting DINO weights: weights={canonical}, "
                    f"dino_weights={legacy}"
                )
            return canonical
        return Path(
            self.weights if self.weights is not None else self.dino_weights
        ).expanduser().resolve()


@dataclass(frozen=True)
class MatchSummary:
    pair_count: int
    failure_count: int
    output_count: int
    failure_manifest: Path


ASSOCIATION_MANIFEST_FILENAME = "association_manifest.json"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pair_identity_sha256(pairs: list[PairRecord]) -> str:
    digest = hashlib.sha256()
    for pair in pairs:
        digest.update(
            f"{pair.pair_index}\t{pair.left_rel.as_posix()}\t"
            f"{pair.right_rel.as_posix()}\n".encode("utf-8")
        )
    return digest.hexdigest()


class _BasisCache:
    def __init__(self, root: Path, template: str, layer: int, device: Any) -> None:
        self.root = Path(root)
        self.template = template
        self.layer = layer
        self.device = device
        self._payloads: dict[tuple[Path, tuple[tuple[str, Any], ...]], dict[str, Any]] = {}

    def for_size(
        self, image_size: tuple[int, int], dino_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        width, height = image_size
        try:
            filename = self.template.format(width=width, height=height)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "basis_filename_template may use only `{width}` and `{height}`"
            ) from exc
        layered_path = self.root / f"layer{self.layer}" / filename
        legacy_path = self.root / filename
        path = layered_path if layered_path.is_file() else legacy_path
        expected = {
            "model_name": dino_metadata["model_name"],
            "layer": dino_metadata["layer"],
            "weights_id": dino_metadata["weights_id"],
            "normalization_id": dino_metadata["normalization_id"],
            "patch_size": dino_metadata["patch_size"],
            "image_height": height,
            "image_width": width,
        }
        cache_key = (path, tuple(expected.items()))
        if cache_key not in self._payloads:
            self._payloads[cache_key] = load_basis_payload(
                path, self.device, expected_metadata=expected
            )
        return self._payloads[cache_key]


def _descriptor_path(root: Path, image_rel: Path, model_family: str) -> Path:
    if image_rel.is_absolute() or ".." in image_rel.parts:
        raise ValueError(f"Image paths must stay relative to dataset root: {image_rel}")
    if model_family not in {"dinov2", "dinov3"}:
        raise ValueError(f"Unsupported descriptor family: {model_family}")
    return (Path(root) / image_rel).with_suffix(".dino.npz")


def _load_features(
    image_rel: Path,
    dino_root: Path,
    superpoint_cache: CacheBackedSuperPoint,
    device: Any,
    expected_dino_metadata: dict[str, Any],
    expected_descriptor_dim: int,
    model_family: str,
) -> ImageFeatures:
    torch, functional = require_torch()
    dino = load_dino_map(
        _descriptor_path(dino_root, image_rel, model_family),
        expected_metadata=expected_dino_metadata,
    )
    if dino.descriptor_map.shape[2] != expected_descriptor_dim:
        raise ValueError(
            f"DINO descriptor dimension for {image_rel} is "
            f"{dino.descriptor_map.shape[2]}, expected {expected_descriptor_dim}"
        )
    local = superpoint_cache.load_or_extract(image_rel)
    cached_hw = local.metadata.get("proc_hw")
    if cached_hw is not None and dino.has_orig_hw and tuple(cached_hw) != tuple(dino.orig_hw):
        raise ValueError(
            f"Preprocessing mismatch for {image_rel}: SuperPoint processed size "
            f"{tuple(cached_hw)} but DINO unpadded size is {dino.orig_hw}. Use the "
            "same long-edge and downscale-only policy for both stages."
        )
    keypoints, descriptors_np = interpolate_descriptors(
        local.keypoints, dino.descriptor_map, dino.patch_size
    )
    descriptors = torch.from_numpy(descriptors_np).to(device=device, dtype=torch.float32)
    if descriptors.numel():
        descriptors = functional.normalize(descriptors, p=2, dim=1)
    return ImageFeatures(
        keypoints=keypoints,
        descriptors=descriptors,
        image_size=(dino.proc_hw[1], dino.proc_hw[0]),
        metadata=dino.metadata,
    )


def _output_path(options: MatchOptions, dim: int, max_k: int, pair: PairRecord, width: int) -> Path:
    return (
        Path(options.output_root)
        / f"debias_svd{dim}"
        / f"progressive_k{max_k}"
        / matching_filename(pair.pair_index, width)
    )


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _write_failure_manifest(path: Path, failures: list[dict[str, Any]]) -> None:
    _write_json_atomic(path, failures)


def run_matching(options: MatchOptions) -> MatchSummary:
    correction, dims = validate_correction_ranks(
        options.model_name, options.svd_components, options.correction
    )
    profile = get_backbone_profile(options.model_name)
    model_indices_for_layers(options.model_name, (options.layer,))
    patch_size = profile.patch_size if options.patch_size is None else options.patch_size
    if patch_size != profile.patch_size:
        raise ValueError(
            f"patch_size={patch_size} does not match {options.model_name} "
            f"patch size {profile.patch_size}"
        )
    max_ks = tuple(sorted(set(options.max_ks)))
    if not max_ks or min(max_ks) <= 0:
        raise ValueError("max_ks must contain positive integers")
    if options.association_upperbound < 0:
        raise ValueError("association_upperbound must be non-negative")
    if options.existing not in {"overwrite", "skip"}:
        raise ValueError("existing must be `overwrite` or `skip`")
    if correction == "positional-debias" and options.basis_root is None:
        raise ValueError("basis_root is required when positional debiasing is enabled")
    if options.source is not None and not Path(options.source).is_dir():
        raise NotADirectoryError(
            f"DINO source checkout is not a directory: {options.source}"
        )
    source_provenance = source_checkout_provenance(options.source)
    weights = options.resolved_weights()
    weights_id = checkpoint_identity(weights)
    expected_dino_metadata = dino_artifact_metadata(
        model_name=options.model_name,
        layer=options.layer,
        weights_id=weights_id,
        long_edge=options.superpoint.long_edge,
        downscale_only=options.superpoint.downscale_only,
    )
    expected_dino_metadata["patch_size"] = patch_size
    if profile.family == "dinov2":
        descriptor_provenance = backbone_provenance(
            options.model_name, correction="none"
        )
        expected_dino_metadata.update(
            {
                key: value
                for key, value in descriptor_provenance.items()
                if key != "patch_size"
            }
        )
        if options.source is not None:
            expected_dino_metadata.update(source_provenance)
    for label, path in (
        ("image_root", options.image_root),
        ("dino_root", options.dino_root),
        ("keypoint_cache_root", options.keypoint_cache_root),
    ):
        if label == "keypoint_cache_root" and options.compute_missing_keypoints:
            Path(path).mkdir(parents=True, exist_ok=True)
        elif not Path(path).is_dir():
            raise NotADirectoryError(f"{label} is not a directory: {path}")

    device = resolve_device(options.device)
    adapter = (
        ExternalLightGlueSuperPoint(device, options.superpoint)
        if options.compute_missing_keypoints
        else None
    )
    superpoint_cache = CacheBackedSuperPoint(
        options.image_root,
        options.keypoint_cache_root,
        options.superpoint,
        adapter,
        overwrite=options.keypoint_cache_overwrite,
        allow_legacy_cache=options.allow_legacy_keypoint_cache,
    )
    basis_cache = (
        _BasisCache(
            options.basis_root, options.basis_filename_template, options.layer, device
        )
        if options.basis_root is not None and correction == "positional-debias"
        else None
    )
    pairs = read_pairs(options.pair_file, options.max_pairs)
    width = max(3, len(str(max((pair.pair_index for pair in pairs), default=0))))
    pair_identity = _pair_identity_sha256(pairs)
    pair_file_identity = _file_sha256(options.pair_file)
    variant_manifests: dict[tuple[int, int], tuple[Path, dict[str, Any]]] = {}
    expected_filenames = {
        matching_filename(pair.pair_index, width) for pair in pairs
    }
    for dim in dims:
        for max_k in max_ks:
            variant_dir = (
                Path(options.output_root)
                / f"debias_svd{dim}"
                / f"progressive_k{max_k}"
            )
            manifest_path = variant_dir / ASSOCIATION_MANIFEST_FILENAME
            payload: dict[str, Any] = {
                "schema_version": 1,
                "pair_file": Path(options.pair_file).name,
                "pair_file_sha256": pair_file_identity,
                "pair_identity_sha256": pair_identity,
                "pair_count": len(pairs),
                "model_name": options.model_name,
                "weights_id": weights_id,
                "layer": options.layer,
                "debias_rank": dim,
                "progressive_max_k": max_k,
                "association_upperbound": options.association_upperbound,
                **backbone_provenance(
                    options.model_name,
                    correction="none" if dim == 0 else correction,
                ),
                **source_provenance,
            }
            existing_matches = {
                path.name for path in variant_dir.glob("matching_*.csv")
            }
            extras = sorted(existing_matches - expected_filenames)
            if extras:
                raise ValueError(
                    f"Unexpected association files in {variant_dir}: {extras[:5]}. "
                    "Use a fresh split/version output root."
                )
            if options.existing == "skip" and (existing_matches or manifest_path.exists()):
                if not manifest_path.is_file():
                    raise FileNotFoundError(
                        f"Cannot resume unbound associations without {manifest_path}"
                    )
                existing_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                if existing_payload != payload:
                    raise ValueError(
                        f"Association manifest does not match the requested pair/config: "
                        f"{manifest_path}"
                    )
            variant_manifests[(dim, max_k)] = (manifest_path, payload)
    for manifest_path, payload in variant_manifests.values():
        _write_json_atomic(manifest_path, payload)

    output_root = Path(options.output_root)
    failure_manifest = output_root / "failures.json"
    retry_outputs: set[Path] = set()
    if options.existing == "skip" and failure_manifest.is_file():
        retained_failures = json.loads(failure_manifest.read_text(encoding="utf-8"))
        if not isinstance(retained_failures, list):
            raise ValueError(f"{failure_manifest}: expected a JSON list")
        for failure in retained_failures:
            incomplete = (
                failure.get("incomplete_outputs")
                if isinstance(failure, dict)
                else None
            )
            if not isinstance(incomplete, list) or not all(
                isinstance(value, str) for value in incomplete
            ):
                raise RuntimeError(
                    f"{failure_manifest}: legacy failure record has no exact "
                    "incomplete_outputs list; refusing to mistake old empty "
                    "placeholders for successful associations"
                )
            for value in incomplete:
                relative = Path(value)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(
                        f"{failure_manifest}: unsafe incomplete output path {value!r}"
                    )
                retry_outputs.add(output_root / relative)

    work_items: list[tuple[PairRecord, dict[tuple[int, int], Path]]] = []
    remaining_uses: Counter[Path] = Counter()
    for pair in pairs:
        outputs = {
            (dim, max_k): _output_path(options, dim, max_k, pair, width)
            for dim in dims
            for max_k in max_ks
        }
        pending = {
            key: path
            for key, path in outputs.items()
            if options.existing != "skip"
            or not path.is_file()
            or path in retry_outputs
        }
        retry_outputs.difference_update(pending.values())
        if not pending:
            continue
        work_items.append((pair, pending))
        # A self-pair needs one cached feature object, not two.  Count uses per
        # pending pair so a feature can be released immediately after its last
        # possible consumer.
        remaining_uses.update(dict.fromkeys((pair.left_rel, pair.right_rel)).keys())

    if retry_outputs:
        unexpected = sorted(str(path) for path in retry_outputs)
        raise ValueError(
            f"{failure_manifest}: incomplete outputs are outside the requested "
            f"pair/config variants: {unexpected[:5]}"
        )

    feature_cache: dict[Path, ImageFeatures] = {}
    feature_errors: dict[Path, Exception] = {}
    failures: list[dict[str, Any]] = []
    output_count = 0
    for pair, pending in work_items:
        completed: set[Path] = set()
        left: ImageFeatures | None = None
        right: ImageFeatures | None = None
        left_desc: Any = None
        right_desc: Any = None
        try:
            for image_rel in (pair.left_rel, pair.right_rel):
                if image_rel not in feature_cache:
                    if image_rel in feature_errors:
                        previous = feature_errors[image_rel]
                        raise RuntimeError(
                            f"Feature loading previously failed for {image_rel}: "
                            f"{previous}"
                        ) from previous
                    try:
                        feature_cache[image_rel] = _load_features(
                            image_rel,
                            options.dino_root,
                            superpoint_cache,
                            device,
                            expected_dino_metadata,
                            profile.descriptor_dim,
                            profile.family,
                        )
                    except Exception as exc:
                        # Deterministic cache/provenance failures should not
                        # repeatedly reload the same image for every pair.
                        feature_errors[image_rel] = exc
                        raise
            left, right = feature_cache[pair.left_rel], feature_cache[pair.right_rel]
            if left.descriptors.shape[1] != right.descriptors.shape[1]:
                raise ValueError("Left and right DINO descriptor dimensions do not agree")

            for dim in dims:
                if not any((dim, max_k) in pending for max_k in max_ks):
                    continue
                if dim == 0:
                    left_desc, right_desc = left.descriptors, right.descriptors
                else:
                    assert basis_cache is not None
                    descriptor_dim = left.descriptors.shape[1]
                    left_basis = basis_for_dim(
                        basis_cache.for_size(left.image_size, left.metadata),
                        dim,
                        descriptor_dim,
                    )
                    right_basis = basis_for_dim(
                        basis_cache.for_size(right.image_size, right.metadata),
                        dim,
                        descriptor_dim,
                    )
                    left_desc = debias_descriptors(left.descriptors, left_basis)
                    right_desc = debias_descriptors(right.descriptors, right_basis)
                for max_k in max_ks:
                    output = pending.get((dim, max_k))
                    if output is None:
                        continue
                    left_idx, right_idx, scores, first_ks = progressive_mutual_knn(
                        left_desc,
                        right_desc,
                        max_k,
                        options.association_upperbound,
                    )
                    rows = build_association_rows(
                        left, right, left_idx, right_idx, scores, first_ks
                    )
                    write_association_csv(output, rows)
                    completed.add(output)
                    output_count += 1
        except Exception as exc:
            failures.append(
                {
                    "pair_index": pair.pair_index,
                    "left": str(pair.left_rel),
                    "right": str(pair.right_rel),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "incomplete_outputs": sorted(
                        str(path.relative_to(options.output_root))
                        for path in pending.values()
                        if path not in completed
                    ),
                }
            )
        finally:
            # Drop local tensor references before evicting the cache entries.
            # The CUDA allocator can then reuse this memory for later images.
            left = right = None
            left_desc = right_desc = None
            for image_rel in dict.fromkeys((pair.left_rel, pair.right_rel)):
                remaining_uses[image_rel] -= 1
                if remaining_uses[image_rel] == 0:
                    del remaining_uses[image_rel]
                    feature_cache.pop(image_rel, None)
                    feature_errors.pop(image_rel, None)

    _write_failure_manifest(failure_manifest, failures)
    return MatchSummary(len(pairs), len(failures), output_count, failure_manifest)
