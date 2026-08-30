"""Cache-first SuperPoint access with a lazy external LightGlue adapter.

No LightGlue or SuperPoint source/weights are distributed by this repository.
Precomputed caches can be used without installing LightGlue. Computing missing
features requires users to install the upstream package separately and accept
its license.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .provenance import checkpoint_identity
from .resize import resize_array_long_edge
from .schemas import SuperPointFeatures, load_superpoint_cache, save_superpoint_cache


@dataclass(frozen=True)
class SuperPointConfig:
    max_num_keypoints: int = 2048
    long_edge: int = 1024
    downscale_only: bool = False
    weights: Path | None = None
    expected_weights_id: str | None = None

    def __post_init__(self) -> None:
        if self.max_num_keypoints < 0:
            raise ValueError("max_num_keypoints must be non-negative")
        if self.long_edge < 0:
            raise ValueError("long_edge must be non-negative")
        if self.expected_weights_id is not None and not self.expected_weights_id.strip():
            raise ValueError("expected_weights_id must be a non-empty string")

    @property
    def weights_id(self) -> str | None:
        if self.weights is None:
            return self.expected_weights_id
        actual = checkpoint_identity(self.weights)
        if self.expected_weights_id is not None and self.expected_weights_id != actual:
            raise ValueError(
                "Configured SuperPoint weights ID does not match the checkpoint contents: "
                f"{self.expected_weights_id!r} != {actual!r}"
            )
        return actual

    @property
    def cache_tag(self) -> str:
        return (
            f"lr{self.long_edge or 0}_kp{self.max_num_keypoints}_"
            f"du{int(self.downscale_only)}"
        )


class ExternalLightGlueSuperPoint:
    """Use the installed ``lightglue`` package only when extraction is requested."""

    def __init__(self, device: str | Any, config: SuperPointConfig) -> None:
        self.device_arg = device
        self.config = config
        self._torch = None
        self._extractor = None
        self._weights_id: str | None = None

    def _initialize(self) -> None:
        if self._extractor is not None:
            return
        try:
            import lightglue
            import torch
            try:
                from lightglue import SuperPoint
            except ImportError:
                from lightglue.superpoint import SuperPoint
        except ImportError as exc:  # pragma: no cover - depends on user environment
            raise RuntimeError(
                "Computing SuperPoint features requires an external LightGlue "
                "installation. See the repository setup instructions, or provide "
                "precomputed .spkp.npz caches."
            ) from exc
        device = torch.device(self.device_arg)
        kwargs: dict[str, Any] = {}
        if self.config.max_num_keypoints > 0:
            kwargs["max_num_keypoints"] = self.config.max_num_keypoints
        extractor = SuperPoint(**kwargs).eval().to(device)
        if self.config.weights is not None:
            if not self.config.weights.is_file():
                raise FileNotFoundError(f"SuperPoint weights not found: {self.config.weights}")
            checkpoint_path = self.config.weights.expanduser().resolve()
            state = torch.load(checkpoint_path, map_location=device)
            if isinstance(state, Mapping) and "model" in state:
                state = state["model"]
            if not isinstance(state, Mapping):
                raise ValueError(
                    f"SuperPoint checkpoint {checkpoint_path} must contain a state dict"
                )
            try:
                extractor.load_state_dict(state, strict=True)
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    f"Failed to load SuperPoint checkpoint {checkpoint_path}: {exc}"
                ) from exc
            weights_id = checkpoint_identity(checkpoint_path)
        else:
            try:
                package_version = distribution_version("lightglue")
            except PackageNotFoundError:  # editable/source-only LightGlue checkout
                package_version = str(getattr(lightglue, "__version__", "unknown"))
            state_hash = _state_dict_sha256(extractor.state_dict())
            weights_id = (
                f"lightglue:{package_version}:superpoint-default:sha256:{state_hash}"
            )
        if (
            self.config.expected_weights_id is not None
            and self.config.expected_weights_id != weights_id
        ):
            raise ValueError(
                "Installed/default SuperPoint identity does not match the configured "
                f"expected ID: {weights_id!r} != {self.config.expected_weights_id!r}"
            )
        self.device = device
        self._torch = torch
        self._extractor = extractor
        self._weights_id = weights_id

    def resolved_weights_id(self) -> str:
        self._initialize()
        assert self._weights_id is not None
        return self._weights_id

    def extract(self, image_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
        self._initialize()
        assert self._torch is not None and self._extractor is not None
        torch = self._torch
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        orig_hw = (int(rgb.shape[0]), int(rgb.shape[1]))
        rgb, _ = resize_array_long_edge(
            rgb,
            self.config.long_edge,
            downscale_only=self.config.downscale_only,
        )
        proc_hw = (int(rgb.shape[0]), int(rgb.shape[1]))
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)
            .to(device=self.device, dtype=torch.float32)
            / 255.0
        )
        with torch.inference_mode():
            features = self._extractor.extract(image_tensor, resize=None)
        keypoints = self._unbatch(features["keypoints"])
        descriptors = self._unbatch(features["descriptors"])
        score_key = "keypoint_scores" if "keypoint_scores" in features else "scores"
        scores = self._unbatch(features[score_key])
        keypoints_np = keypoints.detach().cpu().numpy().astype(np.float32, copy=False)
        descriptors_np = descriptors.detach().cpu().numpy().astype(np.float32, copy=False)
        scores_np = scores.detach().cpu().numpy().astype(np.float32, copy=False)
        if descriptors_np.ndim != 2:
            raise ValueError(f"External SuperPoint returned descriptors {descriptors_np.shape}")
        if descriptors_np.shape[0] != len(keypoints_np) and descriptors_np.shape[1] == len(keypoints_np):
            descriptors_np = descriptors_np.T
        if descriptors_np.shape[0] != len(keypoints_np) or len(scores_np) != len(keypoints_np):
            raise ValueError("External SuperPoint returned inconsistent feature counts")
        return keypoints_np, descriptors_np, scores_np, orig_hw, proc_hw

    @staticmethod
    def _unbatch(value: Any) -> Any:
        if getattr(value, "ndim", 0) >= 2 and value.shape[0] == 1:
            return value[0]
        return value


class CacheBackedSuperPoint:
    """Read validated caches and optionally compute missing entries."""

    def __init__(
        self,
        image_root: Path,
        cache_root: Path,
        config: SuperPointConfig,
        adapter: ExternalLightGlueSuperPoint | None = None,
        overwrite: bool = False,
        allow_legacy_cache: bool = False,
    ) -> None:
        self.image_root = Path(image_root).resolve()
        self.cache_root = Path(cache_root).resolve()
        self.config = config
        self.adapter = adapter
        self.overwrite = overwrite
        self.allow_legacy_cache = allow_legacy_cache
        declared_weights_id = config.weights_id
        if declared_weights_id is None:
            if adapter is None:
                raise ValueError(
                    "Cache-only SuperPoint use requires an expected weights identity. "
                    "Pass --superpoint-weights-id with the ID recorded during extraction."
                )
            declared_weights_id = adapter.resolved_weights_id()
        if declared_weights_id == "upstream-default" and not allow_legacy_cache:
            raise ValueError(
                "The legacy `upstream-default` SuperPoint identity is ambiguous. "
                "Regenerate the cache, or pass --allow-legacy-keypoint-cache after "
                "manually verifying the producer."
            )
        self.weights_id = declared_weights_id

    def cache_path(self, image_rel: Path) -> Path:
        image_path = self._image_path(image_rel)
        rel = image_path.relative_to(self.image_root)
        return self.cache_root / rel.parent / f"{rel.stem}.{self.config.cache_tag}.spkp.npz"

    def legacy_cache_path(self, image_rel: Path) -> Path:
        """Location used before resize policy was encoded in the cache filename."""
        image_path = self._image_path(image_rel)
        rel = image_path.relative_to(self.image_root)
        tag = f"lr{self.config.long_edge or 0}_kp{self.config.max_num_keypoints}"
        return self.cache_root / rel.parent / f"{rel.stem}.{tag}.spkp.npz"

    def load_or_extract(
        self,
        image_rel: Path,
        *,
        include_descriptors: bool = False,
    ) -> SuperPointFeatures:
        image_path = self._image_path(image_rel)
        cache_path = self.cache_path(image_rel)
        read_path = cache_path
        legacy_path = self.legacy_cache_path(image_rel)
        using_legacy = False
        if not self.overwrite and not read_path.is_file() and legacy_path.is_file():
            if not self.allow_legacy_cache:
                raise FileNotFoundError(
                    f"Legacy SuperPoint cache found at {legacy_path}, but its resize policy "
                    "cannot be inferred from the filename. Pass "
                    "--allow-legacy-keypoint-cache to opt in after verifying preprocessing."
                )
            read_path = legacy_path
            using_legacy = True
        expected = {
            "long_edge": self.config.long_edge,
            "max_num_keypoints": self.config.max_num_keypoints,
            "downscale_only": self.config.downscale_only,
            "weights_id": self.weights_id,
            "source_path": str(image_rel),
        }
        if read_path.is_file() and not self.overwrite:
            result = load_superpoint_cache(
                read_path,
                expected,
                allow_missing_metadata=using_legacy,
            )
            if not include_descriptors or result.descriptors is not None:
                return result
        if self.adapter is None:
            reason = "descriptors are absent" if read_path.is_file() else "cache is absent"
            raise FileNotFoundError(
                f"Cannot use {image_rel}: {reason} at {cache_path}, and external "
                "SuperPoint computation is disabled."
            )
        keypoints, descriptors, scores, orig_hw, proc_hw = self.adapter.extract(image_path)
        save_superpoint_cache(
            cache_path,
            keypoints,
            descriptors=descriptors if include_descriptors else None,
            scores=scores if include_descriptors else None,
            long_edge=self.config.long_edge,
            max_num_keypoints=self.config.max_num_keypoints,
            downscale_only=self.config.downscale_only,
            weights_id=self.weights_id,
            orig_hw=orig_hw,
            proc_hw=proc_hw,
            source_path=str(image_rel),
        )
        return load_superpoint_cache(cache_path, expected)

    def _image_path(self, image_rel: Path) -> Path:
        image_rel = Path(image_rel)
        if image_rel.is_absolute():
            raise ValueError(f"Pair-list image paths must be relative: {image_rel}")
        path = (self.image_root / image_rel).resolve()
        try:
            path.relative_to(self.image_root)
        except ValueError as exc:
            raise ValueError(f"Image path escapes image root: {image_rel}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
        return path


def _state_dict_sha256(state: Mapping[str, Any]) -> str:
    """Hash tensor names, dtypes, shapes, and bytes in a stable key order."""
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not hasattr(value, "detach"):
            raise ValueError(f"SuperPoint state entry {key!r} is not a tensor")
        array = value.detach().cpu().contiguous().numpy()
        for token in (key, str(array.dtype), repr(tuple(array.shape))):
            encoded = token.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        raw = array.tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()
