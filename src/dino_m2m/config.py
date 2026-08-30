"""Configuration loading with deterministic, file-relative path handling."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


_PATH_KEYS = {
    "dino_source",
    "dino_weights",
    "dinov2_source",
    "dinov2_weights",
    "dinov3_source",
    "dinov3_weights",
    "superpoint_weights",
    "image_root",
    "pair_file",
    "dino_root",
    "basis_root",
    "keypoint_cache_root",
    "association_root",
    "correspondence_root",
    "estimation_root",
}


def _yaml_module():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise RuntimeError(
            "YAML configuration requires PyYAML. Install the project with "
            "`pip install -e .` or pass all paths on the command line."
        ) from exc
    return yaml


def _resolve_paths(value: Any, base_dir: Path, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_paths(v, base_dir, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(v, base_dir, key) for v in value]
    if key in _PATH_KEYS and isinstance(value, str):
        path = Path(value).expanduser()
        return path if path.is_absolute() else (base_dir / path).resolve()
    return value


def load_config(path: Path) -> dict[str, Any]:
    """Load one YAML mapping; configured paths are relative to the YAML file."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    data = _yaml_module().safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return _resolve_paths(data, path.parent)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge nested mappings while replacing scalar and list values."""
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_configs(paths: list[Path] | tuple[Path, ...]) -> dict[str, Any]:
    """Load YAML files in order; values in later files take precedence."""
    result: dict[str, Any] = {}
    for path in paths:
        result = deep_merge(result, load_config(path))
    return result


def get_value(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read ``section.key`` without forcing callers to know missing sections."""
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def apply_dataset_override(config: Mapping[str, Any], dataset: str | None) -> dict[str, Any]:
    """Apply a named paper-profile override (for example, METU resize policy)."""
    result = deepcopy(dict(config))
    if dataset is None:
        return result
    overrides = config.get("dataset_overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("`dataset_overrides` must be a mapping")
    if dataset not in overrides:
        available = ", ".join(sorted(str(name) for name in overrides)) or "none"
        raise ValueError(
            f"Unknown dataset override {dataset!r}; available overrides: {available}"
        )
    override = overrides[dataset]
    if not isinstance(override, Mapping):
        raise ValueError(f"Dataset override for {dataset!r} must be a mapping")
    return deep_merge(result, override)
