"""Image resizing shared by DINO and SuperPoint preprocessing."""

from __future__ import annotations

import numpy as np
from PIL import Image


def resized_hw_long_edge(
    height: int,
    width: int,
    max_size: int,
    *,
    downscale_only: bool = False,
) -> tuple[int, int, float]:
    """Return the exact output ``(height, width, scale)`` for long-edge resize."""
    if min(height, width) <= 0:
        raise ValueError("Image dimensions must be positive")
    if max_size <= 0 or (downscale_only and max(height, width) <= max_size):
        return height, width, 1.0
    scale = max_size / max(height, width)
    new_width, new_height = int(width * scale), int(height * scale)
    if min(new_width, new_height) <= 0:
        raise ValueError(f"Resize target is empty: {new_width}x{new_height}")
    return new_height, new_width, scale


def resize_array_long_edge(
    image: np.ndarray,
    max_size: int,
    *,
    downscale_only: bool = False,
) -> tuple[np.ndarray, float]:
    """Apply the experiment's OpenCV ``INTER_AREA`` long-edge resize policy."""
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected HxW or HxWxC image, got {image.shape}")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")
    new_height, new_width, scale = resized_hw_long_edge(
        height, width, max_size, downscale_only=downscale_only
    )
    if (new_height, new_width) == (height, width):
        return image, scale
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise RuntimeError(
            "OpenCV is required to reproduce the experiment resize exactly. "
            "Install the `vision` dependencies."
        ) from exc
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA), scale


def resize_pil_long_edge(
    image: Image.Image,
    max_size: int,
    *,
    downscale_only: bool = False,
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    resized, _ = resize_array_long_edge(rgb, max_size, downscale_only=downscale_only)
    return Image.fromarray(resized)
