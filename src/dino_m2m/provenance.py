"""Small, dependency-free helpers for artifact provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path


def checkpoint_identity(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a content-addressed identifier for a model checkpoint."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
