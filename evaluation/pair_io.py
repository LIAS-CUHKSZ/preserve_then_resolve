"""Expose the pipeline's canonical pair-list parser to evaluation modules."""

from __future__ import annotations

try:
    from dino_m2m.pairs import PairRecord, matching_filename, read_pairs
except ModuleNotFoundError as exc:  # Support running tests from a source checkout.
    if exc.name != "dino_m2m":
        raise
    from src.dino_m2m.pairs import PairRecord, matching_filename, read_pairs

__all__ = ["PairRecord", "matching_filename", "read_pairs"]
