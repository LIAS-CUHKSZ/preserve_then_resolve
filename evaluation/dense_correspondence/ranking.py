"""Bidirectional cosine top-K and mutual-entry ranks.

The dense diagnostic evaluates the complete, non-padding DINO patch grids.
Recall queries are selected later with the object/depth mask, but mutual edge
counts deliberately use the complete graph to match the mask-free pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dino_m2m.matching import require_torch


@dataclass(frozen=True)
class BidirectionalTopK:
    """Cosine candidates and their first mutual-KNN entry rank."""

    a_to_b_indices: np.ndarray
    a_to_b_cosine: np.ndarray
    a_to_b_mutual_entry_k: np.ndarray
    b_to_a_indices: np.ndarray
    b_to_a_cosine: np.ndarray
    b_to_a_mutual_entry_k: np.ndarray
    mutual_match_count_at_k: np.ndarray


def _canonical_topk(oriented_similarity: Any, max_k: int) -> tuple[Any, Any]:
    """Return cosine-descending top-K with destination-index tie-breaking.

    ``torch.topk`` is used for the ordinary case. Rows whose K/K+1 boundary
    is tied fall back to a stable full ordering, so exact cosine ties do not
    make the raw shard backend-dependent.
    """
    torch, _ = require_torch()
    if oriented_similarity.ndim != 2:
        raise ValueError("Similarity must be a matrix")
    row_count, candidate_count = (int(value) for value in oriented_similarity.shape)
    if row_count == 0 or candidate_count < max_k:
        raise ValueError(
            f"Every query needs at least max_k={max_k} candidates; "
            f"got shape {tuple(oriented_similarity.shape)}"
        )
    probe_k = min(max_k + 1, candidate_count)
    values, indices = torch.topk(
        oriented_similarity, k=probe_k, dim=1, largest=True, sorted=False
    )

    # First order by target index, then stably by descending cosine. Stable
    # sorting preserves the index order for equal scores.
    index_order = torch.argsort(indices, dim=1, stable=True)
    indices = torch.gather(indices, 1, index_order)
    values = torch.gather(values, 1, index_order)
    score_order = torch.argsort(values, dim=1, descending=True, stable=True)
    indices = torch.gather(indices, 1, score_order)
    values = torch.gather(values, 1, score_order)

    if probe_k > max_k:
        boundary_ties = values[:, max_k - 1] == values[:, max_k]
        for row in torch.nonzero(boundary_ties, as_tuple=False).flatten().tolist():
            full_order = torch.argsort(
                oriented_similarity[row], descending=True, stable=True
            )[:max_k]
            indices[row, :max_k] = full_order
            values[row, :max_k] = oriented_similarity[row, full_order]
    return indices[:, :max_k], values[:, :max_k]


def mutual_entry_ranks(
    forward_indices: np.ndarray,
    reverse_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return first mutual K for every forward edge and cumulative counts.

    Arrays contain zero-based candidate positions. The returned entry ranks
    are one-based; ``K+1`` is the sentinel for an edge that is not mutual by
    the stored maximum K.
    """
    forward = np.asarray(forward_indices, dtype=np.int64)
    reverse = np.asarray(reverse_indices, dtype=np.int64)
    if forward.ndim != 2 or reverse.ndim != 2:
        raise ValueError("Forward and reverse candidate arrays must be matrices")
    source_count, max_k = forward.shape
    destination_count, reverse_k = reverse.shape
    if max_k <= 0 or reverse_k != max_k:
        raise ValueError("Forward and reverse candidate arrays need the same positive K")
    if np.any(forward < 0) or np.any(forward >= destination_count):
        raise IndexError("Forward candidate index is out of range")
    if np.any(reverse < 0) or np.any(reverse >= source_count):
        raise IndexError("Reverse candidate index is out of range")
    if np.any(np.diff(np.sort(forward, axis=1), axis=1) == 0) or np.any(
        np.diff(np.sort(reverse, axis=1), axis=1) == 0
    ):
        raise ValueError("Top-K candidate indices must be unique per query")

    sentinel = max_k + 1
    reverse_rank = np.full(
        (destination_count, source_count), sentinel, dtype=np.int16
    )
    rows = np.repeat(np.arange(destination_count, dtype=np.int64), max_k)
    reverse_rank[rows, reverse.reshape(-1)] = np.tile(
        np.arange(1, max_k + 1, dtype=np.int16), destination_count
    )
    source_rows = np.arange(source_count, dtype=np.int64)[:, None]
    backward = reverse_rank[forward, source_rows]
    forward_rank = np.arange(1, max_k + 1, dtype=np.int16)[None, :]
    entry = np.maximum(backward, forward_rank)
    entry[backward == sentinel] = sentinel
    counts = np.asarray(
        [np.count_nonzero(entry <= k) for k in range(1, max_k + 1)],
        dtype=np.int64,
    )
    return entry, counts


def compute_bidirectional_topk(
    descriptors_a: Any,
    descriptors_b: Any,
    max_k: int,
) -> BidirectionalTopK:
    """Compute both cosine top-K lists from one complete similarity matrix."""
    if max_k <= 0:
        raise ValueError("max_k must be positive")
    if descriptors_a.ndim != 2 or descriptors_b.ndim != 2:
        raise ValueError("Descriptor inputs must be matrices")
    if int(descriptors_a.shape[1]) != int(descriptors_b.shape[1]):
        raise ValueError("Descriptor channel dimensions disagree")
    torch, _ = require_torch()
    with torch.inference_mode():
        similarity = descriptors_a @ descriptors_b.T
    return compute_bidirectional_topk_from_similarity(similarity, max_k)


def compute_bidirectional_topk_from_similarity(
    similarity: Any,
    max_k: int,
) -> BidirectionalTopK:
    """Compute both top-K lists from one precomputed cosine matrix."""
    if max_k <= 0:
        raise ValueError("max_k must be positive")
    if similarity.ndim != 2:
        raise ValueError("Similarity must be a matrix")
    torch, _ = require_torch()
    with torch.inference_mode():
        a_indices_t, a_values_t = _canonical_topk(similarity, max_k)
        b_indices_t, b_values_t = _canonical_topk(similarity.T, max_k)
    a_indices = a_indices_t.detach().cpu().numpy().astype(np.int64, copy=False)
    b_indices = b_indices_t.detach().cpu().numpy().astype(np.int64, copy=False)
    a_cosine = a_values_t.detach().cpu().numpy().astype(np.float32, copy=False)
    b_cosine = b_values_t.detach().cpu().numpy().astype(np.float32, copy=False)
    a_entry, counts = mutual_entry_ranks(a_indices, b_indices)
    b_entry, reverse_counts = mutual_entry_ranks(b_indices, a_indices)
    if not np.array_equal(counts, reverse_counts):
        raise AssertionError("Bidirectional mutual edge counts disagree")
    return BidirectionalTopK(
        a_to_b_indices=a_indices,
        a_to_b_cosine=a_cosine,
        a_to_b_mutual_entry_k=a_entry,
        b_to_a_indices=b_indices,
        b_to_a_cosine=b_cosine,
        b_to_a_mutual_entry_k=b_entry,
        mutual_match_count_at_k=counts,
    )
