"""Pair-list parsing shared by extraction, matching, and evaluation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class PairRecord:
    pair_index: int
    left_rel: Path
    right_rel: Path


def _integer_token(token: str) -> int | None:
    try:
        return int(token.strip(), 10)
    except ValueError:
        return None


def read_pairs(pair_file: Path, max_pairs: int | None = None) -> list[PairRecord]:
    """Read whitespace pair rows or a CSV with ``image_1,image_2`` columns.

    Blank lines and lines beginning with ``#`` are ignored. Explicit and
    automatically assigned IDs share one namespace and must be unique.
    """
    pair_file = Path(pair_file)
    if not pair_file.is_file():
        raise FileNotFoundError(f"Pair file does not exist: {pair_file}")
    if max_pairs is not None and max_pairs < 0:
        raise ValueError("max_pairs must be non-negative or None")
    if max_pairs == 0:
        return []

    records: list[PairRecord] = []
    used_indices: set[int] = set()
    next_auto_index = 1
    raw_lines = pair_file.read_text(encoding="utf-8").splitlines()
    if pair_file.suffix.lower() == ".csv":
        parsed_rows = (
            (line_no, [field.strip() for field in fields])
            for line_no, fields in enumerate(csv.reader(raw_lines), 1)
        )
    else:
        parsed_rows = (
            (line_no, raw.split("\t") if "\t" in raw else raw.split())
            for line_no, raw in enumerate(raw_lines, 1)
        )

    for line_no, fields in parsed_rows:
        if not fields or not any(fields) or fields[0].startswith("#"):
            continue
        normalized = [field.lower() for field in fields]
        if normalized[:2] == ["image_1", "image_2"]:
            continue
        if normalized[:3] == ["pair_index", "image_1", "image_2"]:
            continue
        if len(fields) < 2:
            raise ValueError(
                f"Invalid pair row {line_no} in {pair_file}: expected at least two fields"
            )

        explicit_index = _integer_token(fields[0]) if len(fields) >= 3 else None
        if explicit_index is None:
            while next_auto_index in used_indices:
                next_auto_index += 1
            pair_index = next_auto_index
            left_token, right_token = fields[0], fields[1]
            next_auto_index += 1
        else:
            if explicit_index < 0:
                raise ValueError(f"Negative pair index on row {line_no}: {explicit_index}")
            pair_index = explicit_index
            left_token, right_token = fields[1], fields[2]

        if pair_index in used_indices:
            raise ValueError(f"Duplicate pair index {pair_index} on row {line_no} in {pair_file}")
        used_indices.add(pair_index)
        records.append(PairRecord(pair_index, Path(left_token), Path(right_token)))
        if max_pairs is not None and len(records) >= max_pairs:
            break

    records.sort(key=lambda record: record.pair_index)
    return records


def unique_images(records: list[PairRecord]) -> list[Path]:
    return sorted(
        {path for record in records for path in (record.left_rel, record.right_rel)},
        key=str,
    )


def matching_filename(pair_index: int, width: int = 3) -> str:
    if pair_index < 0:
        raise ValueError("pair_index must be non-negative")
    if width <= 0:
        raise ValueError("width must be positive")
    return f"matching_{pair_index:0{width}d}.csv"
