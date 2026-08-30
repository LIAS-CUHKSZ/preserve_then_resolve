#!/usr/bin/env python3
"""Fail when a generated figure PDF contains non-Type-1 or unembedded fonts.

With no arguments, checks every PDF below ``outputs/paper_figures``. Explicit
PDF paths may be supplied to audit a different set.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def font_rows(pdf_path: Path) -> list[tuple[str, str, str]]:
    result = subprocess.run(
        ["pdffonts", str(pdf_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines()[2:]:
        if not line.strip():
            continue
        rows.append((line[:36].strip(), line[37:54].strip(), line[72:75].strip()))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, nargs="*")
    args = parser.parse_args()
    paths = args.pdf or sorted((REPO_ROOT / "outputs" / "paper_figures").rglob("*.pdf"))
    if not paths:
        raise SystemExit("No PDF figures found")

    failures: list[str] = []
    font_count = 0
    for path in paths:
        resolved = path if path.is_absolute() else Path.cwd() / path
        if not resolved.is_file():
            failures.append(f"missing: {resolved}")
            continue
        rows = font_rows(resolved)
        if not rows:
            failures.append(f"no font records: {resolved}")
            continue
        font_count += len(rows)
        for name, font_type, embedded in rows:
            if font_type != "Type 1" or embedded != "yes":
                failures.append(
                    f"{resolved}: {name}: type={font_type}, embedded={embedded}"
                )

    if failures:
        raise SystemExit("Type 1 font audit failed:\n  " + "\n  ".join(failures))
    print(f"PASS: {len(paths)} PDFs, {font_count} embedded Type 1 font records")


if __name__ == "__main__":
    main()
