#!/usr/bin/env python3
"""Merge the 512-env and 1536-env headline CSVs into one deduplicated curve."""

from __future__ import annotations

import csv
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    phase1 = root / "curves" / "phase1_512env" / "headline.csv"
    phase2 = root / "curves" / "phase2_1536env" / "headline.csv"
    rows = {}
    for path in (phase1, phase2):
        with path.open() as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows[int(row["iteration"])] = row
    with phase1.open() as handle:
        fieldnames = csv.DictReader(handle).fieldnames
    out = root / "curves" / "headline_merged.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for iteration in sorted(rows):
            writer.writerow(rows[iteration])
    print(f"wrote {out} ({len(rows)} iterations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
