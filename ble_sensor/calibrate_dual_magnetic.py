#!/usr/bin/env python3
"""Build left/right Hall baseline, temperature and scale normalization files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibrate_magnetic import fit_normalization_document, write_normalization_document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("normalization"))
    parser.add_argument("--target-range", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = json.loads((args.session / "manifest.json").read_text(encoding="utf-8"))
        complete = [item for item in manifest.get("phases", []) if item.get("status") == "complete"]
        baseline = [args.session / item["csv"] for item in complete if item.get("group") == "baseline"]
        motion = [args.session / item["csv"] for item in complete if item.get("group") == "motion"]
        if not baseline or not motion:
            raise ValueError("session needs at least one complete baseline and one motion phase")
        summary = {
            "format": "g1-dual-foot-magnetic-normalization-summary-v1",
            "measurement_boundary": "Hall Bx/By/Bz and temperature; no force conversion",
            "session": str(args.session.resolve()),
            "feet": {},
        }
        for side in ("left", "right"):
            document = fit_normalization_document(
                baseline, motion, side, args.target_range
            )
            output = args.output / f"{side}.json"
            write_normalization_document(document, output)
            summary["feet"][side] = {
                "output": str(output.resolve()),
                "samples": document["samples"],
                "diagnostics": document["diagnostics"],
            }
        summary_path = args.output / "summary.json"
        write_normalization_document(summary, summary_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"[ERROR] {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
