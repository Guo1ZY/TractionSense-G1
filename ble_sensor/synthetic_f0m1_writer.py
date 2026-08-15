#!/usr/bin/env python3
"""50 Hz synthetic F0M1 writer: valid zero-field dual-foot packet.

Pipeline smoke only.  Magnetic values are exact zeros, both feet valid, age 0,
period 0.02 s.  Later replace this with real F0R1 replay for the same socket.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from dual_foot_bridge.magnetic_ipc import F0M1Writer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("/tmp/g1_foot_rl_obs_synth.bin"),
    )
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument(
        "--valid",
        type=float,
        default=1.0,
        help="valid flag for both feet: 1.0 = sensor healthy (zero field), "
        "0.0 = sensor invalid (policy proprio-only fallback).",
    )
    args = parser.parse_args()
    if args.rate_hz <= 0.0:
        parser.error("--rate-hz must be positive")
    valid = float(max(0.0, min(1.0, args.valid)))
    writer = F0M1Writer(args.path.expanduser().resolve())
    zeros = np.zeros((15, 3), dtype=np.float32)
    period = 1.0 / args.rate_hz
    while True:
        start = time.perf_counter()
        writer.write(
            zeros,
            zeros,
            valid_left=valid,
            valid_right=valid,
            age_left_s=0.0,
            age_right_s=0.0,
            period_left_s=period,
            period_right_s=period,
        )
        elapsed = time.perf_counter() - start
        time.sleep(max(0.0, period - elapsed))


if __name__ == "__main__":
    raise SystemExit(main())
