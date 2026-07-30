#!/usr/bin/env python3
"""Convert G1 deploy OBS1 logs into labeled 480-D traction data.

The label is the known test-floor class supplied by the experimenter.  It is
not a measured friction coefficient.  Only the common 480-D proprioceptive
prefix is retained; every plantar/Hall suffix is discarded by construction.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


OBS1_MAGIC = 0x3153424F
PROPRIO_DIM = 480
# Term-major v0 observation: gyro[15], gravity[15], command[15], ...
COMMAND_SLICE = slice(30, 45)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label 480-D G1 OBS1 data")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--floor", choices=("low", "high"), required=True)
    parser.add_argument(
        "--class-mu",
        type=float,
        help="Class encoding only; default 0.15 for low and 1.20 for high",
    )
    parser.add_argument("--command", type=float, default=0.20)
    parser.add_argument("--skip-seconds", type=float, default=2.0)
    parser.add_argument("--min-command", type=float, default=0.12)
    return parser.parse_args()


def read_obs1(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = path.read_bytes()
    observations: list[np.ndarray] = []
    stamps: list[float] = []
    offset = 0
    while offset + 16 <= len(payload):
        magic, dim, stamp_ns = struct.unpack_from("<IIQ", payload, offset)
        offset += 16
        byte_count = int(dim) * 4
        if (
            magic != OBS1_MAGIC
            or dim < PROPRIO_DIM
            or offset + byte_count > len(payload)
        ):
            raise ValueError(f"{path}: invalid OBS1 record at byte {offset - 16}")
        values = np.frombuffer(
            payload, dtype="<f4", count=dim, offset=offset
        ).copy()
        offset += byte_count
        observations.append(values[:PROPRIO_DIM])
        stamps.append(stamp_ns * 1.0e-9)
    if not observations:
        raise ValueError(f"{path}: no OBS1 records")
    return np.stack(observations).astype(np.float32), np.asarray(stamps)


def main() -> int:
    args = parse_args()
    class_mu = args.class_mu
    if class_mu is None:
        class_mu = 0.15 if args.floor == "low" else 1.20

    kept_obs: list[np.ndarray] = []
    kept_stamps: list[np.ndarray] = []
    sources: list[dict] = []
    for path in args.input:
        observation, stamps = read_obs1(path)
        elapsed = stamps - stamps[0]
        moving = np.max(np.abs(observation[:, COMMAND_SLICE]), axis=1)
        keep = (
            (elapsed >= args.skip_seconds)
            & (moving >= args.min_command)
            & np.isfinite(observation).all(axis=1)
        )
        kept_obs.append(observation[keep])
        kept_stamps.append(stamps[keep])
        sources.append(
            {
                "path": str(path.resolve()),
                "records": int(len(observation)),
                "kept": int(np.sum(keep)),
            }
        )

    obs = np.concatenate(kept_obs)
    wall_time = np.concatenate(kept_stamps)
    if len(obs) < 50:
        raise RuntimeError(
            f"only {len(obs)} walking frames remain; collect a longer trial"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        obs=obs,
        mu=np.full(len(obs), class_mu, dtype=np.float32),
        cmd_vx=np.full(len(obs), args.command, dtype=np.float32),
        wall_time=wall_time,
        floor_class=np.asarray(args.floor),
    )
    metadata = {
        "output": str(args.output.resolve()),
        "floor_class": args.floor,
        "mu_field": (
            f"{class_mu:.2f} class encoding; not a measured coefficient"
        ),
        "input_dim": PROPRIO_DIM,
        "plantar_hall_channels": "discarded",
        "samples": int(len(obs)),
        "sources": sources,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
