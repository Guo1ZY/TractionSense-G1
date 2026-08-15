#!/usr/bin/env python3
"""Measure one physical foot's raw Hall notification rate in isolation."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time

import numpy as np

from dual_foot_bridge.bridge import _load_config
from dual_foot_bridge.protocol import CHAR_UUID, FrameError, FrameParser


async def measure(side: str, address: str, duration_s: float, characteristic: str) -> dict:
    try:
        from bleak import BleakClient
    except ImportError as error:
        raise RuntimeError("missing bleak") from error

    parser = FrameParser()
    arrivals: list[float] = []
    lengths: list[int] = []
    temperature_min = float("inf")
    temperature_max = float("-inf")
    magnetic_min = float("inf")
    magnetic_max = float("-inf")
    notifications = 0
    rejected = 0
    client = BleakClient(address, timeout=20.0)
    connect_started = time.monotonic()
    try:
        await client.connect()
        connect_s = time.monotonic() - connect_started

        def on_notify(_sender, data: bytearray) -> None:
            nonlocal notifications, rejected
            nonlocal temperature_min, temperature_max, magnetic_min, magnetic_max
            notifications += 1
            lengths.append(len(data))
            try:
                frame = parser.parse(bytes(data))
            except (FrameError, ValueError):
                rejected += 1
                return
            arrivals.append(frame.received_monotonic)
            temperature_min = min(temperature_min, float(np.min(frame.temperature_x10)) * 0.1)
            temperature_max = max(temperature_max, float(np.max(frame.temperature_x10)) * 0.1)
            magnetic_min = min(magnetic_min, float(np.min(frame.magnetic_xyz)))
            magnetic_max = max(magnetic_max, float(np.max(frame.magnetic_xyz)))

        await client.start_notify(characteristic, on_notify)
        started = time.monotonic()
        await asyncio.sleep(duration_s)
        elapsed = time.monotonic() - started
        dt = np.diff(np.asarray(arrivals, dtype=np.float64))
        return {
            "format": "g1-single-foot-hall-rate-test-v1",
            "measurement": "15 x Bx/By/Bz raw counts plus temperature; no force conversion",
            "side": side,
            "address": address,
            "connect_s": round(connect_s, 3),
            "window_s": round(elapsed, 3),
            "notifications": notifications,
            "valid_frames": len(arrivals),
            "rejected_frames": rejected,
            "payload_lengths": sorted(set(lengths)),
            "window_rate_hz": round(len(arrivals) / elapsed, 3),
            "timestamp_rate_hz": (
                round((len(arrivals) - 1) / (arrivals[-1] - arrivals[0]), 3)
                if len(arrivals) > 1
                else 0.0
            ),
            "dt_ms_p05_median_p95_max": (
                [round(float(value * 1000.0), 3) for value in np.quantile(dt, [0.05, 0.5, 0.95, 1.0])]
                if len(dt)
                else []
            ),
            "gaps_over_100ms": int(np.count_nonzero(dt > 0.1)),
            "gaps_over_200ms": int(np.count_nonzero(dt > 0.2)),
            "temperature_c_min_max": [round(temperature_min, 2), round(temperature_max, 2)],
            "magnetic_counts_min_max": [round(magnetic_min, 1), round(magnetic_max, 1)],
        }
    finally:
        if client.is_connected:
            try:
                await asyncio.wait_for(client.stop_notify(characteristic), timeout=3.0)
            except Exception:
                pass
            try:
                await asyncio.wait_for(client.disconnect(), timeout=5.0)
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.magnetic.json"))
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if not 2.0 <= args.duration <= 300.0:
            raise ValueError("--duration must be in [2, 300] seconds")
        config = _load_config(args.config)
        address = str(config[args.side].get("address", "")).strip()
        if not address:
            raise ValueError(f"{args.side}.address is empty")
        characteristic = str(config.get("ble", {}).get("characteristic_uuid", CHAR_UUID))
        result = asyncio.run(
            asyncio.wait_for(
                measure(args.side, address, args.duration, characteristic),
                timeout=args.duration + 30.0,
            )
        )
        text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        print(text, end="", flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_name(f".{args.output.name}.tmp")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(args.output)
        return 0
    except Exception as error:
        print(f"[ERROR] {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
