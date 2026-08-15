#!/usr/bin/env python3
"""Discover and verify uniquely named left/right FootSensor15 peripherals."""

from __future__ import annotations

import argparse
import asyncio


async def discover(timeout: float, expected_names: dict[str, str], show_all: bool) -> int:
    try:
        from bleak import BleakScanner
    except ImportError as error:
        print(f"[ERROR] missing bleak: {error}")
        return 2
    devices = await BleakScanner.discover(timeout=timeout, return_adv=True)
    matches = []
    for address, (device, advertisement) in devices.items():
        observed_name = (advertisement.local_name or device.name or "").strip()
        side = next(
            (
                candidate
                for candidate, name in expected_names.items()
                if observed_name.casefold() == name.casefold()
            ),
            "",
        )
        if show_all or side:
            matches.append((address, observed_name, advertisement.rssi, side))
    if not matches:
        print(
            "No device advertised as "
            f"{expected_names['left']!r} or {expected_names['right']!r} "
            f"was found in {timeout:.1f}s"
        )
        return 1
    print("ADDRESS                         RSSI  SIDE   NAME")
    for address, observed_name, rssi, side in sorted(matches):
        print(f"{address:<31} {rssi:>4}  {side or '-':<5}  {observed_name}")
    found = {side for *_rest, side in matches if side}
    missing = [side for side in ("left", "right") if side not in found]
    if missing:
        print(f"\nMissing unique advertising identity: {', '.join(missing)}")
        return 1
    print("\nBoth unique names were found; copy their addresses into config.magnetic.json.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--all", action="store_true", help="also print unrelated BLE devices")
    args = parser.parse_args()
    if not args.left_name.strip() or not args.right_name.strip():
        parser.error("left/right names must be non-empty")
    if args.left_name.casefold() == args.right_name.casefold():
        parser.error("left/right names must be different")
    return asyncio.run(
        discover(
            args.timeout,
            {"left": args.left_name.strip(), "right": args.right_name.strip()},
            args.all,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
