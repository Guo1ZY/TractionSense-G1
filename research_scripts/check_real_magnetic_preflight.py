#!/usr/bin/env python3
"""Read-only preflight for dual-foot magnetic G1 harness testing."""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import time
from pathlib import Path


F0M1 = 0x46304D31
PACKET_SIZE = 400
FINAL_SLOT = "config/policy/velocity/traction_magnetic_speedboost112_guard"
PROFILE = (
    0.70, 0.76, 0.70,
    0.76, 0.82, 0.76,
    0.82, 0.88, 0.82,
    0.88, 0.94, 0.88,
    0.94, 1.00, 0.94,
)
REPO_ROOT = Path(
    os.environ.get("TRACTIONSENSE_ROOT", Path(__file__).resolve().parents[1])
).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            os.environ.get(
                "G1_MAGNETIC_CONFIG",
                Path.home() / ".config/tractionsense-g1/magnetic.json",
            )
        ),
    )
    parser.add_argument(
        "--health",
        type=Path,
        default=Path("/tmp/g1_foot_magnetic_health.json"),
    )
    parser.add_argument(
        "--packet",
        type=Path,
        default=Path("/tmp/g1_foot_rl_obs.bin"),
    )
    parser.add_argument(
        "--controller-config",
        type=Path,
        default=REPO_ROOT / "deploy/robots/g1_29dof/config/config.yaml",
    )
    parser.add_argument("--max-source-age", type=float, default=0.20)
    parser.add_argument("--max-period", type=float, default=0.10)
    parser.add_argument("--require-policy-active", action="store_true")
    return parser.parse_args()


class Checks:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.failures.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: {error}") from error


def check_config(path: Path, checks: Checks) -> None:
    config = read_json(path)
    checks.require(
        config.get("format") == "g1-dual-foot-ble-config-v1",
        "BLE config format is not g1-dual-foot-ble-config-v1",
    )
    addresses = []
    for side in ("left", "right"):
        foot = config.get(side, {})
        address = str(foot.get("address", "")).strip()
        addresses.append(address.casefold())
        checks.require(
            bool(address) and "replace" not in address.casefold(),
            f"{side} BLE address is missing/placeholder",
        )
        normalizer = str(foot.get("normalization", "")).strip()
        normalizer_path = (path.resolve().parent / normalizer).resolve()
        checks.require(
            bool(normalizer) and normalizer_path.is_file(),
            f"{side} normalization file is missing: {normalizer_path}",
        )
        if normalizer_path.is_file():
            document = read_json(normalizer_path)
            checks.require(
                document.get("side") == side,
                f"{side} normalization side tag does not match",
            )
    checks.require(
        len(addresses) == 2 and addresses[0] != addresses[1],
        "left/right BLE addresses must be different",
    )


def check_health(path: Path, max_age: float, max_period: float, checks: Checks) -> None:
    health = read_json(path)
    checks.require(
        health.get("format") == "g1-dual-foot-magnetic-health-v1",
        "magnetic health format mismatch",
    )
    publishing = health.get(
        "publishing_f0m1", health.get("publishing_f0t1", False)
    )
    checks.require(bool(publishing), "F0M1 bridge is not publishing")
    wall_age = (time.time_ns() - int(health.get("wall_time_ns", 0))) * 1.0e-9
    checks.require(
        math.isfinite(wall_age) and 0.0 <= wall_age <= 0.5,
        f"health document is stale ({wall_age:.3f}s)",
    )
    feet = health.get("feet", {})
    for side in ("left", "right"):
        foot = feet.get(side, {})
        age = float(foot.get("age_s", math.inf))
        period = float(foot.get("sample_period_s", math.inf))
        checks.require(bool(foot.get("connected")), f"{side} BLE is disconnected")
        checks.require(bool(foot.get("fresh")), f"{side} BLE data is stale")
        checks.require(bool(foot.get("normalized")), f"{side} data is not normalized")
        checks.require(age <= max_age, f"{side} source age {age:.3f}s exceeds limit")
        checks.require(
            0.001 <= period <= max_period,
            f"{side} sample period {period:.4f}s is outside limits",
        )
        checks.require(
            int(foot.get("frames", 0)) >= 20,
            f"{side} has fewer than 20 accepted frames",
        )
        checks.require(
            int(foot.get("rejected_frames", 0)) == 0,
            f"{side} has rejected BLE frames",
        )
        checks.require(
            not str(foot.get("last_error", "")).strip(),
            f"{side} reports error: {foot.get('last_error')}",
        )


def profile_residual(magnetic: tuple[float, ...]) -> float:
    normalized_profile = [value / (sum(PROFILE) / len(PROFILE)) for value in PROFILE]
    residual = 0.0
    evidence = 0.0
    count = 0
    for foot in range(2):
        for axis in range(3):
            values = [
                magnetic[(foot * 15 + sensor) * 3 + axis]
                / normalized_profile[sensor]
                for sensor in range(15)
            ]
            mean = sum(values) / len(values)
            residual += sum(abs(value - mean) for value in values)
            evidence += abs(mean) * len(values)
            count += len(values)
    return (residual / count) / (evidence / count + 0.05)


def check_packet(
    path: Path, max_source_age: float, max_period: float, checks: Checks
) -> None:
    payload = path.read_bytes()
    checks.require(len(payload) == PACKET_SIZE, f"F0M1 packet is {len(payload)} bytes")
    if len(payload) != PACKET_SIZE:
        return
    header = struct.unpack_from("<IIQffffff", payload)
    magic, sequence, stamp_ns = header[:3]
    valid = header[3:5]
    source_age = header[5:7]
    period = header[7:9]
    magnetic = struct.unpack_from("<90f", payload, 40)
    checks.require(magic == F0M1, f"packet magic is 0x{magic:08x}, expected F0M1")
    checks.require(sequence > 0, "packet sequence has not advanced")
    wall_age = (time.time_ns() - stamp_ns) * 1.0e-9
    checks.require(0.0 <= wall_age <= 0.25, f"packet wall age is {wall_age:.3f}s")
    for index, side in enumerate(("left", "right")):
        checks.require(valid[index] >= 0.999, f"{side} packet valid flag is {valid[index]}")
        checks.require(
            0.0 <= source_age[index] <= max_source_age,
            f"{side} packet source age is {source_age[index]:.3f}s",
        )
        checks.require(
            0.001 <= period[index] <= max_period,
            f"{side} packet period is {period[index]:.4f}s",
        )
    checks.require(all(math.isfinite(value) for value in magnetic), "magnetic data is non-finite")
    checks.require(max(map(abs, magnetic)) > 1.0e-4, "magnetic array is all zero")
    score = profile_residual(magnetic)
    mode = "FAST eligible" if score <= 0.06 else "SAFE fallback"
    checks.note(f"current-frame calibration residual={score:.5f} -> {mode}")


def check_controller(path: Path, require_active: bool, checks: Checks) -> None:
    text = path.read_text(encoding="utf-8")
    active = f"policy_dir: {FINAL_SLOT}" in text
    if require_active:
        checks.require(active, f"controller policy is not {FINAL_SLOT}")
    else:
        checks.note(
            "controller slot is "
            + ("final magnetic policy" if active else "not activated (expected before harness)")
        )


def main() -> int:
    args = parse_args()
    checks = Checks()
    for label, function in (
        ("config", lambda: check_config(args.config, checks)),
        (
            "health",
            lambda: check_health(
                args.health, args.max_source_age, args.max_period, checks
            ),
        ),
        (
            "packet",
            lambda: check_packet(
                args.packet, args.max_source_age, args.max_period, checks
            ),
        ),
        (
            "controller",
            lambda: check_controller(
                args.controller_config, args.require_policy_active, checks
            ),
        ),
    ):
        try:
            function()
        except (OSError, ValueError, TypeError) as error:
            checks.failures.append(f"{label}: {error}")
    for note in checks.notes:
        print(f"[INFO] {note}")
    if checks.failures:
        for failure in checks.failures:
            print(f"[FAIL] {failure}")
        print(f"PREFLIGHT FAIL ({len(checks.failures)} issue(s))")
        return 2
    print("PREFLIGHT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
