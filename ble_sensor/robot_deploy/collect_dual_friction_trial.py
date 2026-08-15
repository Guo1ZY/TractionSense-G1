#!/usr/bin/env python3
"""Record one labelled real-surface trial from the live dual-foot F0R1 stream.

Start capture_robot_hall.py first.  This tool never commands the robot; the
operator uses one fixed official controller mode and one fixed requested speed
for every high/low trial.  It records only Hall/temperature/timing/health plus
operator annotations.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import List

import numpy as np

from dual_foot_bridge.capture_ipc import read_packet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/tmp/g1_foot_hall_capture.bin"))
    parser.add_argument("--surface", choices=("high", "low"), required=True)
    parser.add_argument("--surface-name", required=True)
    parser.add_argument("--controller-mode", choices=("walkrun", "waist_walk"), required=True)
    parser.add_argument("--requested-vx", type=float, required=True)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--max-age", type=float, default=0.20)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("logs/friction_trials"))
    parser.add_argument("--note", default="")
    parser.add_argument("--yes", action="store_true", help="skip the operator Enter prompt")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.duration < 5.0:
        raise ValueError("duration must be at least 5 seconds")
    if not 0.0 < args.requested_vx <= 1.0:
        raise ValueError("requested-vx must be in (0,1.0] m/s")
    if not args.yes:
        print(
            f"表面={args.surface}/{args.surface_name}，模式={args.controller_mode}，"
            f"指令={args.requested_vx:.3f} m/s。\n"
            "确认机器人有吊架/保护员、路面清空，并已启动双 BLE 原始采集。"
        )
        input("准备后按 Enter；本程序不会向机器人发送运动命令：")

    deadline_ready = time.monotonic() + 30.0
    first = None
    while time.monotonic() < deadline_ready:
        try:
            candidate = read_packet(args.input)
            if all(candidate.valid) and max(candidate.age_s) <= args.max_age:
                first = candidate
                break
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    if first is None:
        raise RuntimeError("dual-foot F0R1 did not become healthy within 30 seconds")

    sequences: List[int] = []
    publish_ns: List[int] = []
    frame_ns: List[tuple] = []
    valid: List[tuple] = []
    age_s: List[tuple] = []
    period_s: List[tuple] = []
    magnetic: List[np.ndarray] = []
    temperature: List[np.ndarray] = []
    last_sequence = -1
    started = time.monotonic()
    deadline = started + args.duration
    while time.monotonic() < deadline:
        sample = read_packet(args.input)
        if int(sample.sequence) == last_sequence:
            time.sleep(0.002)
            continue
        last_sequence = int(sample.sequence)
        sequences.append(last_sequence)
        publish_ns.append(int(sample.publish_monotonic_ns))
        frame_ns.append(tuple(int(value) for value in sample.frame_monotonic_ns))
        valid.append(tuple(bool(value) for value in sample.valid))
        age_s.append(tuple(float(value) for value in sample.age_s))
        period_s.append(tuple(float(value) for value in sample.period_s))
        magnetic.append(sample.magnetic.copy())
        temperature.append(sample.temperature_x10.copy())
        time.sleep(0.002)

    if len(magnetic) < int(args.duration * 30.0):
        raise RuntimeError("fewer than 30 synchronized samples/s were recorded")
    sequence_array = np.asarray(sequences, dtype=np.uint64)
    if np.any(np.diff(sequence_array.astype(np.int64)) != 1):
        raise RuntimeError("F0R1 sequence has gaps; repeat this trial")
    valid_array = np.asarray(valid, dtype=bool)
    age_array = np.asarray(age_s, dtype=np.float32)
    healthy = np.all(valid_array, axis=1) & (np.max(age_array, axis=1) <= args.max_age)
    if np.mean(healthy) < 0.98:
        raise RuntimeError("less than 98% of the trial has two healthy feet")

    metadata = {
        "format": "g1-dual-foot-labelled-friction-trial-v1",
        "measurement": "raw dual-foot Hall Bx/By/Bz and temperature only",
        "force_conversion": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "surface_label": args.surface,
        "surface_name": args.surface_name,
        "controller_mode": args.controller_mode,
        "requested_vx_mps": args.requested_vx,
        "trial_id": args.trial_id,
        "duration_s": args.duration,
        "operator_note": args.note,
        "anti_confound_contract": (
            "All high/low trials used for one classifier must use the same "
            "controller_mode and requested_vx_mps."
        ),
        "samples": len(magnetic),
        "healthy_fraction": float(np.mean(healthy)),
    }
    output = args.output_root / f"{args.trial_id}_{args.surface}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    np.savez_compressed(
        temporary,
        sequence=sequence_array,
        publish_monotonic_ns=np.asarray(publish_ns, dtype=np.int64),
        frame_monotonic_ns=np.asarray(frame_ns, dtype=np.int64),
        valid=valid_array,
        age_s=age_array,
        period_s=np.asarray(period_s, dtype=np.float32),
        hall_xyz=np.asarray(magnetic, dtype=np.int64),
        temperature_x10=np.asarray(temperature, dtype=np.int32),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    temporary.replace(output)
    manifest = dict(metadata)
    manifest["data_file"] = output.name
    manifest["data_sha256"] = _sha256(output)
    output.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"trial_complete path={output} samples={len(magnetic)} healthy={np.mean(healthy):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
