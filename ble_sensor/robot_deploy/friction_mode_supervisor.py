#!/usr/bin/env python3
"""Observe live F0R1 Hall data and emit fail-safe semantic mode requests.

This release is intentionally observe-only.  It cannot send Unitree motion or
FSM commands.  The semantic output is consumed later by a firmware-specific
executor only after the App mode mapping has been measured and approved.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from typing import Deque, Tuple

import numpy as np

from dual_foot_bridge.capture_ipc import read_packet
from friction_runtime import (
    FrictionDecisionStateMachine,
    LinearFrictionModel,
    extract_window_features,
)


def _write_json_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/tmp/g1_foot_hall_capture.bin"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--status", type=Path, default=Path("/tmp/g1_hall_friction_status.json"))
    parser.add_argument("--log", type=Path, default=Path("logs/friction_supervisor.jsonl"))
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--max-age", type=float, default=0.20)
    parser.add_argument("--apply-mode-requests", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.apply_mode_requests:
        raise RuntimeError(
            "active mode execution is intentionally unavailable until the App "
            "walkrun/waist firmware mapping and control lease are verified"
        )
    if not 10.0 <= args.rate <= 200.0 or not 0.02 <= args.max_age <= 0.50:
        raise ValueError("invalid runtime rate or max age")

    model = LinearFrictionModel.load(args.model, require_passed_gate=True)
    decision = FrictionDecisionStateMachine(
        enter_low_probability=model.enter_low_probability,
        clear_low_probability=model.clear_low_probability,
    )
    frames: Deque[Tuple[int, np.ndarray]] = deque(maxlen=model.window_frames)
    last_sequence = -1
    last_time = time.monotonic()
    period = 1.0 / args.rate
    args.log.parent.mkdir(parents=True, exist_ok=True)

    with args.log.open("a", encoding="utf-8") as stream:
        while True:
            started = time.monotonic()
            both_healthy = False
            probability_low = float("nan")
            source_sequence = None
            error = ""
            try:
                sample = read_packet(args.input)
                source_sequence = int(sample.sequence)
                both_healthy = bool(
                    all(sample.valid)
                    and max(sample.age_s) <= args.max_age
                    and all(value > 0.0 for value in sample.period_s)
                )
                sequence_gap = last_sequence >= 0 and int(sample.sequence) != last_sequence + 1
                if sequence_gap or not both_healthy:
                    frames.clear()
                if sample.sequence != last_sequence and both_healthy:
                    last_sequence = int(sample.sequence)
                    frames.append((sample.publish_monotonic_ns, sample.magnetic.copy()))
                elif sample.sequence != last_sequence:
                    last_sequence = int(sample.sequence)
                if both_healthy and len(frames) == model.window_frames:
                    probability_low = model.probability_low(
                        extract_window_features(np.stack([value for _, value in frames]))
                    )
            except (OSError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"

            now = time.monotonic()
            dt_s = max(1.0e-4, min(0.2, now - last_time))
            last_time = now
            output = decision.update(
                probability_low,
                dt_s,
                both_feet_healthy=both_healthy and len(frames) == model.window_frames,
                model_valid=True,
            )
            document = {
                "format": "g1-hall-friction-semantic-request-v1",
                "observe_only": True,
                "measurement": "dual-foot multiframe Hall Bx/By/Bz only",
                "monotonic_s": now,
                "source_sequence": source_sequence,
                "window_fill": len(frames),
                "window_frames": model.window_frames,
                "both_feet_healthy": both_healthy,
                "error": error,
                "decision": asdict(output),
                "control_boundary": (
                    "semantic request only; no Unitree FSM or velocity command was sent"
                ),
            }
            _write_json_atomic(args.status, document)
            stream.write(json.dumps(document, ensure_ascii=False) + "\n")
            stream.flush()
            remaining = period - (time.monotonic() - started)
            if remaining > 0.0:
                time.sleep(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
