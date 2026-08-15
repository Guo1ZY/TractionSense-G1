#!/usr/bin/env python3
"""Read-only observer for the Unitree G1 locomotion service.

This program deliberately registers and calls only the official GET APIs.
It must not import or invoke SetFsmId, SetVelocity, or any other command API.
Use it while manually selecting modes in the Unitree App to establish the
firmware-specific mapping between the App labels and the reported FSM state.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_api import (
    LOCO_API_VERSION,
    LOCO_SERVICE_NAME,
    ROBOT_API_ID_LOCO_GET_BALANCE_MODE,
    ROBOT_API_ID_LOCO_GET_FSM_ID,
    ROBOT_API_ID_LOCO_GET_FSM_MODE,
    ROBOT_API_ID_LOCO_GET_STAND_HEIGHT,
    ROBOT_API_ID_LOCO_GET_SWING_HEIGHT,
)
from unitree_sdk2py.rpc.client import Client


GET_APIS = {
    "fsm_id": ROBOT_API_ID_LOCO_GET_FSM_ID,
    "fsm_mode": ROBOT_API_ID_LOCO_GET_FSM_MODE,
    "balance_mode": ROBOT_API_ID_LOCO_GET_BALANCE_MODE,
    "stand_height": ROBOT_API_ID_LOCO_GET_STAND_HEIGHT,
    "swing_height": ROBOT_API_ID_LOCO_GET_SWING_HEIGHT,
}


class ReadOnlyLocoClient(Client):
    """Minimal client with no command API registered."""

    def __init__(self) -> None:
        super().__init__(LOCO_SERVICE_NAME, False)

    def initialize(self, timeout_s: float) -> None:
        self._SetApiVerson(LOCO_API_VERSION)
        self.SetTimeout(timeout_s)
        for api_id in GET_APIS.values():
            self._RegistApi(api_id, 0)

    def read_value(self, api_id: int) -> Tuple[int, Optional[Any]]:
        code, payload = self._Call(api_id, "{}")
        if code != 0:
            return int(code), None
        try:
            return 0, json.loads(payload).get("data")
        except (TypeError, json.JSONDecodeError):
            return -1, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--rate", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    args = parse_args()
    if args.duration <= 0.0 or args.rate <= 0.0 or args.timeout <= 0.0:
        raise ValueError("duration, rate, and timeout must be positive")

    ChannelFactoryInitialize(0, args.interface)
    client = ReadOnlyLocoClient()
    client.initialize(args.timeout)

    stream = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stream = args.output.open("a", encoding="utf-8")

    print(
        "READ_ONLY_LOCO_PROBE "
        f"service={LOCO_SERVICE_NAME} api={LOCO_API_VERSION} "
        f"interface={args.interface}"
    )
    print("This process has registered GET APIs only; it cannot move the robot.")

    previous: Optional[Dict[str, Any]] = None
    deadline = time.monotonic() + args.duration
    period = 1.0 / args.rate
    sequence = 0
    failures = 0
    try:
        while time.monotonic() < deadline:
            started = time.monotonic()
            values: Dict[str, Any] = {}
            codes: Dict[str, int] = {}
            for name, api_id in GET_APIS.items():
                code, value = client.read_value(api_id)
                codes[name] = code
                values[name] = _safe_json_value(value)
                failures += int(code != 0)
            row = {
                "format": "unitree-g1-read-only-loco-state-v1",
                "sequence": sequence,
                "wall_time_utc": datetime.now(timezone.utc).isoformat(),
                "monotonic_s": started,
                "interface": args.interface,
                "service": LOCO_SERVICE_NAME,
                "api_version": LOCO_API_VERSION,
                "values": values,
                "return_codes": codes,
            }
            if stream is not None:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
            if previous != values or any(code != 0 for code in codes.values()):
                print(json.dumps(row, ensure_ascii=False), flush=True)
            previous = values
            sequence += 1
            remaining = period - (time.monotonic() - started)
            if remaining > 0.0:
                time.sleep(remaining)
    finally:
        if stream is not None:
            stream.close()

    print(f"probe_complete samples={sequence} api_failures={failures}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
