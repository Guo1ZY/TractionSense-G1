#!/usr/bin/env python3
"""Record time-aligned real dual-foot Hall data for robot experiments.

This tool records only the hardware measurements that exist: per-foot P00-P14
Bx/By/Bz counts and temperature.  It also records timing and BLE health fields
needed to align Hall samples with a robot log.  It never creates force,
pressure, contact-force or friction values.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

from dual_foot_bridge.bridge import _load_config
from dual_foot_bridge.capture_ipc import PACKET_SIZE
from dual_foot_bridge.magnetic_bridge import run as run_magnetic_bridge


DEFAULT_CAPTURE_IPC = Path("/tmp/g1_foot_hall_capture.bin")
COMPETING_SCRIPTS = {
    "ble_viz_dashboard_demo.py",
    "ble_viz_superres_hot_detail.py",
    "capture_magnetic_dataset.py",
    "capture_robot_hall.py",
    "run_magnetic_bridge.py",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _available_adapters() -> dict[str, str]:
    adapters: dict[str, str] = {}
    for path in sorted(Path("/sys/class/bluetooth").glob("hci*")):
        address_path = path / "address"
        try:
            adapters[path.name] = address_path.read_text(encoding="utf-8").strip()
        except OSError:
            result = subprocess.run(
                ["hciconfig", path.name],
                check=False,
                capture_output=True,
                text=True,
            )
            match = re.search(r"BD Address:\s*([0-9A-Fa-f:]{17})", result.stdout)
            adapters[path.name] = match.group(1).upper() if match else "unknown"
    return adapters


def _competing_processes() -> list[dict[str, object]]:
    found = []
    own_pid = os.getpid()
    for process_dir in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(process_dir.name)
            if pid == own_pid:
                continue
            tokens = [
                value.decode("utf-8", errors="replace")
                for value in (process_dir / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except (OSError, ValueError):
            continue
        if not tokens or "python" not in Path(tokens[0]).name.casefold():
            continue
        script_names = {Path(token).name for token in tokens[1:]}
        matches = sorted(script_names & COMPETING_SCRIPTS)
        if matches:
            found.append({"pid": pid, "script": matches[0], "command": tokens})
    return found


def _count_raw_rows(path: Path) -> dict[str, int]:
    result = {"left": 0, "right": 0}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            side = row.get("side", "")
            if side in result:
                result[side] += 1
    return result


def _percentile_ceiling(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, int(len(ordered) * percentile + 0.999999) - 1)
    return float(ordered[index])


def _summarize_raw_frame_timing(path: Path) -> dict[str, dict[str, float | int]]:
    """Summarize actual Notify arrival timing without interpreting Hall values."""
    states = {
        side: {
            "frames": 0,
            "first_ns": 0,
            "last_ns": 0,
            "previous_ns": 0,
            "intervals_ms": [],
            "nonmonotonic_timestamps": 0,
        }
        for side in ("left", "right")
    }
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            side = row.get("side", "")
            if side not in states:
                continue
            try:
                timestamp_ns = int(row["monotonic_ns"])
            except (KeyError, TypeError, ValueError):
                continue
            state = states[side]
            previous_ns = int(state["previous_ns"])
            if state["frames"] and timestamp_ns <= previous_ns:
                state["nonmonotonic_timestamps"] = int(
                    state["nonmonotonic_timestamps"]
                ) + 1
                continue
            if not state["frames"]:
                state["first_ns"] = timestamp_ns
            if state["frames"]:
                state["intervals_ms"].append((timestamp_ns - previous_ns) / 1.0e6)
            state["previous_ns"] = timestamp_ns
            state["last_ns"] = timestamp_ns
            state["frames"] = int(state["frames"]) + 1

    result: dict[str, dict[str, float | int]] = {}
    for side, state in states.items():
        frames = int(state["frames"])
        duration_s = max(
            0.0, (int(state["last_ns"]) - int(state["first_ns"])) / 1.0e9
        )
        intervals = list(state["intervals_ms"])
        result[side] = {
            "frames": frames,
            "duration_s": round(duration_s, 6),
            "mean_rate_hz": round((frames - 1) / duration_s, 3)
            if frames > 1 and duration_s > 0.0
            else 0.0,
            "interval_ms_p50": round(_percentile_ceiling(intervals, 0.50), 6),
            "interval_ms_p95": round(_percentile_ceiling(intervals, 0.95), 6),
            "interval_ms_max": round(max(intervals) if intervals else 0.0, 6),
            "intervals_ge_40ms": sum(value >= 40.0 for value in intervals),
            "intervals_ge_100ms": sum(value >= 100.0 for value in intervals),
            "nonmonotonic_timestamps": int(state["nonmonotonic_timestamps"]),
        }
    return result


def _count_paired_rows(path: Path) -> dict[str, int]:
    result = {
        "rows": 0,
        "both_valid": 0,
        "left_valid": 0,
        "right_valid": 0,
        "rows_after_ready": 0,
        "both_valid_after_ready": 0,
    }
    ready_seen = False
    absolute_skew_ns: list[int] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            result["rows"] += 1
            left_valid = row.get("left_valid") == "1"
            right_valid = row.get("right_valid") == "1"
            ready_seen = ready_seen or (left_valid and right_valid)
            result["left_valid"] += int(left_valid)
            result["right_valid"] += int(right_valid)
            result["both_valid"] += int(left_valid and right_valid)
            if left_valid and right_valid:
                try:
                    absolute_skew_ns.append(
                        abs(int(row.get("left_right_frame_skew_ns", "0")))
                    )
                except ValueError:
                    pass
            if ready_seen:
                result["rows_after_ready"] += 1
                result["both_valid_after_ready"] += int(left_valid and right_valid)
    absolute_skew_ns.sort()
    if absolute_skew_ns:
        p95_index = max(0, int(0.95 * len(absolute_skew_ns) + 0.999999) - 1)
        result["abs_frame_skew_ns_p95"] = absolute_skew_ns[p95_index]
        result["abs_frame_skew_ns_max"] = absolute_skew_ns[-1]
    else:
        result["abs_frame_skew_ns_p95"] = 0
        result["abs_frame_skew_ns_max"] = 0
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.magnetic.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("logs/robot_capture_sessions")
    )
    parser.add_argument("--session-name")
    parser.add_argument("--left-adapter", default="hci0")
    parser.add_argument("--right-adapter", default="hci1")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--note", default="")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = _load_config(args.config)
    except (OSError, ValueError) as error:
        print(f"[ERROR] {error}")
        return 2
    if args.duration < 0.0:
        print("[ERROR] --duration must be non-negative")
        return 2
    if args.left_adapter == args.right_adapter:
        print("[ERROR] left and right must use different Bluetooth adapters")
        return 2

    adapters = _available_adapters()
    missing = sorted({args.left_adapter, args.right_adapter} - set(adapters))
    competitors = _competing_processes()
    preflight = {
        "format": "g1-dual-foot-hall-capture-preflight-v1",
        "adapters": adapters,
        "assignment": {
            "left": args.left_adapter,
            "right": args.right_adapter,
        },
        "missing_adapters": missing,
        "competing_ble_processes": competitors,
        "ready": not missing and not competitors,
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    if args.preflight_only:
        return 0 if preflight["ready"] else 3
    if missing:
        print(f"[ERROR] missing Bluetooth adapter(s): {', '.join(missing)}")
        return 3
    if competitors:
        print("[ERROR] stop the listed BLE process(es) before real collection")
        return 3

    config["left"]["adapter"] = args.left_adapter
    config["right"]["adapter"] = args.right_adapter
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = args.session_name or f"robot_hall_{stamp}"
    if not session_name or Path(session_name).name != session_name:
        print("[ERROR] --session-name must be one path component")
        return 2
    session_dir = (args.output_root / session_name).resolve()
    try:
        session_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"[ERROR] session already exists: {session_dir}")
        return 2

    raw_frames = session_dir / "raw_frames.csv"
    paired = session_dir / "paired_50hz.csv"
    health = session_dir / "health.json"
    manifest_path = session_dir / "manifest.json"
    manifest = {
        "format": "g1-dual-foot-robot-hall-capture-v1",
        "measurement_boundary": (
            "raw Hall Bx/By/Bz counts and temperature only; no force, pressure, "
            "contact-force or friction conversion"
        ),
        "sensor_order": "left/right remain separate; each side is P00..P14, Bx/By/Bz",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "operator_note": args.note,
        "requested_duration_s": args.duration,
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "identity": {
            side: {
                "device_name": str(config[side].get("device_name", side)),
                "address": str(config[side].get("address", "")),
                "adapter": str(config[side]["adapter"]),
                "adapter_address": adapters[str(config[side]["adapter"])],
            }
            for side in ("left", "right")
        },
        "timing": {
            "paired_rate_hz": float(config.get("output", {}).get("rate_hz", 50.0)),
            "pairing_method": (
                "latest real frame per side at each publish tick; no Hall interpolation"
            ),
            "clock_alignment": (
                "use publish_monotonic_ns to join same-host robot logs; per-foot frame "
                "timestamps, age and left_right_frame_skew_ns are retained"
            ),
        },
        "interfaces": {
            "live_f0r1": str(DEFAULT_CAPTURE_IPC),
            "f0r1_packet_bytes": PACKET_SIZE,
            "raw_frames_csv": str(raw_frames),
            "paired_csv": str(paired),
            "health_json": str(health),
        },
    }
    _write_json_atomic(manifest_path, manifest)

    bridge_args = SimpleNamespace(
        config=args.config,
        out=None,
        health=health,
        record=raw_frames,
        paired_record=paired,
        capture_out=DEFAULT_CAPTURE_IPC,
        reference_left=None,
        reference_right=None,
        raw_only=True,
        duration=args.duration,
        duration_after_ready=True,
        ready_timeout_s=30.0,
    )
    print(f"[CAPTURE] session={session_dir}")
    print("[CAPTURE] raw Hall/temperature only; Ctrl-C requests a clean stop")
    try:
        result = asyncio.run(run_magnetic_bridge(bridge_args, config))
        manifest["status"] = "complete" if result == 0 else "failed"
        manifest["return_code"] = int(result)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        result = 2
    manifest["stop_reason"] = str(getattr(bridge_args, "stop_reason", "exception"))
    if (
        args.duration > 0.0
        and manifest["status"] == "complete"
        and manifest["stop_reason"] != "duration_complete"
    ):
        manifest["status"] = "incomplete"
        manifest["error"] = (
            f"capture ended before the fixed duration: {manifest['stop_reason']}"
        )
        result = 4
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    for key, path in (("raw_frames", raw_frames), ("paired", paired)):
        if path.exists():
            manifest[f"{key}_sha256"] = _sha256(path)
    if raw_frames.exists():
        manifest["raw_frame_rows"] = _count_raw_rows(raw_frames)
        manifest["raw_frame_timing"] = _summarize_raw_frame_timing(raw_frames)
    if paired.exists():
        paired_rows = _count_paired_rows(paired)
        manifest["paired_rows"] = paired_rows
        paired_rate_hz = float(
            config.get("output", {}).get("rate_hz", 50.0)
        )
        if args.duration > 0.0:
            expected_rows = args.duration * paired_rate_hz
            manifest["expected_paired_rows_after_ready"] = expected_rows
            if (
                manifest["status"] == "complete"
                and paired_rows["rows_after_ready"] < 0.90 * expected_rows
            ):
                manifest["status"] = "incomplete"
                manifest["error"] = (
                    "capture stopped before 90% of the requested post-ready duration"
                )
                result = 4
    if health.exists():
        manifest["health_sha256"] = _sha256(health)
        try:
            manifest["final_health"] = json.loads(health.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            manifest["final_health_error"] = str(error)
    _write_json_atomic(manifest_path, manifest)
    print(f"[CAPTURE] status={manifest['status']} manifest={manifest_path}")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
