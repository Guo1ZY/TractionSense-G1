#!/usr/bin/env python3
"""Guided dual-foot Hall/temperature acquisition for calibration and sim-to-real.

The recorder stores only the real measurements available from the insole:
15 x (Bx, By, Bz) raw counts and temperature for each named foot.  Loading
instructions are experiment annotations; this program never converts Hall data
to normal or tangential force.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from dual_foot_bridge.bridge import _load_config
from dual_foot_bridge.magnetic_bridge import run as run_magnetic_bridge


@dataclass(frozen=True)
class Phase:
    key: str
    group: str
    duration_s: float
    instruction: str


PHASES = (
    Phase("baseline_unloaded", "baseline", 120.0, "双脚悬空且 TPU 完全卸载，保持静止"),
    Phase("temperature_drift", "baseline", 300.0, "双脚继续卸载，覆盖正常电子温升"),
    Phase("forefoot_normal", "motion", 45.0, "依次按压前掌区域并完全卸载，记录多次循环"),
    Phase("midfoot_normal", "motion", 45.0, "依次按压中足区域并完全卸载，记录多次循环"),
    Phase("heel_normal", "motion", 45.0, "依次按压后跟区域并完全卸载，记录多次循环"),
    Phase("shear_x", "motion", 45.0, "保持接触并沿足底局部 X 正负方向缓慢剪切"),
    Phase("shear_y", "motion", 45.0, "保持接触并沿脚尖/脚跟方向缓慢剪切"),
    Phase("tilt_pitch", "motion", 45.0, "施加前后倾斜并回到完全卸载状态"),
    Phase("tilt_roll", "motion", 45.0, "施加左右倾斜并回到完全卸载状态"),
    Phase("standing_shift", "motion", 60.0, "吊架保护下站立，缓慢进行前后和左右重心转移"),
    Phase("walking", "motion", 90.0, "吊架保护下低速行走，包含启动、停止和转向"),
    Phase("slip_events", "motion", 60.0, "吊架保护下在低摩擦表面制造小幅可控滑移"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _row_counts(path: Path) -> dict[str, int]:
    counts = {"left": 0, "right": 0}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            side = row.get("side", "")
            if side in counts:
                counts[side] += 1
    return counts


def _selected_phases(keys: list[str] | None, quick: bool) -> list[Phase]:
    requested = set(keys or [phase.key for phase in PHASES])
    unknown = requested - {phase.key for phase in PHASES}
    if unknown:
        raise ValueError(f"unknown phases: {sorted(unknown)}")
    result = []
    for phase in PHASES:
        if phase.key not in requested:
            continue
        duration = min(phase.duration_s, 12.0 if phase.group == "baseline" else 8.0)
        result.append(
            Phase(phase.key, phase.group, duration if quick else phase.duration_s, phase.instruction)
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.magnetic.json"))
    parser.add_argument("--output-root", type=Path, default=Path("calibration/sessions"))
    parser.add_argument("--session-name")
    parser.add_argument("--phase", action="append", choices=[phase.key for phase in PHASES])
    parser.add_argument("--quick", action="store_true", help="8-12 second wiring check per phase")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="write the plan without opening BLE")
    parser.add_argument("--note", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = _load_config(args.config)
        phases = _selected_phases(args.phase, args.quick)
    except (OSError, ValueError) as error:
        print(f"[ERROR] {error}")
        return 2
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = args.session_name or f"dual_hall_{stamp}"
    session_dir = (args.output_root / session_name).resolve()
    session_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = session_dir / "manifest.json"
    manifest = {
        "format": "g1-dual-foot-hall-dataset-v1",
        "measurement_boundary": "raw Hall Bx/By/Bz counts and temperature only; no force conversion",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "identity": {
            side: {
                "device_name": str(config[side].get("device_name", side)),
                "address": str(config[side].get("address", "")),
            }
            for side in ("left", "right")
        },
        "operator_note": args.note,
        "dry_run": bool(args.dry_run),
        "phases": [],
    }
    _write_json_atomic(manifest_path, manifest)

    for phase in phases:
        entry = asdict(phase)
        entry["status"] = "planned"
        entry["csv"] = f"{phase.key}.csv"
        manifest["phases"].append(entry)
        _write_json_atomic(manifest_path, manifest)
        print(f"\n[{phase.key}] {phase.instruction} · {phase.duration_s:.0f}s")
        if args.dry_run:
            entry["status"] = "dry_run"
            continue
        if not args.non_interactive:
            answer = input("准备好后按 Enter；输入 s 跳过，q 结束：").strip().casefold()
            if answer == "q":
                entry["status"] = "operator_stopped"
                break
            if answer == "s":
                entry["status"] = "skipped"
                continue
        csv_path = session_dir / entry["csv"]
        entry["started_utc"] = datetime.now(timezone.utc).isoformat()
        bridge_args = SimpleNamespace(
            config=args.config,
            out=None,
            health=session_dir / f"health_{phase.key}.json",
            record=csv_path,
            reference_left=None,
            reference_right=None,
            raw_only=True,
            duration=phase.duration_s,
        )
        result = asyncio.run(run_magnetic_bridge(bridge_args, config))
        if result != 0 or not csv_path.exists():
            entry["status"] = "failed"
            entry["return_code"] = int(result)
            _write_json_atomic(manifest_path, manifest)
            return 2
        entry.update(
            {
                "status": "complete",
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "rows": _row_counts(csv_path),
                "sha256": _sha256(csv_path),
            }
        )
        _write_json_atomic(manifest_path, manifest)

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(manifest_path, manifest)
    print(f"\nDataset manifest: {manifest_path}")
    print("Only Hall/temperature measurements were recorded; no force channel was created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
