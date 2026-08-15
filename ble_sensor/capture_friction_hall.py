#!/usr/bin/env python3
"""Collect a controlled single-foot Hall dataset on two labelled surfaces.

The insole measurement boundary is unchanged: every sample is only
15 x (Bx, By, Bz) raw Hall counts plus temperature and host timestamps.
Surface, phase, and trial are operator annotations; no force or friction
coefficient is inferred by this recorder.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import random
import time

import numpy as np

from dual_foot_bridge.bridge import _load_config
from dual_foot_bridge.protocol import CHAR_UUID, FrameError, FrameParser


PHASES = (
    ("baseline_unloaded", "保持该脚完全悬空，不接触试样"),
    ("static_contact", "只施加竖直载荷，保持足底与试样相对静止"),
    ("shear_probe", "保持相同竖直载荷，将试样沿前后方向约 1 Hz 小幅往复移动"),
    ("unload", "完全卸载，等待 TPU 回弹"),
)


@dataclass
class Recorder:
    parser: FrameParser = field(default_factory=FrameParser)
    surface: str = "setup"
    phase: str = "setup"
    trial: int = -1
    phase_started_ns: int = 0
    monotonic_ns: list[int] = field(default_factory=list)
    wall_ns: list[int] = field(default_factory=list)
    trial_id: list[int] = field(default_factory=list)
    surface_label: list[str] = field(default_factory=list)
    phase_label: list[str] = field(default_factory=list)
    phase_time_s: list[float] = field(default_factory=list)
    source_sequence: list[int] = field(default_factory=list)
    hall_xyz: list[np.ndarray] = field(default_factory=list)
    temperature_x10: list[np.ndarray] = field(default_factory=list)
    saturated: list[bool] = field(default_factory=list)
    rejected: int = 0

    def set_phase(self, trial: int, surface: str, phase: str) -> None:
        self.trial = trial
        self.surface = surface
        self.phase = phase
        self.phase_started_ns = time.monotonic_ns()

    def feed(self, payload: bytes) -> None:
        try:
            frame = self.parser.parse(payload)
        except (FrameError, ValueError):
            self.rejected += 1
            return
        self.monotonic_ns.append(frame.received_monotonic_ns)
        self.wall_ns.append(frame.received_wall_ns)
        self.trial_id.append(self.trial)
        self.surface_label.append(self.surface)
        self.phase_label.append(self.phase)
        self.phase_time_s.append(
            max(0, frame.received_monotonic_ns - self.phase_started_ns) * 1.0e-9
            if self.phase_started_ns
            else 0.0
        )
        self.source_sequence.append(frame.source_sequence)
        self.hall_xyz.append(frame.magnetic_xyz.copy())
        self.temperature_x10.append(frame.temperature_x10.copy())
        saturated = frame.saturation_xyz
        self.saturated.append(bool(saturated is not None and np.any(saturated)))

    def save(self, path: Path, metadata: dict) -> None:
        if not self.hall_xyz:
            raise RuntimeError("no valid Hall frames were recorded")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp.npz")
        np.savez_compressed(
            temporary,
            monotonic_ns=np.asarray(self.monotonic_ns, dtype=np.int64),
            wall_ns=np.asarray(self.wall_ns, dtype=np.int64),
            trial_id=np.asarray(self.trial_id, dtype=np.int32),
            surface=np.asarray(self.surface_label, dtype="U16"),
            phase=np.asarray(self.phase_label, dtype="U24"),
            phase_time_s=np.asarray(self.phase_time_s, dtype=np.float32),
            source_sequence=np.asarray(self.source_sequence, dtype=np.int16),
            hall_xyz=np.asarray(self.hall_xyz, dtype=np.int32),
            temperature_x10=np.asarray(self.temperature_x10, dtype=np.int32),
            saturated=np.asarray(self.saturated, dtype=bool),
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
        temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trial_order(trials_per_surface: int, seed: int) -> list[str]:
    if trials_per_surface < 3:
        raise ValueError("at least three trials per surface are required")
    values = ["high"] * trials_per_surface + ["low"] * trials_per_surface
    rng = random.Random(seed)
    for _ in range(1000):
        rng.shuffle(values)
        if all(values[i : i + 3] not in (["high"] * 3, ["low"] * 3) for i in range(len(values) - 2)):
            return values
    raise RuntimeError("could not create a balanced trial order")


async def _prompt(text: str, non_interactive: bool, pause_s: float) -> None:
    print(f"\n\a{text}", flush=True)
    if non_interactive:
        await asyncio.sleep(pause_s)
    else:
        answer = await asyncio.to_thread(input, "准备好后按 Enter；输入 q 结束：")
        if answer.strip().casefold() == "q":
            raise KeyboardInterrupt


async def _countdown(duration_s: float) -> None:
    deadline = time.monotonic() + duration_s
    last = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        seconds = int(np.ceil(remaining))
        if seconds != last and (seconds <= 5 or seconds % 5 == 0):
            print(f"  剩余 {seconds:2d} s", flush=True)
            last = seconds
        await asyncio.sleep(min(0.2, remaining))


async def collect(args: argparse.Namespace) -> tuple[Recorder, dict]:
    try:
        from bleak import BleakClient
    except ImportError as error:
        raise RuntimeError("missing bleak") from error

    config = _load_config(args.config)
    foot_cfg = config[args.side]
    address = str(foot_cfg.get("address", "")).strip()
    if not address:
        raise ValueError(f"{args.side}.address is empty")
    characteristic = str(config.get("ble", {}).get("characteristic_uuid", CHAR_UUID))
    order = _trial_order(args.trials_per_surface, args.seed)
    durations = {
        "baseline_unloaded": args.baseline_s,
        "static_contact": args.static_s,
        "shear_probe": args.shear_s,
        "unload": args.unload_s,
    }
    metadata = {
        "format": "g1-single-foot-labelled-friction-hall-v1",
        "measurement_boundary": "15xBx/By/Bz raw counts plus temperature only; labels are operator annotations; no force conversion",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "side": args.side,
        "device_name": str(foot_cfg.get("device_name", args.side)),
        "address": address,
        "adapter": args.adapter,
        "surface_high": args.high_surface,
        "surface_low": args.low_surface,
        "normal_load_control": args.normal_load_control,
        "trial_order": order,
        "durations_s": durations,
        "seed": args.seed,
        "protocol_note": "Each trial: unloaded baseline -> static normal contact -> approximately 1 Hz fore-aft tangential probe -> full unload.",
    }
    recorder = Recorder()
    client_options: dict[str, object] = {}
    if args.adapter:
        if "bluez" in inspect.signature(BleakClient).parameters:
            client_options["bluez"] = {"adapter": args.adapter}
        else:
            client_options["adapter"] = args.adapter
    client = BleakClient(address, timeout=20.0, **client_options)
    try:
        print(f"连接 {args.side} / {address} / {args.adapter} ...", flush=True)
        await client.connect()
        await client.start_notify(characteristic, lambda _sender, data: recorder.feed(bytes(data)))
        await asyncio.sleep(1.0)
        if len(recorder.hall_xyz) < 20:
            raise RuntimeError("BLE connected but fewer than 20 valid frames arrived in 1 s")
        print(f"链路正常：首秒 {len(recorder.hall_xyz)} 帧。试次顺序：{' -> '.join(order)}", flush=True)

        for trial, surface in enumerate(order):
            surface_name = args.high_surface if surface == "high" else args.low_surface
            recorder.set_phase(trial, surface, "operator_setup")
            await _prompt(
                f"试次 {trial + 1}/{len(order)}：换成 {surface.upper()} 表面 [{surface_name}]，脚保持完全卸载。",
                args.non_interactive,
                args.setup_pause_s,
            )
            for phase, instruction in PHASES:
                recorder.set_phase(trial, surface, phase)
                print(f"\a[{surface.upper()} T{trial:02d} / {phase}] {instruction}", flush=True)
                await _countdown(durations[phase])
        recorder.set_phase(-1, "done", "done")
        await asyncio.sleep(0.2)
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
    metadata["valid_frames"] = len(recorder.hall_xyz)
    metadata["rejected_frames"] = recorder.rejected
    metadata["saturated_frames"] = int(np.count_nonzero(recorder.saturated))
    return recorder, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.magnetic.json"))
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--adapter", default="hci0")
    parser.add_argument("--output-root", type=Path, default=Path("logs/friction_surface_sessions"))
    parser.add_argument("--session-name")
    parser.add_argument("--high-surface", default="rubber_high_friction")
    parser.add_argument("--low-surface", default="smooth_low_friction")
    parser.add_argument("--normal-load-control", default="same_manual_load_unverified")
    parser.add_argument("--trials-per-surface", type=int, default=4)
    parser.add_argument("--baseline-s", type=float, default=3.0)
    parser.add_argument("--static-s", type=float, default=4.0)
    parser.add_argument("--shear-s", type=float, default=8.0)
    parser.add_argument("--unload-s", type=float, default=3.0)
    parser.add_argument("--setup-pause-s", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--non-interactive", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.session_name or f"{args.side}_friction_{stamp}"
    session = (args.output_root / name).resolve()
    if session.exists():
        print(f"[ERROR] session already exists: {session}", flush=True)
        return 2
    session.mkdir(parents=True)
    data_path = session / "raw_labelled_hall.npz"
    manifest_path = session / "manifest.json"
    try:
        recorder, metadata = asyncio.run(collect(args))
        recorder.save(data_path, metadata)
        metadata["data_file"] = data_path.name
        metadata["data_sha256"] = _sha256(data_path)
        manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n采集完成：{session}", flush=True)
        print(f"有效帧={metadata['valid_frames']}，坏帧={metadata['rejected_frames']}，饱和帧={metadata['saturated_frames']}", flush=True)
        print("只记录 Hall/温度；未创建法向力、切向力或摩擦系数通道。", flush=True)
        return 0
    except KeyboardInterrupt:
        print("\n[STOP] 操作员结束；不把未完成试验封装为有效数据。", flush=True)
        return 130
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
