#!/usr/bin/env python3
"""Log MuJoCo / ROS foot-bridge packets to CSV (+ optional JSONL).

Reads /tmp/g1_foot_rl_obs.bin (or G1_FOOT_BRIDGE_PATH), same layout as
foot_bridge.h / watch_foot_bridge.sh / MuJoCo write_foot_bridge_from_mujoco.

F0T1 (40 B): magic, seq, stamp_ns, cL,cR, nL,nR, tL,tR  (n,t already *0.01)
F0T2 (48 B): magic, seq, stamp_ns, cL,cR, nL,nR, loadL,loadR, valid, age_norm

Usage:
  # Terminal A: MuJoCo with foot bridge
  export G1_MUJOCO_FOOT_BRIDGE=1
  ./research_scripts/run_mujoco_friction.sh normal

  # Terminal B: log at 20 Hz
  ./research_scripts/log_foot_bridge.py --hz 20 --tag baseline_full

  # During run, type and Enter to mark friction phase (also written as event rows):
  #   1 / ice     → mu_mode=ice
  #   2 / normal  → mu_mode=normal
  #   3 / grip    → mu_mode=grip
  #   4 / ultra   → mu_mode=ultra_ice
  #   note text  → note=<text>
  #   q          → quit

Output default:
  ~/guo/logs/foot_bridge/foot_YYYYMMDD_HHMMSS_<tag>.csv
  ~/guo/logs/foot_bridge/foot_YYYYMMDD_HHMMSS_<tag>.jsonl  (if --jsonl)
  ~/guo/logs/foot_bridge/foot_YYYYMMDD_HHMMSS_<tag>_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import select
import struct
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

MAGIC_F0T1 = 0x46305431
MAGIC_F0T2 = 0x46305432
FORCE_SCALE = 0.01  # train / bridge scale
STALE_SEC = 0.25

CSV_FIELDS = [
    "wall_time",
    "wall_iso",
    "elapsed_s",
    "file_mtime_age_s",
    "status",
    "magic_hex",
    "seq",
    "stamp_ns",
    "stamp_age_s",
    "mu_mode",
    "mu_mode_sim",
    "mu_slide",
    "note",
    "vx",
    "vy",
    "vz",
    "vxy",
    "abs_vx",
    "abs_vy",
    "pos_y",
    "contact_L",
    "contact_R",
    "normal_scaled_L",
    "normal_scaled_R",
    "tangent_scaled_L",
    "tangent_scaled_R",
    "Fn_N_L",
    "Fn_N_R",
    "Ft_N_L",
    "Ft_N_R",
    "rho_L",
    "rho_R",
    "load_L",
    "load_R",
    "sensor_valid",
    "sensor_age_norm",
    "packet_bytes",
]


@dataclass
class RunState:
    mu_mode: str = "unknown"
    note: str = ""
    rows: int = 0
    ok_rows: int = 0
    stale_rows: int = 0
    missing_rows: int = 0
    events: list[dict] = field(default_factory=list)
    # for summary
    fn_sum: float = 0.0
    ft_sum: float = 0.0
    rho_sum: float = 0.0
    samples_force: int = 0
    by_mode: dict = field(default_factory=dict)
    # velocity summary by mode (from MuJoCo sidecar)
    vel_by_mode: dict = field(default_factory=dict)


def read_base_vel(path: Path) -> dict | None:
    """Read /tmp/g1_base_vel.json written by unitree_mujoco."""
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        # one JSON object per file (latest)
        return json.loads(raw.splitlines()[-1])
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def default_out_dir() -> Path:
    env = os.environ.get("G1_FOOT_LOG_DIR")
    if env:
        return Path(env).expanduser()
    # prefer project logs
    project_root = Path(
        os.environ.get(
            "TRACTIONSENSE_ROOT", Path(__file__).resolve().parents[1]
        )
    )
    for cand in (
        project_root / "logs/foot_bridge",
        Path("/tmp/g1_foot_bridge_logs"),
    ):
        try:
            cand.mkdir(parents=True, exist_ok=True)
            return cand
        except OSError:
            continue
    return Path("/tmp")


def parse_packet(data: bytes) -> dict | None:
    if len(data) < 16:
        return None
    magic, seq = struct.unpack_from("<II", data, 0)
    stamp_ns = struct.unpack_from("<Q", data, 8)[0]
    out = {
        "magic": magic,
        "seq": seq,
        "stamp_ns": stamp_ns,
        "packet_bytes": len(data),
        "contact_L": 0.0,
        "contact_R": 0.0,
        "normal_scaled_L": 0.0,
        "normal_scaled_R": 0.0,
        "tangent_scaled_L": 0.0,
        "tangent_scaled_R": 0.0,
        "load_L": 0.5,
        "load_R": 0.5,
        "sensor_valid": 0.0,
        "sensor_age_norm": 1.0,
    }
    if magic == MAGIC_F0T1 and len(data) >= 40:
        cL, cR = struct.unpack_from("<ff", data, 16)
        nL, nR = struct.unpack_from("<ff", data, 24)
        tL, tR = struct.unpack_from("<ff", data, 32)
        out.update(
            contact_L=cL,
            contact_R=cR,
            normal_scaled_L=nL,
            normal_scaled_R=nR,
            tangent_scaled_L=tL,
            tangent_scaled_R=tR,
            sensor_valid=1.0,
            sensor_age_norm=0.0,
        )
        s = abs(nL) + abs(nR) + 1e-6
        out["load_L"] = abs(nL) / s
        out["load_R"] = abs(nR) / s
        return out
    if magic == MAGIC_F0T2 and len(data) >= 48:
        cL, cR = struct.unpack_from("<ff", data, 16)
        nL, nR = struct.unpack_from("<ff", data, 24)
        ldL, ldR = struct.unpack_from("<ff", data, 32)
        valid, age = struct.unpack_from("<ff", data, 40)
        out.update(
            contact_L=cL,
            contact_R=cR,
            normal_scaled_L=nL,
            normal_scaled_R=nR,
            load_L=ldL,
            load_R=ldR,
            sensor_valid=valid,
            sensor_age_norm=age,
        )
        return out
    return None


def enrich(
    row: dict,
    wall: float,
    t0: float,
    mtime_age: float,
    state: RunState,
    vel: dict | None = None,
) -> dict:
    nL = float(row["normal_scaled_L"])
    nR = float(row["normal_scaled_R"])
    tL = float(row["tangent_scaled_L"])
    tR = float(row["tangent_scaled_R"])
    fnL = nL / FORCE_SCALE
    fnR = nR / FORCE_SCALE
    ftL = tL / FORCE_SCALE
    ftR = tR / FORCE_SCALE
    rhoL = ftL / (fnL + 1.0)
    rhoR = ftR / (fnR + 1.0)

    stamp_ns = int(row["stamp_ns"])
    stamp_age = (wall * 1e9 - stamp_ns) * 1e-9 if stamp_ns > 0 else float("nan")

    magic = int(row["magic"])
    status = "OK"
    if mtime_age > STALE_SEC:
        status = "STALE_FILE"
    if stamp_ns > 0 and stamp_age == stamp_age and (stamp_age < 0 or stamp_age > STALE_SEC):
        status = "STALE_STAMP"
    if magic not in (MAGIC_F0T1, MAGIC_F0T2):
        status = "BAD_MAGIC"
    if row.get("sensor_valid", 1.0) < 0.5 and magic == MAGIC_F0T2:
        status = "INVALID"

    vx = vy = vz = vxy = pos_y = float("nan")
    mu_mode_sim = ""
    mu_slide = ""
    if vel:
        try:
            vx = float(vel.get("vx", float("nan")))
            vy = float(vel.get("vy", float("nan")))
            vz = float(vel.get("vz", float("nan")))
            vxy = float(vel.get("vxy", math.hypot(vx, vy)))
            pos_y = float(vel.get("y", float("nan")))
            mu_mode_sim = str(vel.get("mu_mode", "") or "")
            if "mu_slide" in vel and vel["mu_slide"] is not None:
                mu_slide = f"{float(vel['mu_slide']):.4f}"
        except (TypeError, ValueError):
            pass

    # Prefer operator tag; if still "unknown"/"normal" and sim reports grip/ice, show sim in mu_mode_sim
    return {
        "wall_time": f"{wall:.6f}",
        "wall_iso": datetime.fromtimestamp(wall).isoformat(timespec="milliseconds"),
        "elapsed_s": f"{wall - t0:.3f}",
        "file_mtime_age_s": f"{mtime_age:.4f}",
        "status": status,
        "magic_hex": f"0x{magic:08X}",
        "seq": row["seq"],
        "stamp_ns": stamp_ns,
        "stamp_age_s": f"{stamp_age:.4f}" if stamp_age == stamp_age else "",
        "mu_mode": state.mu_mode,
        "mu_mode_sim": mu_mode_sim,
        "mu_slide": mu_slide,
        "note": state.note,
        "vx": f"{vx:.4f}" if vx == vx else "",
        "vy": f"{vy:.4f}" if vy == vy else "",
        "vz": f"{vz:.4f}" if vz == vz else "",
        "vxy": f"{vxy:.4f}" if vxy == vxy else "",
        "abs_vx": f"{abs(vx):.4f}" if vx == vx else "",
        "abs_vy": f"{abs(vy):.4f}" if vy == vy else "",
        "pos_y": f"{pos_y:.4f}" if pos_y == pos_y else "",
        "contact_L": f"{row['contact_L']:.6f}",
        "contact_R": f"{row['contact_R']:.6f}",
        "normal_scaled_L": f"{nL:.6f}",
        "normal_scaled_R": f"{nR:.6f}",
        "tangent_scaled_L": f"{tL:.6f}",
        "tangent_scaled_R": f"{tR:.6f}",
        "Fn_N_L": f"{fnL:.3f}",
        "Fn_N_R": f"{fnR:.3f}",
        "Ft_N_L": f"{ftL:.3f}",
        "Ft_N_R": f"{ftR:.3f}",
        "rho_L": f"{rhoL:.4f}",
        "rho_R": f"{rhoR:.4f}",
        "load_L": f"{float(row['load_L']):.4f}",
        "load_R": f"{float(row['load_R']):.4f}",
        "sensor_valid": f"{float(row['sensor_valid']):.3f}",
        "sensor_age_norm": f"{float(row['sensor_age_norm']):.3f}",
        "packet_bytes": row["packet_bytes"],
        "_fn_mean": 0.5 * (fnL + fnR),
        "_ft_mean": 0.5 * (ftL + ftR),
        "_rho_mean": 0.5 * (rhoL + rhoR),
        "_status": status,
        "_vx": vx,
        "_vy": vy,
        "_vxy": vxy,
    }


def handle_stdin_line(line: str, state: RunState, wall: float) -> bool:
    """Return False to quit."""
    s = line.strip()
    if not s:
        return True
    low = s.lower()
    if low in ("q", "quit", "exit"):
        return False
    mapping = {
        "1": "ice",
        "ice": "ice",
        "slip": "ice",
        "2": "normal",
        "normal": "normal",
        "3": "grip",
        "grip": "grip",
        "4": "ultra_ice",
        "ultra": "ultra_ice",
        "ultra_ice": "ultra_ice",
        "0": "zones",
        "zones": "zones",
    }
    if low in mapping:
        state.mu_mode = mapping[low]
        state.note = ""
        state.events.append({"t": wall, "type": "mu_mode", "value": state.mu_mode})
        print(f"\n[event] mu_mode → {state.mu_mode}", flush=True)
        return True
    if low.startswith("note ") or low.startswith("n "):
        state.note = s.split(None, 1)[1] if " " in s else ""
        state.events.append({"t": wall, "type": "note", "value": state.note})
        print(f"\n[event] note → {state.note!r}", flush=True)
        return True
    # free-form note
    state.note = s
    state.events.append({"t": wall, "type": "note", "value": state.note})
    print(f"\n[event] note → {state.note!r}", flush=True)
    return True


def poll_stdin(state: RunState, wall: float) -> bool:
    if not sys.stdin.isatty():
        return True
    try:
        r, _, _ = select.select([sys.stdin], [], [], 0.0)
    except (ValueError, OSError):
        return True
    if not r:
        return True
    line = sys.stdin.readline()
    if line == "":
        return True
    return handle_stdin_line(line, state, wall)


def update_summary(state: RunState, rec: dict) -> None:
    state.rows += 1
    st = rec["_status"]
    if st == "OK":
        state.ok_rows += 1
    elif "STALE" in st:
        state.stale_rows += 1
    elif st in ("NO_FILE", "BAD_SIZE", "BAD_MAGIC", "INVALID"):
        state.missing_rows += 1

    if st == "OK":
        state.fn_sum += rec["_fn_mean"]
        state.ft_sum += rec["_ft_mean"]
        state.rho_sum += rec["_rho_mean"]
        state.samples_force += 1
        mode = rec["mu_mode"]
        b = state.by_mode.setdefault(
            mode, {"n": 0, "fn": 0.0, "ft": 0.0, "rho": 0.0}
        )
        b["n"] += 1
        b["fn"] += rec["_fn_mean"]
        b["ft"] += rec["_ft_mean"]
        b["rho"] += rec["_rho_mean"]
        vx = rec.get("_vx", float("nan"))
        vy = rec.get("_vy", float("nan"))
        vxy = rec.get("_vxy", float("nan"))
        if vx == vx and vy == vy:
            vb = state.vel_by_mode.setdefault(
                mode,
                {
                    "n": 0,
                    "vxy": 0.0,
                    "abs_vx": 0.0,
                    "abs_vy": 0.0,
                    "n_move": 0,
                    "vxy_move": 0.0,
                    "abs_vx_move": 0.0,
                    "abs_vy_move": 0.0,
                },
            )
            vb["n"] += 1
            vb["vxy"] += abs(vxy) if vxy == vxy else 0.0
            vb["abs_vx"] += abs(vx)
            vb["abs_vy"] += abs(vy)
            if vxy == vxy and abs(vxy) > 0.15:
                vb["n_move"] += 1
                vb["vxy_move"] += abs(vxy)
                vb["abs_vx_move"] += abs(vx)
                vb["abs_vy_move"] += abs(vy)


def write_summary(path: Path, state: RunState, t0: float, csv_path: Path) -> None:
    dur = time.time() - t0
    lines = [
        "foot_bridge log summary",
        f"csv: {csv_path}",
        f"duration_s: {dur:.1f}",
        f"rows: {state.rows}  ok: {state.ok_rows}  stale: {state.stale_rows}  missing/bad: {state.missing_rows}",
        f"final_mu_mode: {state.mu_mode}",
    ]
    if state.samples_force:
        n = state.samples_force
        lines.append(
            f"OK mean Fn(N): {state.fn_sum / n:.1f}  "
            f"Ft(N): {state.ft_sum / n:.1f}  "
            f"rho: {state.rho_sum / n:.3f}"
        )
    lines.append("by mu_mode (OK samples only):")
    for mode, b in sorted(state.by_mode.items()):
        if b["n"] == 0:
            continue
        lines.append(
            f"  {mode}: n={b['n']}  "
            f"Fn={b['fn']/b['n']:.1f}N  Ft={b['ft']/b['n']:.1f}N  rho={b['rho']/b['n']:.3f}"
        )
    if state.vel_by_mode:
        lines.append("velocity by mu_mode (need rebuilt MuJoCo writing /tmp/g1_base_vel.json):")
        for mode, b in sorted(state.vel_by_mode.items()):
            if b["n"] == 0:
                continue
            lines.append(
                f"  {mode}: n={b['n']}  "
                f"|v|={b['vxy']/b['n']:.3f}  |vx|={b['abs_vx']/b['n']:.3f}  |vy|={b['abs_vy']/b['n']:.3f}"
            )
            if b["n_move"] > 0:
                lines.append(
                    f"    moving(|v|>0.15) n={b['n_move']}  "
                    f"|v|={b['vxy_move']/b['n_move']:.3f}  "
                    f"|vx|={b['abs_vx_move']/b['n_move']:.3f}  "
                    f"|vy|={b['abs_vy_move']/b['n_move']:.3f}"
                )
    if state.events:
        lines.append("events:")
        for e in state.events:
            lines.append(f"  t={e['t']-t0:.1f}s  {e['type']}={e['value']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Log g1 foot bridge packets to CSV")
    p.add_argument(
        "--path",
        default=os.environ.get("G1_FOOT_BRIDGE_PATH", "/tmp/g1_foot_rl_obs.bin"),
        help="Bridge binary path",
    )
    p.add_argument(
        "--vel-path",
        default=os.environ.get("G1_BASE_VEL_PATH", "/tmp/g1_base_vel.json"),
        help="MuJoCo base velocity JSON sidecar",
    )
    p.add_argument("--hz", type=float, default=20.0, help="Sample rate (default 20)")
    p.add_argument("--tag", type=str, default="run", help="Filename tag")
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory (default ~/guo/logs/foot_bridge)",
    )
    p.add_argument("--jsonl", action="store_true", help="Also write JSONL")
    p.add_argument(
        "--mu-mode",
        type=str,
        default="unknown",
        help="Initial friction label (ice/normal/grip/...)",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Auto-stop after N seconds (0 = until Ctrl+C)",
    )
    p.add_argument(
        "--no-console",
        action="store_true",
        help="Do not print live line (still logs)",
    )
    p.add_argument(
        "--print-hz",
        type=float,
        default=5.0,
        help="Console refresh rate (default 5; log rate is --hz)",
    )
    args = p.parse_args(argv)

    bridge = Path(args.path)
    vel_path = Path(args.vel_path)
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.tag)
    base = out_dir / f"foot_{ts}_{safe_tag}"
    csv_path = base.with_suffix(".csv")
    jsonl_path = base.with_suffix(".jsonl")
    summary_path = Path(str(base) + "_summary.txt")

    state = RunState(mu_mode=args.mu_mode)
    dt = 1.0 / max(args.hz, 0.5)
    print_every = 1.0 / max(args.print_hz, 0.2)
    last_print = 0.0
    t0 = time.time()

    print("============================================================")
    print(" foot bridge logger")
    print(f"  bridge : {bridge}")
    print(f"  vel    : {vel_path}  (vx/vy from MuJoCo; rebuild unitree_mujoco if missing)")
    print(f"  rate   : {args.hz} Hz log / {args.print_hz} Hz console")
    print(f"  csv    : {csv_path}")
    if args.jsonl:
        print(f"  jsonl  : {jsonl_path}")
    print(f"  mu_mode: {state.mu_mode}  (type 1/2/3/4 + Enter to retag)")
    print("  keys   : 1=ice 2=normal 3=grip 4=ultra  note <text>  q=quit")
    print("  test   : 满杆只推前后，左右摇杆回中（避免人为 vy）")
    print("============================================================")
    print(
        "Tip: MuJoCo terminal still uses 1/2/3 for physics; "
        "retag HERE so CSV mu_mode matches what you pressed.\n",
        flush=True,
    )

    f_json = open(jsonl_path, "w", encoding="utf-8") if args.jsonl else None
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()

            while True:
                wall = time.time()
                if args.duration > 0 and (wall - t0) >= args.duration:
                    print("\n[info] duration reached", flush=True)
                    break
                if not poll_stdin(state, wall):
                    print("\n[info] quit", flush=True)
                    break

                mtime_age = 1e9
                vel = read_base_vel(vel_path)
                rec: dict
                if not bridge.exists():
                    rec = {
                        "wall_time": f"{wall:.6f}",
                        "wall_iso": datetime.fromtimestamp(wall).isoformat(timespec="milliseconds"),
                        "elapsed_s": f"{wall - t0:.3f}",
                        "file_mtime_age_s": "",
                        "status": "NO_FILE",
                        "magic_hex": "",
                        "seq": "",
                        "stamp_ns": "",
                        "stamp_age_s": "",
                        "mu_mode": state.mu_mode,
                        "mu_mode_sim": (vel or {}).get("mu_mode", "") if vel else "",
                        "mu_slide": "",
                        "note": state.note,
                        "vx": "",
                        "vy": "",
                        "vz": "",
                        "vxy": "",
                        "abs_vx": "",
                        "abs_vy": "",
                        "pos_y": "",
                        "contact_L": "",
                        "contact_R": "",
                        "normal_scaled_L": "",
                        "normal_scaled_R": "",
                        "tangent_scaled_L": "",
                        "tangent_scaled_R": "",
                        "Fn_N_L": "",
                        "Fn_N_R": "",
                        "Ft_N_L": "",
                        "Ft_N_R": "",
                        "rho_L": "",
                        "rho_R": "",
                        "load_L": "",
                        "load_R": "",
                        "sensor_valid": "",
                        "sensor_age_norm": "",
                        "packet_bytes": 0,
                        "_fn_mean": 0.0,
                        "_ft_mean": 0.0,
                        "_rho_mean": 0.0,
                        "_status": "NO_FILE",
                        "_vx": float("nan"),
                        "_vy": float("nan"),
                        "_vxy": float("nan"),
                    }
                    if vel:
                        try:
                            rec["_vx"] = float(vel.get("vx", float("nan")))
                            rec["_vy"] = float(vel.get("vy", float("nan")))
                            rec["_vxy"] = float(vel.get("vxy", float("nan")))
                            rec["vx"] = f"{rec['_vx']:.4f}"
                            rec["vy"] = f"{rec['_vy']:.4f}"
                            rec["vxy"] = f"{rec['_vxy']:.4f}"
                            rec["abs_vx"] = f"{abs(rec['_vx']):.4f}"
                            rec["abs_vy"] = f"{abs(rec['_vy']):.4f}"
                        except (TypeError, ValueError):
                            pass
                else:
                    try:
                        raw = bridge.read_bytes()
                        mtime_age = wall - bridge.stat().st_mtime
                    except OSError as e:
                        raw = b""
                        mtime_age = 1e9
                        print(f"\n[warn] read failed: {e}", flush=True)
                    parsed = parse_packet(raw) if raw else None
                    if parsed is None:
                        rec = {
                            "wall_time": f"{wall:.6f}",
                            "wall_iso": datetime.fromtimestamp(wall).isoformat(
                                timespec="milliseconds"
                            ),
                            "elapsed_s": f"{wall - t0:.3f}",
                            "file_mtime_age_s": f"{mtime_age:.4f}",
                            "status": "BAD_SIZE",
                            "magic_hex": "",
                            "seq": "",
                            "stamp_ns": "",
                            "stamp_age_s": "",
                            "mu_mode": state.mu_mode,
                            "mu_mode_sim": (vel or {}).get("mu_mode", "") if vel else "",
                            "mu_slide": "",
                            "note": state.note,
                            "vx": "",
                            "vy": "",
                            "vz": "",
                            "vxy": "",
                            "abs_vx": "",
                            "abs_vy": "",
                            "pos_y": "",
                            "contact_L": "",
                            "contact_R": "",
                            "normal_scaled_L": "",
                            "normal_scaled_R": "",
                            "tangent_scaled_L": "",
                            "tangent_scaled_R": "",
                            "Fn_N_L": "",
                            "Fn_N_R": "",
                            "Ft_N_L": "",
                            "Ft_N_R": "",
                            "rho_L": "",
                            "rho_R": "",
                            "load_L": "",
                            "load_R": "",
                            "sensor_valid": "",
                            "sensor_age_norm": "",
                            "packet_bytes": len(raw),
                            "_fn_mean": 0.0,
                            "_ft_mean": 0.0,
                            "_rho_mean": 0.0,
                            "_status": "BAD_SIZE",
                            "_vx": float("nan"),
                            "_vy": float("nan"),
                            "_vxy": float("nan"),
                        }
                    else:
                        rec = enrich(parsed, wall, t0, mtime_age, state, vel=vel)

                writer.writerow(rec)
                f_csv.flush()
                if f_json is not None:
                    dump = {k: rec[k] for k in CSV_FIELDS if k in rec}
                    f_json.write(json.dumps(dump, ensure_ascii=False) + "\n")
                    f_json.flush()

                update_summary(state, rec)

                if not args.no_console and (wall - last_print) >= print_every:
                    last_print = wall
                    if rec["_status"] == "NO_FILE":
                        line = "NO_FILE  start MuJoCo with G1_MUJOCO_FOOT_BRIDGE=1"
                    elif rec["_status"] == "BAD_SIZE":
                        line = f"BAD_SIZE bytes={rec.get('packet_bytes')}"
                    else:
                        vx_s = rec.get("vx") or "-"
                        vy_s = rec.get("vy") or "-"
                        line = (
                            f"{rec['status']:10s} seq={str(rec['seq']):<7} "
                            f"mu={state.mu_mode:<9} "
                            f"vx={vx_s:>6} vy={vy_s:>6} "
                            f"Fn={rec['Fn_N_L']:>6}/{rec['Fn_N_R']:<6} N  "
                            f"Ft={rec['Ft_N_L']:>5}/{rec['Ft_N_R']:<5} N  "
                            f"ρ={rec['rho_L']}/{rec['rho_R']}"
                        )
                    sys.stdout.write("\r\033[K" + line)
                    sys.stdout.flush()

                # sleep remaining
                spent = time.time() - wall
                time.sleep(max(0.0, dt - spent))
    except KeyboardInterrupt:
        print("\n[info] KeyboardInterrupt", flush=True)
    finally:
        if f_json is not None:
            f_json.close()
        write_summary(summary_path, state, t0, csv_path)
        print(f"\nsaved: {csv_path}", flush=True)
        print(f"summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
