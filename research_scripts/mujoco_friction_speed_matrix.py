#!/usr/bin/env python3
"""One-click latest-policy MuJoCo friction x forward-speed validation.

This is simulation-only: g1_ctrl is forced onto DDS loopback (``--network lo``).
It exports the newest TractionAdaptive checkpoint, starts MuJoCo and g1_ctrl,
then measures an exact friction/speed matrix and writes CSV + Markdown reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pty
import re
import select
import signal
import statistics
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


LAB = Path(
    os.environ.get("TRACTIONSENSE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
WORKSPACE = Path(
    os.environ.get("TRACTIONSENSE_WORKSPACE", LAB.parent)
).resolve()
MUJOCO_DIR = Path(
    os.environ.get("UNITREE_MUJOCO_ROOT", WORKSPACE / "unitree_mujoco")
) / "simulate"
MUJOCO_BUILD = MUJOCO_DIR / "build"
MUJOCO_BIN = MUJOCO_BUILD / "unitree_mujoco"
G1_DIR = LAB / "deploy/robots/g1_29dof"
G1_BUILD = G1_DIR / "build"
G1_BIN = G1_BUILD / "g1_ctrl"
G1_CONFIG = G1_DIR / "config/config.yaml"
MUJOCO_CONFIG = MUJOCO_DIR / "config.yaml"
EXPORT_SCRIPT = LAB / "research_scripts/export_g1_foot_onnx.sh"
ADAPTIVE_EXPERIMENT_ROOT = (
    LAB / "logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_adaptive"
)
TEACHER_EXPERIMENT_ROOT = (
    LAB / "logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_teacher"
)
ADAPTIVE_TASK = "Unitree-G1-29dof-Velocity-Foot-TractionAdaptive"
TEACHER_TASK = "Unitree-G1-29dof-Velocity-Foot-TractionTeacher"
MUJOCO_COMMAND = Path("/tmp/mujoco_rl_cmd")
VELOCITY_COMMAND = Path("/tmp/g1_rl_velocity_cmd")
BASE_VELOCITY = Path("/tmp/g1_base_vel.json")
FOOT_BRIDGE = Path("/tmp/g1_foot_rl_obs.bin")
FOOT_MAGIC_F0T1 = 0x46305431
FOOT_MAGIC_F0T2 = 0x46305432
FOOT_MAGIC_F0M1 = 0x46304D31
POLICY_OBS_MAGIC = 0x3153424F


@dataclass
class Sample:
    wall_time: float
    mu: float
    cmd_vx: float
    vx: float
    vy: float
    vxy: float
    x: float
    y: float
    z: float
    fn: float
    ft: float
    rho: float
    sensor_valid: float


class PtyLog:
    def __init__(self, process: subprocess.Popen, master: int, path: Path):
        self.process = process
        self.master = master
        self.path = path
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        with self.path.open("wb") as output:
            while not self._stop.is_set():
                ready, _, _ = select.select([self.master], [], [], 0.1)
                if self.master not in ready:
                    if self.process.poll() is not None:
                        break
                    continue
                try:
                    chunk = os.read(self.master, 8192)
                except OSError:
                    break
                if not chunk:
                    break
                output.write(chunk)
                output.flush()

    def send(self, key: str) -> None:
        os.write(self.master, key.encode("ascii"))

    def size(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def text_since(self, offset: int) -> str:
        try:
            with self.path.open("rb") as stream:
                stream.seek(offset)
                return stream.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            os.close(self.master)
        except OSError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-click MuJoCo friction x forward-speed policy matrix"
    )
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint; default: newest healthy model")
    parser.add_argument(
        "--profile",
        choices=["adaptive", "teacher"],
        default="adaptive",
        help="Policy/task profile; teacher enables the Oracle 641st mu observation",
    )
    parser.add_argument("--slot", help="Temporary deploy policy slot; defaults from --profile")
    parser.add_argument(
        "--task",
        help="Override Isaac Lab task used for export (needed by observation-compatible continuations)",
    )
    parser.add_argument(
        "--motion-feedback",
        action="store_true",
        help="Feed exact MuJoCo body-vy plus IMU-relative yaw to the 641-D Motion Teacher",
    )
    parser.add_argument(
        "--motion-vy-gain",
        type=float,
        default=1.0,
        help="Deployment feedback gain applied to body-vy (motion policies only)",
    )
    parser.add_argument(
        "--motion-heading-gain",
        type=float,
        default=1.0,
        help="Deployment feedback gain applied to relative heading (motion policies only)",
    )
    parser.add_argument("--device", default="cuda:0", help="Isaac export device")
    parser.add_argument(
        "--estimator",
        type=Path,
        help="Optional 640-D friction-estimator ONNX; replaces Teacher oracle mu at runtime",
    )
    parser.add_argument(
        "--lateral-estimator",
        type=Path,
        help="Optional 1862-D ONNX body-vy estimator for the magnetic motion policy",
    )
    parser.add_argument(
        "--mus",
        nargs="+",
        type=float,
        default=[0.08, 0.20, 0.40, 0.80, 1.20],
        help="Exact MuJoCo floor sliding friction values",
    )
    parser.add_argument(
        "--switch-sequence",
        nargs="+",
        type=float,
        help=(
            "Run one continuous episode with one unchanged speed command, "
            "for example: --switch-sequence 1.2 0.15 1.2"
        ),
    )
    parser.add_argument(
        "--switch-phase-sec",
        type=float,
        default=6.0,
        help="Duration of each continuous friction phase",
    )
    parser.add_argument(
        "--switch-settle-sec",
        type=float,
        default=1.0,
        help="Initial portion excluded from each phase's steady metrics",
    )
    parser.add_argument(
        "--speeds",
        nargs="+",
        type=float,
        default=[0.1, 0.5, 1.0],
        help="Forward commands in m/s; hard-capped at 1.0",
    )
    parser.add_argument("--stand-sec", type=float, default=8.0)
    parser.add_argument("--warmup-sec", type=float, default=2.0)
    parser.add_argument("--measure-sec", type=float, default=6.0)
    parser.add_argument("--sample-hz", type=float, default=20.0)
    parser.add_argument(
        "--record-frames-dir", type=Path,
        help="Optional PNG output directory; recording requires exactly one matrix cell",
    )
    parser.add_argument("--record-fps", type=float, default=20.0)
    parser.add_argument("--skip-export", action="store_true", help="Use ONNX already installed in --slot")
    parser.add_argument("--skip-build", action="store_true", help="Do not rebuild MuJoCo/g1_ctrl")
    parser.add_argument(
        "--magnetic-bridge",
        action="store_true",
        help="Publish the 400-byte F0M1 dual-foot 15xXYZ packet for magnetic policies",
    )
    parser.add_argument(
        "--disable-command-slew",
        action="store_true",
        help="Remove deploy-time velocity slew so compared slots receive the same step command",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast 2-friction x 2-speed integration test",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 3 when model gates fail (default: a completed report exits 0)",
    )
    parser.add_argument("--output-dir", type=Path, help="Custom result directory")
    parser.add_argument(
        "--report-only",
        type=Path,
        metavar="MATRIX_CSV",
        help="Recompute reports from an existing matrix without starting simulators",
    )
    return parser.parse_args()


def latest_checkpoint(experiment_root: Path) -> Path:
    candidates = sorted(
        experiment_root.glob("*/model_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        try:
            size_a = candidate.stat().st_size
            time.sleep(0.05)
            size_b = candidate.stat().st_size
        except OSError:
            continue
        if size_a >= 1_000_000 and size_a == size_b:
            return candidate.resolve()
    raise FileNotFoundError(f"no healthy checkpoint under {experiment_root}")


def run_checked(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def export_policy(checkpoint: Path, slot: str, device: str, task: str) -> Path:
    run_checked(
        [
            str(EXPORT_SCRIPT),
            "--checkpoint",
            str(checkpoint),
            "--dest",
            slot,
            "--task",
            task,
            "--device",
            device,
        ],
        cwd=LAB,
    )
    slot_dir = G1_DIR / "config/policy/velocity" / slot
    onnx = slot_dir / "exported/policy.onnx"
    deploy = slot_dir / "params/deploy.yaml"
    if not onnx.is_file() or not deploy.is_file():
        raise FileNotFoundError(f"incomplete exported policy slot: {slot_dir}")
    (slot_dir / "checkpoint.txt").write_text(str(checkpoint) + "\n", encoding="utf-8")
    return slot_dir


def build_binaries() -> None:
    run_checked(["cmake", "--build", str(MUJOCO_BUILD), "-j", "4"])
    run_checked(["cmake", "--build", str(G1_BUILD), "-j", "4"])
    if not MUJOCO_BIN.is_file() or not G1_BIN.is_file():
        raise FileNotFoundError("MuJoCo or g1_ctrl binary missing after build")


def replace_yaml_scalar(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^(\s*{re.escape(key)}\s*:).*$")
    if pattern.search(text):
        return pattern.sub(rf"\1 {value}", text, count=1)
    return text + f"\n{key}: {value}\n"


def configure_mujoco(text: str) -> str:
    text = replace_yaml_scalar(text, "robot", '"g1"')
    text = replace_yaml_scalar(text, "robot_scene", '"scene_29dof.xml"')
    text = replace_yaml_scalar(text, "domain_id", "0")
    text = replace_yaml_scalar(text, "interface", '"lo"')
    text = replace_yaml_scalar(text, "enable_elastic_band", "1")
    text = replace_yaml_scalar(text, "use_joystick", "0")
    text = replace_yaml_scalar(text, "print_scene_information", "0")
    return text


def configure_g1(text: str, slot: str) -> str:
    pattern = re.compile(r"(?m)^(\s*policy_dir:\s*)config/policy/velocity/[^\s#]+(\s*)$")
    updated, count = pattern.subn(rf"\1config/policy/velocity/{slot}\2", text, count=1)
    if count != 1:
        raise RuntimeError("could not select Velocity policy_dir in config.yaml")
    return updated


def configure_deploy(text: str, max_speed: float, disable_command_slew: bool = False) -> str:
    if disable_command_slew:
        text = re.sub(r"(?m)^\s+slew_rate:\s*\{[^\n]*\}\s*\n", "", text, count=1)
    # `G1_CMD_FILE` is consumed by the automation-only observation registered
    # as `keyboard_velocity_commands`.  A normal deployment intentionally uses
    # `velocity_commands` (gamepad), so switch only the temporary YAML used by
    # this MuJoCo harness; the original file is restored in `finally`.
    text, command_observation_count = re.subn(
        r"(?m)^(\s{2})velocity_commands:(\s*)$",
        r"\1keyboard_velocity_commands:\2",
        text,
        count=1,
    )
    if command_observation_count != 1:
        raise RuntimeError("deploy.yaml has no velocity command observation")
    if not re.search(r"(?m)^\s{2}keyboard_velocity_commands:\s*$", text):
        raise RuntimeError("failed to enable exact file-driven velocity commands")
    text, count = re.subn(
        r"(?m)^(\s*lin_vel_x:\s*)\[[^\]]+\](\s*)$",
        rf"\1[-0.5, {max_speed:.3f}]\2",
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("deploy.yaml has no lin_vel_x range")
    return text


def atomic_text(path: Path, value: str) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def terminate(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except OSError:
        process.terminate()
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except OSError:
            process.kill()


def stop_old_sim_stack() -> None:
    subprocess.run(["pkill", "-x", "unitree_mujoco"], check=False)
    subprocess.run(
        ["pkill", "-f", "/deploy/robots/g1_29dof/build/g1_ctrl"],
        check=False,
    )
    time.sleep(0.7)


def start_g1(env: dict[str, str], log_path: Path) -> PtyLog:
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [str(G1_BIN), "--network", "lo"],
        cwd=G1_BUILD,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        start_new_session=True,
    )
    os.close(slave)
    return PtyLog(process, master, log_path)


def wait_fresh(path: Path, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"process exited early with rc={process.returncode}")
        try:
            if path.is_file() and time.time() - path.stat().st_mtime < 0.5:
                return
        except OSError:
            pass
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for fresh {path}")


def mujoco_command(command: str, process: subprocess.Popen) -> None:
    atomic_text(MUJOCO_COMMAND, command + "\n")
    deadline = time.time() + 2.0
    while MUJOCO_COMMAND.exists() and time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("MuJoCo exited while handling command")
        time.sleep(0.05)
    if MUJOCO_COMMAND.exists():
        raise TimeoutError(f"MuJoCo did not consume command: {command}")


def velocity_command(vx: float) -> None:
    atomic_text(VELOCITY_COMMAND, f"{vx:.6f} 0.0 0.0\n")


def read_velocity() -> dict | None:
    try:
        data = json.loads(BASE_VELOCITY.read_text(encoding="utf-8"))
        stamp = float(data.get("stamp_ns", 0)) * 1e-9
        if stamp <= 0 or abs(time.time() - stamp) > 0.5:
            return None
        return data
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def read_foot() -> tuple[float, float, float, float]:
    try:
        data = FOOT_BRIDGE.read_bytes()
    except OSError:
        return math.nan, math.nan, math.nan, 0.0
    if len(data) < 40:
        return math.nan, math.nan, math.nan, 0.0
    magic = struct.unpack_from("<I", data, 0)[0]
    if magic == FOOT_MAGIC_F0T1 and len(data) >= 40:
        n_left, n_right = struct.unpack_from("<ff", data, 24)
        t_left, t_right = struct.unpack_from("<ff", data, 32)
        # Bridge channels are force[N] * 0.01; undo that scale when reporting
        # the sum across the two feet.
        fn = 100.0 * (abs(n_left) + abs(n_right))
        ft = 100.0 * (abs(t_left) + abs(t_right))
        return fn, ft, ft / (fn + 1.0), 1.0
    if magic == FOOT_MAGIC_F0T2 and len(data) >= 48:
        n_left, n_right = struct.unpack_from("<ff", data, 24)
        valid = struct.unpack_from("<f", data, 40)[0]
        fn = 100.0 * (abs(n_left) + abs(n_right))
        return fn, 0.0, 0.0, float(valid)
    if magic == FOOT_MAGIC_F0M1 and len(data) >= 400:
        valid_left, valid_right = struct.unpack_from("<ff", data, 16)
        magnetic = np.asarray(struct.unpack_from("<90f", data, 40), dtype=np.float64)
        magnetic = magnetic.reshape(2, 15, 3).mean(axis=1)
        # Invert the deterministic MuJoCo Hall proxy for reporting only.  The
        # controller still receives exclusively the raw normalized XYZ array.
        signal = 5.0 * np.arctanh(np.clip(magnetic / 5.0, -0.999, 0.999))
        response = np.asarray(
            [[0.14, 1.00], [-0.10, 0.42], [1.00, 0.12]],
            dtype=np.float64,
        )
        forces = np.stack(
            [np.linalg.lstsq(response, foot, rcond=None)[0] for foot in signal]
        )
        forces = np.maximum(forces, 0.0)
        fn = 100.0 * float(forces[:, 0].sum())
        ft = 100.0 * float(forces[:, 1].sum())
        return fn, ft, ft / (fn + 1.0), float(min(valid_left, valid_right))
    return math.nan, math.nan, math.nan, 0.0


def write_labeled_policy_observations(
    raw_path: Path,
    windows: list[tuple[float, float, float, float]],
    output_path: Path,
) -> int:
    """Convert C++ OBS1 records into deploy-observation NPZ labels.

    Windows are ``(start_wall, end_wall, mu, cmd_vx)`` and deliberately cover
    warm-up plus measurement, while excluding stand/reset transients.
    """
    try:
        payload = raw_path.read_bytes()
    except OSError:
        return 0
    observations: list[np.ndarray] = []
    labels_mu: list[float] = []
    labels_cmd: list[float] = []
    stamps: list[float] = []
    offset = 0
    while offset + 16 <= len(payload):
        magic, dim, stamp_ns = struct.unpack_from("<IIQ", payload, offset)
        offset += 16
        byte_count = int(dim) * 4
        if magic != POLICY_OBS_MAGIC or dim <= 0 or offset + byte_count > len(payload):
            break
        stamp = stamp_ns * 1e-9
        values = np.frombuffer(payload, dtype="<f4", count=dim, offset=offset).copy()
        offset += byte_count
        for start, end, mu, cmd_vx in windows:
            if start <= stamp <= end:
                observations.append(values)
                labels_mu.append(mu)
                labels_cmd.append(cmd_vx)
                stamps.append(stamp)
                break
    if not observations:
        return 0
    np.savez_compressed(
        output_path,
        obs=np.stack(observations).astype(np.float32),
        mu=np.asarray(labels_mu, dtype=np.float32),
        cmd_vx=np.asarray(labels_cmd, dtype=np.float32),
        wall_time=np.asarray(stamps, dtype=np.float64),
    )
    return len(observations)


def sample_phase(
    mu: float,
    cmd_vx: float,
    duration: float,
    sample_hz: float,
    mujoco_process: subprocess.Popen,
    g1: PtyLog,
) -> tuple[list[Sample], bool]:
    samples: list[Sample] = []
    start_log = g1.size()
    deadline = time.time() + duration
    dt = 1.0 / max(1.0, sample_hz)
    while time.time() < deadline:
        if mujoco_process.poll() is not None or g1.process.poll() is not None:
            return samples, True
        velocity = read_velocity()
        if velocity is not None:
            fn, ft, rho, sensor_valid = read_foot()
            samples.append(
                Sample(
                    wall_time=time.time(),
                    mu=mu,
                    cmd_vx=cmd_vx,
                    vx=float(velocity.get("vx", math.nan)),
                    vy=float(velocity.get("vy", math.nan)),
                    vxy=float(velocity.get("vxy", math.nan)),
                    x=float(velocity.get("x", math.nan)),
                    y=float(velocity.get("y", math.nan)),
                    z=float(velocity.get("z", math.nan)),
                    fn=fn,
                    ft=ft,
                    rho=rho,
                    sensor_valid=sensor_valid,
                )
            )
        time.sleep(dt)
    fsm_fall = "from Velocity to Passive" in g1.text_since(start_log)
    height_fall = any(sample.z == sample.z and sample.z < 0.45 for sample in samples)
    return samples, fsm_fall or height_fall


def mean_finite(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else math.nan


def stdev_finite(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.pstdev(finite) if len(finite) >= 2 else math.nan


def summarize_case(samples: list[Sample], mu: float, cmd_vx: float, fall: bool) -> dict:
    valid = [sample for sample in samples if math.isfinite(sample.vx)]
    ys = [sample.y for sample in valid if math.isfinite(sample.y)]
    xs = [sample.x for sample in valid if math.isfinite(sample.x)]
    zs = [sample.z for sample in valid if math.isfinite(sample.z)]
    mean_vx = mean_finite([sample.vx for sample in valid])
    mean_abs_vy = mean_finite([abs(sample.vy) for sample in valid])
    min_z = min(zs, default=math.nan)
    lateral_drift = abs(ys[-1] - ys[0]) if len(ys) >= 2 else math.nan
    forward_distance = xs[-1] - xs[0] if len(xs) >= 2 else math.nan
    coverage_ok = len(valid) >= 5
    stable = (
        coverage_ok
        and not fall
        and math.isfinite(min_z)
        and min_z >= 0.45
        and math.isfinite(mean_abs_vy)
        and mean_abs_vy <= 0.30
    )
    return {
        "mu": mu,
        "cmd_vx": cmd_vx,
        "mean_vx": mean_vx,
        "std_vx": stdev_finite([sample.vx for sample in valid]),
        "mean_abs_vy": mean_abs_vy,
        "lateral_drift": lateral_drift,
        "forward_distance": forward_distance,
        "mean_base_z": mean_finite(zs),
        "min_base_z": min_z,
        "mean_fn": mean_finite([sample.fn for sample in valid]),
        "mean_ft": mean_finite([sample.ft for sample in valid]),
        "mean_rho": mean_finite([sample.rho for sample in valid]),
        "sensor_valid": mean_finite([sample.sensor_valid for sample in valid]),
        "samples": len(valid),
        "fall": int(fall),
        "stable": "PASS" if stable else "FAIL",
    }


def prepare_episode(mu: float, mujoco_process: subprocess.Popen, g1: PtyLog, stand_sec: float) -> bool:
    velocity_command(0.0)
    g1.send("b")
    time.sleep(0.3)
    # Engage the safety band before reset. Most importantly, enter FixStand
    # immediately after reset; leaving the free-floating robot passive for even
    # one second lets it fall before the controller can stand it up.
    mujoco_command("L", mujoco_process)
    # At the 3 m anchor, length=1.9 only contributes roughly 60 N at a
    # standing pelvis height and therefore does not actually suspend G1.
    # length=0.4 contributes about 360 N, enough to keep the reset posture
    # upright while FixStand takes ownership of all joints.
    for _ in range(18):
        mujoco_command("7", mujoco_process)
    mujoco_command("reset_stand", mujoco_process)
    time.sleep(0.05)
    start_log = g1.size()
    g1.send("a")
    # Bring-up always uses normal grip. Applying ice before FixStand makes the
    # startup maneuver itself slip, which confounds policy evaluation.
    mujoco_command("mu 0.800000", mujoco_process)
    time.sleep(stand_sec)
    velocity = read_velocity()
    suspended_z = float(velocity.get("z", 0.0)) if velocity is not None else 0.0
    print(f"    bring-up suspended z={suspended_z:.3f}", flush=True)
    if suspended_z < 0.55:
        return False

    # Let the learned controller own the joints at zero speed before removing
    # support. FixStand alone is an open-loop pose and can tip over while its
    # load changes; the policy is the component trained to actively balance.
    g1.send("x")
    time.sleep(1.5)
    if "from Velocity to Passive" in g1.text_since(start_log):
        return False

    # Gradually unload the support instead of dropping the full robot weight
    # onto the feet in one physics step. 0.4 -> 2.2 m in 0.1 m increments.
    for _ in range(18):
        mujoco_command("8", mujoco_process)
        time.sleep(0.15)
    time.sleep(0.5)
    # Give FixStand a short extension when the base has not yet reached a
    # usable upright height. This is a startup check, not part of measurement.
    height_deadline = time.time() + 3.0
    velocity = read_velocity()
    while (
        velocity is not None
        and float(velocity.get("z", 0.0)) < 0.55
        and time.time() < height_deadline
    ):
        time.sleep(0.2)
        velocity = read_velocity()
    landed_z = float(velocity.get("z", 0.0)) if velocity is not None else 0.0
    print(f"    bring-up landed z={landed_z:.3f}", flush=True)
    if landed_z < 0.55:
        return False
    # The policy now owns the joints at zero command; switch to the target
    # floor friction only after this stable hand-off.
    mujoco_command(f"mu {mu:.6f}", mujoco_process)
    time.sleep(0.5)
    mujoco_command("9", mujoco_process)
    time.sleep(1.0)
    return True


def write_reports(
    output_dir: Path,
    checkpoint: Path | None,
    rows: list[dict],
    samples: list[Sample],
) -> tuple[Path, str]:
    matrix_path = output_dir / "matrix.csv"
    sample_path = output_dir / "samples.csv"
    fields = list(rows[0])
    with matrix_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with sample_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(Sample.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(sample.__dict__ for sample in samples)

    max_cmd = max(float(row["cmd_vx"]) for row in rows)
    stress = [row for row in rows if abs(float(row["cmd_vx"]) - max_cmd) < 1e-6]
    # Match the Isaac acceptance protocol and the stated requirements: judge
    # the low/high endpoints, while still reporting every intermediate cell.
    # Averaging μ=0.8 and μ=1.2 hid whether the actual high-friction endpoint
    # met its gate and made the verdict depend on which intermediate bins were
    # included in a run.
    low_mu = min(float(row["mu"]) for row in stress)
    high_mu = max(float(row["mu"]) for row in stress)
    low = [row for row in stress if abs(float(row["mu"]) - low_mu) < 1.0e-6]
    high = [row for row in stress if abs(float(row["mu"]) - high_mu) < 1.0e-6]

    def average(group: list[dict], key: str) -> float:
        return mean_finite([float(row[key]) for row in group])

    governor_enabled = os.environ.get("G1_TRACTION_GOVERNOR", "0").lower() in {
        "1",
        "true",
        "on",
    }
    low_limit = (
        float(os.environ.get("G1_TRACTION_LOW_SPEED", max_cmd))
        if governor_enabled
        else max_cmd
    )
    high_limit = (
        float(os.environ.get("G1_TRACTION_HIGH_SPEED", max_cmd))
        if governor_enabled
        else max_cmd
    )
    effective_low_command = min(abs(max_cmd), abs(low_limit))
    effective_high_command = min(abs(max_cmd), abs(high_limit))
    speed_delta_target = 0.65 * max(
        effective_high_command - effective_low_command, 0.0
    )

    gates: list[tuple[str, float, bool, str]] = []
    no_falls = sum(int(row["fall"]) for row in rows)
    gates.append(("全矩阵摔倒次数", float(no_falls), no_falls == 0, "= 0"))
    if low and high:
        low_vx = average(low, "mean_vx")
        high_vx = average(high, "mean_vx")
        gates.append(
            (
                "最高指令高低摩擦速度差",
                high_vx - low_vx,
                high_vx - low_vx >= speed_delta_target,
                f">= 65% cap gap ({speed_delta_target:.2f} m/s)",
            )
        )
        gates.append(
            (
                "最高指令低摩擦限速",
                low_vx,
                low_vx <= 1.25 * effective_low_command,
                f"<= 125% low cap ({1.25 * effective_low_command:.2f} m/s)",
            )
        )
        high_speed_target = 0.80 * effective_high_command
        gates.append(
            (
                "最高指令高摩擦速度",
                high_vx,
                high_vx >= high_speed_target,
                f">= 80% high cap ({high_speed_target:.2f} m/s)",
            )
        )
        gates.append(("最高指令高摩擦横向速度", average(high, "mean_abs_vy"), average(high, "mean_abs_vy") <= 0.25, "<= 0.25 m/s"))
    stable_cells = sum(row["stable"] == "PASS" for row in rows)
    gates.append(("稳定单元格比例", stable_cells / len(rows), stable_cells == len(rows), "= 1.00"))
    overall = "PASS" if all(gate[2] for gate in gates) else "NEEDS_WORK"

    lines = [
        "# MuJoCo friction × speed validation",
        "",
        f"- Checkpoint: `{checkpoint or 'preinstalled slot'}`",
        f"- Overall: **{overall}**",
        (
            f"- Effective governor caps: low `{effective_low_command:.3f}`, "
            f"high `{effective_high_command:.3f} m/s`"
            if governor_enabled
            else "- Governor: disabled"
        ),
        "- Network: `lo` (simulation only)",
        "",
        "## Matrix",
        "",
        "| μ | cmd vx | mean vx | σ(vx) | |vy| | lateral drift | min z | Fn | Ft | fall | stable |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mu']:.2f} | {row['cmd_vx']:.2f} | {row['mean_vx']:.3f} | "
            f"{row['std_vx']:.3f} | {row['mean_abs_vy']:.3f} | {row['lateral_drift']:.3f} | "
            f"{row['min_base_z']:.3f} | {row['mean_fn']:.1f} | {row['mean_ft']:.1f} | "
            f"{'YES' if row['fall'] else 'NO'} | {row['stable']} |"
        )
    lines += [
        "",
        "## Gates",
        "",
        "| Gate | Value | Result | Target |",
        "|---|---:|:---:|---:|",
    ]
    for name, value, passed, target in gates:
        lines.append(f"| {name} | {value:.3f} | {'PASS' if passed else 'FAIL'} | {target} |")
    lines.append("")
    report_path = output_dir / "summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path, overall


def switch_response_time(
    samples: list[Sample],
    phase_start: float,
    previous_steady: float,
    current_steady: float,
    sample_hz: float,
) -> float:
    """Time to complete 80% of a speed transition with a 0.2-s dwell."""

    values = np.asarray(
        [sample.vx for sample in samples if math.isfinite(sample.vx)],
        dtype=np.float64,
    )
    if values.size == 0:
        return math.nan
    delta = current_steady - previous_steady
    if abs(delta) < 0.05:
        return 0.0
    window = max(int(round(0.10 * sample_hz)), 1)
    dwell = max(int(round(0.20 * sample_hz)), 1)
    if values.size < window:
        smoothed = values
    else:
        smoothed = np.convolve(
            values, np.ones(window, dtype=np.float64) / window, mode="valid"
        )
    boundary = previous_steady + 0.80 * delta
    reached = smoothed >= boundary if delta > 0.0 else smoothed <= boundary
    dt = 1.0 / max(sample_hz, 1.0)
    for index in range(len(reached)):
        if index + dwell <= len(reached) and bool(
            np.all(reached[index : index + dwell])
        ):
            return (index + window) * dt
    return math.nan


def write_switch_reports(
    output_dir: Path,
    checkpoint: Path,
    sequence: list[float],
    command_vx: float,
    phase_rows: list[dict],
    phase_samples: list[list[Sample]],
    phase_starts: list[float],
    sample_hz: float,
    magnetic_bridge: bool,
    max_response_s: float = 3.0,
) -> tuple[Path, str]:
    """Write continuous high->low->high phase and time-series reports."""

    previous_steady = math.nan
    for index, row in enumerate(phase_rows):
        response = (
            math.nan
            if index == 0
            else switch_response_time(
                phase_samples[index],
                phase_starts[index],
                previous_steady,
                float(row["steady_vx"]),
                sample_hz,
            )
        )
        row["response_time_s"] = response
        previous_steady = float(row["steady_vx"])

    phase_path = output_dir / "switch_phases.csv"
    with phase_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(phase_rows[0]))
        writer.writeheader()
        writer.writerows(phase_rows)

    timeseries_path = output_dir / "switch_timeseries.csv"
    time_fields = [
        "phase",
        "time_since_switch_s",
        *Sample.__dataclass_fields__,
    ]
    with timeseries_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=time_fields)
        writer.writeheader()
        for phase, (samples, start) in enumerate(zip(phase_samples, phase_starts)):
            for sample in samples:
                writer.writerow(
                    {
                        "phase": phase,
                        "time_since_switch_s": sample.wall_time - start,
                        **sample.__dict__,
                    }
                )

    low_rows = [row for row in phase_rows if float(row["mu"]) <= 0.25]
    high_rows = [row for row in phase_rows if float(row["mu"]) >= 0.75]

    def average(rows: list[dict], key: str) -> float:
        return mean_finite([float(row[key]) for row in rows])

    falls = sum(int(row["fall"]) for row in phase_rows)
    responses = [
        float(row["response_time_s"])
        for row in phase_rows[1:]
        if math.isfinite(float(row["response_time_s"]))
    ]
    low_vx = average(low_rows, "steady_vx")
    high_vx = average(high_rows, "steady_vx")
    governor_enabled = os.environ.get("G1_TRACTION_GOVERNOR", "0").lower() in {
        "1",
        "true",
        "on",
    }
    low_limit = (
        float(os.environ.get("G1_TRACTION_LOW_SPEED", command_vx))
        if governor_enabled
        else command_vx
    )
    high_limit = (
        float(os.environ.get("G1_TRACTION_HIGH_SPEED", command_vx))
        if governor_enabled
        else command_vx
    )
    effective_low_command = min(abs(command_vx), abs(low_limit))
    effective_high_command = min(abs(command_vx), abs(high_limit))
    high_target = 0.80 * effective_high_command
    # A speed difference larger than the two governor commands is impossible.
    # Require the measured separation to retain at least 65% of the achievable
    # command separation.
    speed_delta_target = 0.65 * max(
        effective_high_command - effective_low_command, 0.0
    )
    gates = [
        ("全程摔倒次数", float(falls), falls == 0, "= 0"),
        (
            "高低摩擦稳态速度差",
            high_vx - low_vx,
            high_vx - low_vx >= speed_delta_target,
            f">= 65% cap gap ({speed_delta_target:.2f} m/s)",
        ),
        (
            "低摩擦稳态限速",
            low_vx,
            low_vx <= 1.25 * effective_low_command,
            f"<= 125% low cap ({1.25 * effective_low_command:.2f} m/s)",
        ),
        (
            "高摩擦速度恢复",
            high_vx,
            high_vx >= high_target,
            f">= 80% high cap ({high_target:.2f} m/s)",
        ),
        (
            "全部切换均测得响应",
            float(len(responses)),
            len(responses) == len(phase_rows) - 1,
            f"= {len(phase_rows) - 1}",
        ),
        (
            "最大切换响应时间",
            max(responses, default=math.inf),
            bool(responses) and max(responses) <= max_response_s,
            f"<= {max_response_s:.2f} s",
        ),
        (
            "全部阶段稳定",
            float(sum(row["stable"] == "PASS" for row in phase_rows))
            / len(phase_rows),
            all(row["stable"] == "PASS" for row in phase_rows),
            "= 1.00",
        ),
    ]
    overall = "PASS" if all(gate[2] for gate in gates) else "NEEDS_WORK"
    lines = [
        "# MuJoCo continuous friction-switch validation",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Overall: **{overall}**",
        f"- Command: `{command_vx:.3f} m/s` (unchanged)",
        (
            f"- Effective governor caps: low `{effective_low_command:.3f}`, "
            f"high `{effective_high_command:.3f} m/s`"
            if governor_enabled
            else "- Governor: disabled"
        ),
        f"- Sequence: `{sequence}`",
        "- Magnetic bridge: "
        + ("enabled" if magnetic_bridge else "disabled (sensor ablation)"),
        "- Network: `lo` (simulation only)",
        "- Scope: speed, lateral drift, force/slip proxy and falls; the current "
        "MuJoCo bridge does not expose reliable touchdown timing, so cadence "
        "and step length remain Isaac/real-camera metrics.",
        "",
        "## Per-phase behavior",
        "",
        "| phase | μ | vx | |vy| | rho early | rho steady | response | falls | stable |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in phase_rows:
        response = float(row["response_time_s"])
        response_text = f"{response:.3f}" if math.isfinite(response) else "n/a"
        lines.append(
            f"| {int(row['phase'])} | {float(row['mu']):.2f} | "
            f"{float(row['steady_vx']):.3f} | "
            f"{float(row['steady_abs_vy']):.3f} | "
            f"{float(row['early_rho']):.3f} | "
            f"{float(row['steady_rho']):.3f} | {response_text} | "
            f"{int(row['fall'])} | {row['stable']} |"
        )
    lines += [
        "",
        "## Gates",
        "",
        "| Gate | Value | Result | Target |",
        "|---|---:|:---:|---:|",
    ]
    for name, value, passed, target in gates:
        lines.append(
            f"| {name} | {value:.3f} | "
            f"{'PASS' if passed else 'FAIL'} | {target} |"
        )
    lines.append("")
    report_path = output_dir / "switch_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path, overall


def run_continuous_switch(
    args: argparse.Namespace,
    checkpoint: Path,
    output_dir: Path,
    policy_obs_raw: Path,
    mujoco_process: subprocess.Popen,
    g1: PtyLog,
) -> tuple[Path, str, int]:
    """Run one episode while only floor friction changes."""

    sequence = [float(value) for value in args.switch_sequence]
    command_vx = float(args.speeds[0])
    if not prepare_episode(sequence[0], mujoco_process, g1, args.stand_sec):
        raise RuntimeError("could not reach upright Velocity state for switch trial")
    velocity_command(0.0)
    time.sleep(0.7)
    velocity_command(command_vx)
    print(f"  initial command warmup={args.warmup_sec:.1f}s", flush=True)
    warm_samples, warm_fall = sample_phase(
        sequence[0],
        command_vx,
        args.warmup_sec,
        args.sample_hz,
        mujoco_process,
        g1,
    )
    if warm_fall:
        raise RuntimeError("policy fell during initial command warmup")

    phase_rows: list[dict] = []
    all_samples: list[list[Sample]] = []
    phase_starts: list[float] = []
    case_windows: list[tuple[float, float, float, float]] = []
    if args.record_frames_dir is not None:
        mujoco_command("record_start", mujoco_process)
    for phase, mu in enumerate(sequence):
        if phase > 0:
            print(f"  switch μ {sequence[phase - 1]:.2f} -> {mu:.2f}", flush=True)
            mujoco_command(f"mu {mu:.6f}", mujoco_process)
        start = time.time()
        samples, fall = sample_phase(
            mu,
            command_vx,
            args.switch_phase_sec,
            args.sample_hz,
            mujoco_process,
            g1,
        )
        end = time.time()
        steady_samples = [
            sample
            for sample in samples
            if sample.wall_time - start >= args.switch_settle_sec
        ]
        row = summarize_case(steady_samples, mu, command_vx, fall)
        early = [
            sample.rho
            for sample in samples
            if sample.wall_time - start <= min(1.0, args.switch_phase_sec)
        ]
        row = {
            "phase": phase,
            "mu": mu,
            "cmd_vx": command_vx,
            "steady_vx": row["mean_vx"],
            "steady_abs_vy": row["mean_abs_vy"],
            "early_rho": mean_finite(early),
            "steady_rho": row["mean_rho"],
            "min_base_z": row["min_base_z"],
            "lateral_drift": row["lateral_drift"],
            "fall": row["fall"],
            "stable": row["stable"],
        }
        phase_rows.append(row)
        all_samples.append(samples)
        phase_starts.append(start)
        case_windows.append((start, end, mu, command_vx))
        print(
            f"    phase={phase} μ={mu:.2f} vx={row['steady_vx']:.3f} "
            f"|vy|={row['steady_abs_vy']:.3f} rho={row['steady_rho']:.3f} "
            f"fall={row['fall']} {row['stable']}",
            flush=True,
        )
        if fall:
            break
    if args.record_frames_dir is not None:
        mujoco_command("record_stop", mujoco_process)

    report, overall = write_switch_reports(
        output_dir,
        checkpoint,
        sequence[: len(phase_rows)],
        command_vx,
        phase_rows,
        all_samples,
        phase_starts,
        args.sample_hz,
        args.magnetic_bridge,
    )
    obs_count = write_labeled_policy_observations(
        policy_obs_raw, case_windows, output_dir / "policy_obs.npz"
    )
    return report, overall, obs_count


def main() -> int:
    args = parse_args()
    teacher = args.profile == "teacher"
    task = args.task or (TEACHER_TASK if teacher else ADAPTIVE_TASK)
    experiment_root = TEACHER_EXPERIMENT_ROOT if teacher else ADAPTIVE_EXPERIMENT_ROOT
    if args.report_only is not None:
        matrix_path = args.report_only.expanduser().resolve()
        if not matrix_path.is_file():
            print(f"[ERROR] matrix not found: {matrix_path}", file=sys.stderr)
            return 2
        output_dir = (
            args.output_dir.expanduser().resolve() if args.output_dir else matrix_path.parent
        )
        checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
        with matrix_path.open(newline="", encoding="utf-8") as stream:
            rows = []
            for raw in csv.DictReader(stream):
                row = {}
                for key, value in raw.items():
                    if key == "stable":
                        row[key] = value
                    elif key in {"fall", "samples"}:
                        row[key] = int(value)
                    else:
                        row[key] = float(value)
                rows.append(row)
        if not rows:
            print(f"[ERROR] empty matrix: {matrix_path}", file=sys.stderr)
            return 2
        samples = []
        sample_path = matrix_path.with_name("samples.csv")
        if sample_path.is_file():
            with sample_path.open(newline="", encoding="utf-8") as stream:
                samples = [
                    Sample(**{key: float(value) for key, value in raw.items()})
                    for raw in csv.DictReader(stream)
                ]
        report, overall = write_reports(output_dir, checkpoint, rows, samples)
        print(f"[DONE] {overall}: {report}")
        return 3 if args.strict and overall != "PASS" else 0
    slot = args.slot or ("traction_teacher" if teacher else "traction_adaptive")
    if args.estimator is not None:
        args.estimator = args.estimator.expanduser().resolve()
        if not teacher:
            print("[ERROR] --estimator currently requires --profile teacher", file=sys.stderr)
            return 2
    if args.lateral_estimator is not None:
        args.lateral_estimator = args.lateral_estimator.expanduser().resolve()
        if not args.lateral_estimator.is_file():
            print(
                f"[ERROR] lateral estimator not found: {args.lateral_estimator}",
                file=sys.stderr,
            )
            return 2
    if not 0.0 <= args.motion_vy_gain <= 4.0:
        print("[ERROR] --motion-vy-gain must be in [0, 4]", file=sys.stderr)
        return 2
    if not 0.0 <= args.motion_heading_gain <= 4.0:
        print("[ERROR] --motion-heading-gain must be in [0, 4]", file=sys.stderr)
        return 2
        if not args.estimator.is_file():
            print(f"[ERROR] estimator not found: {args.estimator}", file=sys.stderr)
            return 2
    if args.smoke:
        args.mus = [0.08, 1.20]
        args.speeds = [0.1, 1.0]
        args.warmup_sec = min(args.warmup_sec, 1.0)
        args.measure_sec = min(args.measure_sec, 3.0)
    if (
        not args.mus
        or not args.speeds
        or any(mu < 0.01 or mu > 3.0 for mu in args.mus)
        or any(speed < 0.0 or speed > 1.0 for speed in args.speeds)
    ):
        print("[ERROR] invalid friction/speed matrix", file=sys.stderr)
        return 2
    if args.switch_sequence is not None:
        if (
            len(args.switch_sequence) < 2
            or len(args.speeds) != 1
            or any(mu < 0.01 or mu > 3.0 for mu in args.switch_sequence)
            or args.switch_phase_sec <= 0.0
            or not 0.0 <= args.switch_settle_sec < args.switch_phase_sec
        ):
            print(
                "[ERROR] switch mode requires >=2 valid friction values, "
                "exactly one speed, and 0 <= settle < phase duration",
                file=sys.stderr,
            )
            return 2
    if args.record_frames_dir is not None:
        args.record_frames_dir = args.record_frames_dir.expanduser().resolve()
        recording_cases = (
            1
            if args.switch_sequence is not None and len(args.speeds) == 1
            else len(args.mus) * len(args.speeds)
        )
        if recording_cases != 1:
            print(
                "[ERROR] frame recording requires one matrix cell or one "
                "continuous switch trial",
                file=sys.stderr,
            )
            return 2
        if not 1.0 <= args.record_fps <= 60.0:
            print("[ERROR] --record-fps must be in [1, 60]", file=sys.stderr)
            return 2
        args.record_frames_dir.mkdir(parents=True, exist_ok=True)
    slot_dir = G1_DIR / "config/policy/velocity" / slot
    if args.checkpoint:
        checkpoint = args.checkpoint.expanduser().resolve()
    elif args.skip_export and (slot_dir / "checkpoint.txt").is_file():
        checkpoint = Path(
            (slot_dir / "checkpoint.txt").read_text(encoding="utf-8").strip()
        ).expanduser().resolve()
    else:
        checkpoint = latest_checkpoint(experiment_root)
    if not checkpoint.is_file():
        print(f"[ERROR] checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = checkpoint.stem
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else LAB
        / "logs/evaluations"
        / ("mujoco_traction_teacher" if teacher else "mujoco_traction")
        / f"{stamp}_{model_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print(f" MuJoCo {args.profile} friction × speed validation")
    print(f" task       : {task}")
    print(f" checkpoint : {checkpoint}")
    friction_plan = (
        args.switch_sequence
        if args.switch_sequence is not None
        else args.mus
    )
    print(f" mus        : {friction_plan}")
    print(f" speeds     : {args.speeds}")
    print(f" output     : {output_dir}")
    friction_source = (
        str(args.estimator)
        if args.estimator
        else (
            "Oracle true mu"
            if teacher
            else (
                "causal magnetic observation (no true mu)"
                if args.magnetic_bridge
                else "magnetic sensor disabled/invalid"
            )
        )
    )
    print(f" estimator  : {friction_source}")
    print(" network    : lo (NEVER real robot)")
    print("=" * 68, flush=True)

    if not args.skip_export:
        slot_dir = export_policy(checkpoint, slot, args.device, task)
    elif not (slot_dir / "exported/policy.onnx").is_file():
        print(f"[ERROR] --skip-export slot missing ONNX: {slot_dir}", file=sys.stderr)
        return 1
    if not args.skip_build:
        build_binaries()

    deploy_path = slot_dir / "params/deploy.yaml"
    backups = {
        MUJOCO_CONFIG: MUJOCO_CONFIG.read_bytes(),
        G1_CONFIG: G1_CONFIG.read_bytes(),
        deploy_path: deploy_path.read_bytes(),
    }
    mujoco_process: subprocess.Popen | None = None
    g1: PtyLog | None = None
    all_samples: list[Sample] = []
    rows: list[dict] = []
    case_windows: list[tuple[float, float, float, float]] = []
    policy_obs_raw = output_dir / "policy_obs.bin"
    try:
        MUJOCO_CONFIG.write_text(
            configure_mujoco(backups[MUJOCO_CONFIG].decode()), encoding="utf-8"
        )
        G1_CONFIG.write_text(
            configure_g1(backups[G1_CONFIG].decode(), slot), encoding="utf-8"
        )
        deploy_path.write_text(
            configure_deploy(
                backups[deploy_path].decode(),
                max(args.speeds),
                disable_command_slew=args.disable_command_slew,
            ),
            encoding="utf-8",
        )

        stop_old_sim_stack()
        for path in (MUJOCO_COMMAND, VELOCITY_COMMAND, BASE_VELOCITY, FOOT_BRIDGE):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        velocity_command(0.0)

        env_mujoco = os.environ.copy()
        env_mujoco["DISPLAY"] = env_mujoco.get("DISPLAY", ":1")
        # Automated matrices never need a window.  The headless loop also
        # allows MuJoCo validation to run while Isaac training occupies X11.
        env_mujoco["G1_MUJOCO_HEADLESS"] = "1"
        env_mujoco["G1_MUJOCO_FOOT_BRIDGE"] = "1"
        if args.magnetic_bridge:
            env_mujoco["G1_MUJOCO_MAGNETIC_BRIDGE"] = "1"
        if args.record_frames_dir is not None:
            env_mujoco["G1_MUJOCO_RECORD_DIR"] = str(args.record_frames_dir)
            env_mujoco["G1_MUJOCO_RECORD_FPS"] = f"{args.record_fps:.3f}"
        mujoco_log = (output_dir / "mujoco.log").open("wb")
        mujoco_process = subprocess.Popen(
            [str(MUJOCO_BIN)],
            cwd=MUJOCO_BUILD,
            stdin=subprocess.DEVNULL,
            stdout=mujoco_log,
            stderr=subprocess.STDOUT,
            env=env_mujoco,
            start_new_session=True,
        )
        wait_fresh(BASE_VELOCITY, mujoco_process, 15.0)

        env_g1 = os.environ.copy()
        ort = LAB / "deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib"
        local_libs = [
            LAB / ".unitree_sdk2/lib",
            LAB / ".cpp_deps/lib",
            ort,
        ]
        env_g1["LD_LIBRARY_PATH"] = ":".join(
            [str(path) for path in local_libs] + [env_g1.get("LD_LIBRARY_PATH", "")]
        )
        env_g1["G1_MUJOCO_FOOT_BRIDGE"] = "1"
        env_g1["G1_FOOT_BRIDGE_PATH"] = str(FOOT_BRIDGE)
        if teacher:
            env_g1["G1_FRICTION_ORACLE_PATH"] = "/tmp/g1_ground_mu"
        if args.motion_feedback:
            env_g1["G1_MOTION_FEEDBACK_PATH"] = str(BASE_VELOCITY)
        env_g1["G1_MOTION_VY_GAIN"] = f"{args.motion_vy_gain:.6f}"
        env_g1["G1_MOTION_HEADING_GAIN"] = f"{args.motion_heading_gain:.6f}"
        if args.estimator is not None:
            env_g1["G1_FRICTION_ESTIMATOR_ONNX"] = str(args.estimator)
            env_g1["G1_FRICTION_ESTIMATOR_ALPHA"] = "0.20"
        if args.lateral_estimator is not None:
            env_g1["G1_LATERAL_VELOCITY_ESTIMATOR_ONNX"] = str(
                args.lateral_estimator
            )
            env_g1["G1_LATERAL_VELOCITY_ESTIMATOR_ALPHA"] = "0.35"
        # The current Isaac ContactSensor training data has zero tangent
        # channels. Preserve raw MuJoCo Ft in the packet/report, but reproduce
        # the actor's training schema at policy input until shear-aware V2 is
        # retrained with verified non-zero PhysX tangential forces.
        env_g1["G1_FOOT_TANGENT_SCALE"] = "0.0"
        env_g1["G1_CMD_FILE"] = str(VELOCITY_COMMAND)
        env_g1["G1_POLICY_OBS_FILE"] = str(policy_obs_raw)
        # MuJoCo validation detects falls from base height itself. Keep the FSM
        # from exiting early so a fall is measured instead of hidden as Passive.
        env_g1["G1_BAD_ORI_LIMIT"] = "3.0"
        g1 = start_g1(env_g1, output_dir / "g1_ctrl.log")
        time.sleep(5.0)
        if g1.process.poll() is not None:
            raise RuntimeError(f"g1_ctrl exited during startup rc={g1.process.returncode}")

        if args.switch_sequence is not None:
            report, overall, obs_count = run_continuous_switch(
                args,
                checkpoint,
                output_dir,
                policy_obs_raw,
                mujoco_process,
                g1,
            )
            print(f"[OBS] labeled deploy observations: {obs_count}", flush=True)
            print("\n" + report.read_text(encoding="utf-8"), flush=True)
            print(f"[DONE] {overall}: {report}")
            return 3 if args.strict and overall != "PASS" else 0

        for mu in args.mus:
            print(f"\n[μ={mu:.2f}]", flush=True)
            for cmd_vx in args.speeds:
                # Each matrix cell is an independent trial.  Reusing one
                # episode across all commands lets lateral/heading error from
                # earlier cells contaminate later cells and makes the result
                # depend on speed ordering rather than (mu, command) alone.
                print(f"  cmd={cmd_vx:.2f}: reset → stand → policy", flush=True)
                episode_ready = prepare_episode(mu, mujoco_process, g1, args.stand_sec)
                attempts = 1
                while not episode_ready and attempts < 3:
                    attempts += 1
                    print(f"  [retry {attempts}/3] episode not ready; resetting", flush=True)
                    episode_ready = prepare_episode(mu, mujoco_process, g1, args.stand_sec)
                if not episode_ready:
                    raise RuntimeError(
                        f"could not reach upright Velocity state at mu={mu:.3f} after 3 attempts"
                    )
                velocity_command(0.0)
                time.sleep(0.7)
                velocity_command(cmd_vx)
                if args.record_frames_dir is not None:
                    mujoco_command("record_start", mujoco_process)
                print(f"    warmup={args.warmup_sec:.1f}s", flush=True)
                case_start = time.time()
                warm_samples, warm_fall = sample_phase(
                    mu, cmd_vx, args.warmup_sec, args.sample_hz, mujoco_process, g1
                )
                samples, fall = sample_phase(
                    mu, cmd_vx, args.measure_sec, args.sample_hz, mujoco_process, g1
                )
                if args.record_frames_dir is not None:
                    mujoco_command("record_stop", mujoco_process)
                case_windows.append((case_start, time.time(), mu, cmd_vx))
                fall = fall or warm_fall
                all_samples.extend(samples)
                row = summarize_case(samples, mu, cmd_vx, fall)
                rows.append(row)
                print(
                    f"    vx={row['mean_vx']:.3f} |vy|={row['mean_abs_vy']:.3f} "
                    f"drift={row['lateral_drift']:.3f} min_z={row['min_base_z']:.3f} "
                    f"fall={row['fall']} {row['stable']}",
                    flush=True,
                )
                episode_ready = not fall
                if fall:
                    velocity_command(0.0)
                    g1.send("b")
                    time.sleep(0.7)
            velocity_command(0.0)
            g1.send("b")
            time.sleep(0.5)

        report, overall = write_reports(output_dir, checkpoint, rows, all_samples)
        obs_count = write_labeled_policy_observations(
            policy_obs_raw, case_windows, output_dir / "policy_obs.npz"
        )
        print(f"[OBS] labeled deploy observations: {obs_count}", flush=True)
        print("\n" + report.read_text(encoding="utf-8"), flush=True)
        print(f"[DONE] {overall}: {report}")
        return 3 if args.strict and overall != "PASS" else 0
    except Exception as error:
        (output_dir / "error.txt").write_text(repr(error) + "\n", encoding="utf-8")
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    finally:
        try:
            velocity_command(0.0)
        except OSError:
            pass
        if g1 is not None:
            terminate(g1.process)
            g1.close()
        terminate(mujoco_process)
        try:
            mujoco_log.close()
        except (NameError, OSError):
            pass
        for path, content in backups.items():
            try:
                path.write_bytes(content)
            except OSError as error:
                print(f"[WARN] could not restore {path}: {error}", file=sys.stderr)
        for path in (MUJOCO_COMMAND, VELOCITY_COMMAND):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
