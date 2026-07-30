#!/usr/bin/env python3
"""Automated MuJoCo Sim2Sim smoke for foot_mu MuAdapt policy.

- Starts unitree_mujoco (pty) with G1_MUJOCO_FOOT_BRIDGE=1
- Starts g1_ctrl (pty) with keyboard velocity (deploy must use keyboard_velocity_commands)
- Sends a → x → hold w
- Switches friction via MuJoCo stdin: 3=GRIP then 1=ICE
- Runs log_foot_bridge in parallel

Does NOT touch a real robot (network lo only).
"""
from __future__ import annotations

import os
import pty
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(
    os.environ.get("TRACTIONSENSE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
WORKSPACE = Path(
    os.environ.get("TRACTIONSENSE_WORKSPACE", ROOT.parent)
).resolve()
MUJOCO = Path(
    os.environ.get("UNITREE_MUJOCO_ROOT", WORKSPACE / "unitree_mujoco")
) / "simulate/build/unitree_mujoco"
G1 = ROOT / "deploy/robots/g1_29dof/build/g1_ctrl"
LOGGER = ROOT / "research_scripts/log_foot_bridge.py"
LOG_DIR = ROOT / "logs/foot_bridge"


def start_pty(cmd, cwd, env, log_path: Path):
    master, slave = pty.openpty()
    logf = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        preexec_fn=os.setsid,
    )
    os.close(slave)
    return proc, master, logf


def drain(master, logf, timeout=0.1):
    end = time.time() + timeout
    while time.time() < end:
        r, _, _ = select.select([master], [], [], 0.05)
        if master not in r:
            continue
        try:
            chunk = os.read(master, 8192)
        except OSError:
            break
        if not chunk:
            break
        try:
            logf.write(chunk.decode("utf-8", "replace"))
        except Exception:
            pass


def kill_proc(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    time.sleep(1.0)
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()


def main():
    # stop leftovers (sim only)
    subprocess.run(["pkill", "-f", "unitree_mujoco"], check=False)
    subprocess.run(["pkill", "-f", "./g1_ctrl --network lo"], check=False)
    subprocess.run(["pkill", "-f", "log_foot_bridge.py"], check=False)
    time.sleep(1.0)

    if not MUJOCO.is_file() or not G1.is_file():
        print("ERROR: missing binary", MUJOCO, G1)
        return 1

    # ensure scene normal (do NOT call run_mujoco_friction.sh — it execs and blocks)
    cfg = Path(
        os.environ.get("UNITREE_MUJOCO_ROOT", WORKSPACE / "unitree_mujoco")
    ) / "simulate/config.yaml"
    if cfg.is_file():
        lines = []
        for line in cfg.read_text().splitlines():
            if line.startswith("robot_scene:"):
                lines.append('robot_scene: "scene_29dof_normal.xml"')
            else:
                lines.append(line)
        cfg.write_text("\n".join(lines) + "\n")
        print("  robot_scene → scene_29dof_normal.xml")

    env_m = os.environ.copy()
    env_m["G1_MUJOCO_FOOT_BRIDGE"] = "1"
    env_m["DISPLAY"] = env_m.get("DISPLAY", ":1")

    env_g = os.environ.copy()
    env_g["LD_LIBRARY_PATH"] = (
        "/usr/local/lib:"
        + str(ROOT / "deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib")
        + ":"
        + env_g.get("LD_LIBRARY_PATH", "")
    )
    env_g["G1_CMD_GAIN_LIN"] = "1.0"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    mujoco_log = Path("/tmp/auto_mujoco.log")
    g1_log = Path("/tmp/auto_g1.log")

    print("[1] start unitree_mujoco")
    m_proc, m_master, m_log = start_pty(
        [str(MUJOCO)], MUJOCO.parent, env_m, mujoco_log
    )
    time.sleep(3.0)
    drain(m_master, m_log, 0.5)
    print("  mujoco pid", m_proc.pid, "alive", m_proc.poll() is None)

    print("[2] start g1_ctrl --network lo")
    g_proc, g_master, g_log = start_pty(
        [str(G1), "--network", "lo"],
        ROOT / "deploy/robots/g1_29dof/build",
        env_g,
        g1_log,
    )
    time.sleep(3.0)
    drain(g_master, g_log, 0.5)
    print("  g1 pid", g_proc.pid, "alive", g_proc.poll() is None)

    print("[3] start logger 95s")
    log_proc = subprocess.Popen(
        [
            sys.executable,
            str(LOGGER),
            "--tag",
            "muadapt_auto",
            "--hz",
            "20",
            "--mu-mode",
            "default",
            "--duration",
            "95",
            "--no-console",
        ],
        stdout=open("/tmp/auto_logger.out", "w"),
        stderr=subprocess.STDOUT,
    )

    def send_g(b: bytes, note: str):
        os.write(g_master, b)
        print("  g1 <-", note)
        drain(g_master, g_log, 0.2)

    def send_m(b: bytes, note: str):
        os.write(m_master, b)
        print("  mujoco <-", note)
        drain(m_master, m_log, 0.2)

    time.sleep(1.0)
    send_g(b"a", "A stand")
    time.sleep(3.0)
    send_g(b"x", "X velocity")
    time.sleep(1.0)

    # Phase default / normal walk
    print("[4] walk default μ ~20s (hold w)")
    t0 = time.time()
    while time.time() - t0 < 20 and g_proc.poll() is None:
        os.write(g_master, b"w")
        drain(g_master, g_log, 0.05)
        drain(m_master, m_log, 0.02)
        time.sleep(0.12)

    # GRIP
    print("[5] GRIP (key 3) + walk 25s")
    # retag logger via writing note file? logger is separate — user tags via stdin.
    # Write a marker file for post analysis by time
    Path("/tmp/auto_phase_grip_t0").write_text(str(time.time()))
    send_m(b"3\n", "friction GRIP")
    t0 = time.time()
    while time.time() - t0 < 25 and g_proc.poll() is None:
        os.write(g_master, b"w")
        drain(g_master, g_log, 0.05)
        drain(m_master, m_log, 0.02)
        time.sleep(0.12)

    # ICE
    print("[6] ICE (key 1) + walk 25s")
    Path("/tmp/auto_phase_ice_t0").write_text(str(time.time()))
    send_m(b"1\n", "friction ICE")
    t0 = time.time()
    while time.time() - t0 < 25 and g_proc.poll() is None:
        os.write(g_master, b"w")
        drain(g_master, g_log, 0.05)
        drain(m_master, m_log, 0.02)
        time.sleep(0.12)

    print("[7] stop")
    kill_proc(g_proc)
    kill_proc(m_proc)
    try:
        log_proc.wait(timeout=15)
    except Exception:
        log_proc.kill()

    m_log.close()
    g_log.close()
    print("logs: /tmp/auto_mujoco.log /tmp/auto_g1.log /tmp/auto_logger.out")
    print("csv under", LOG_DIR)
    # list latest csv
    csvs = sorted(LOG_DIR.glob("foot_*_muadapt_auto.csv"), key=lambda p: p.stat().st_mtime)
    if csvs:
        print("csv:", csvs[-1])
        summ = Path(str(csvs[-1]).replace(".csv", "_summary.txt"))
        if summ.is_file():
            print(summ.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
