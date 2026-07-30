#!/usr/bin/env python3
"""Drive mujoco + g1_ctrl PTYs — user hang-band workflow.

User sequence:
  1) MuJoCo: 8 (or L) lower robot toward ground
  2) g1: A stand, then X velocity
  3) MuJoCo: 9 release rope
  4) hold w + friction 3/1

Requires enable_elastic_band=1 in unitree_mujoco config.yaml
and rebuilt unitree_mujoco with stdin 7/8/9/L support.
"""
from __future__ import annotations

import os
import pty
import select
import signal
import struct
import subprocess
import sys
time = __import__("time")
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
ORT = ROOT / "deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib"
LOGGER = ROOT / "research_scripts/log_foot_bridge.py"
BRIDGE = Path("/tmp/g1_foot_rl_obs.bin")


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


def drain(master, logf, timeout=0.05):
    end = time.time() + timeout
    while time.time() < end:
        r, _, _ = select.select([master], [], [], 0.02)
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


def killpg(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    time.sleep(0.8)
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()


def foot_fn_sum():
    if not BRIDGE.is_file() or BRIDGE.stat().st_size < 40:
        return 0.0
    d = BRIDGE.read_bytes()
    nL, nR = struct.unpack_from("<ff", d, 24)
    return (abs(nL) + abs(nR)) / 0.01


def main():
    stand = float(os.environ.get("STAND_SEC", "6"))
    settle = float(os.environ.get("SETTLE_SEC", "3"))
    warm = float(os.environ.get("WARM_SEC", "12"))
    grip = float(os.environ.get("GRIP_SEC", "20"))
    ice = float(os.environ.get("ICE_SEC", "20"))
    # How many "8" lower steps (0.1 each). Default 22 → length≈2.2
    lower_n = int(os.environ.get("LOWER_STEPS", "22"))
    tag = os.environ.get("TAG", "oneclick")
    total = stand + settle + warm + grip + ice + 15

    env_m = os.environ.copy()
    env_m["G1_MUJOCO_FOOT_BRIDGE"] = "1"
    env_m["DISPLAY"] = env_m.get("DISPLAY", ":1")

    env_g = os.environ.copy()
    env_g["LD_LIBRARY_PATH"] = f"/usr/local/lib:{ORT}:{env_g.get('LD_LIBRARY_PATH', '')}"
    env_g["G1_CMD_GAIN_LIN"] = "1.0"
    # Loosen fall-to-Passive check for MuJoCo bring-up (default 1.0 if unset)
    env_g.setdefault("G1_BAD_ORI_LIMIT", "1.8")

    m_proc, m_fd, m_log = start_pty(
        [str(MUJOCO)], MUJOCO.parent, env_m, Path("/tmp/oneclick_mujoco.log")
    )
    print(f"[driver] mujoco pid={m_proc.pid}", flush=True)
    time.sleep(3.5)
    drain(m_fd, m_log, 0.5)

    g_proc, g_fd, g_log = start_pty(
        [str(G1), "--network", "lo"],
        ROOT / "deploy/robots/g1_29dof/build",
        env_g,
        Path("/tmp/oneclick_g1.log"),
    )
    print(f"[driver] g1_ctrl pid={g_proc.pid}", flush=True)
    time.sleep(3.0)
    drain(g_fd, g_log, 0.5)

    log_proc = subprocess.Popen(
        [
            sys.executable,
            str(LOGGER),
            "--tag",
            tag,
            "--hz",
            "20",
            "--mu-mode",
            "default",
            "--duration",
            str(int(total + 5)),
            "--no-console",
        ],
        stdout=open("/tmp/oneclick_logger.out", "w"),
        stderr=subprocess.STDOUT,
    )
    print(f"[driver] logger pid={log_proc.pid} tag={tag}", flush=True)

    def g_key(b: bytes, note: str):
        os.write(g_fd, b)
        print(f"[driver] g1 <- {note}", flush=True)
        drain(g_fd, g_log, 0.15)

    def m_cmd(s: str, note: str):
        os.write(m_fd, (s if s.endswith("\n") else s + "\n").encode())
        print(f"[driver] mujoco <- {note}", flush=True)
        drain(m_fd, m_log, 0.2)

    # ---- User workflow: 8 lower → A → X → 9 release ----
    print(f"[driver] step1: lower with key 8 x{lower_n} (or L one-shot)", flush=True)
    # One-shot L first (fast), then a few 8 for fine tune
    m_cmd("L", "L=lower near ground length=2.2")
    time.sleep(1.5)
    for i in range(max(0, lower_n - 22)):
        m_cmd("8", f"8 lower step {i+1}")
        time.sleep(0.08)
    time.sleep(1.0)

    print("[driver] step2: A stand", flush=True)
    g_key(b"a", "a=stand")
    t_end = time.time() + stand
    while time.time() < t_end:
        drain(g_fd, g_log, 0.05)
        drain(m_fd, m_log, 0.02)
        time.sleep(0.05)

    # wait contact while still on band
    peak = 0.0
    for _ in range(60):
        fn = foot_fn_sum()
        peak = max(peak, fn)
        if fn > 100:
            print(f"[driver] contact while hanging/standing Fn≈{fn:.0f}N", flush=True)
            break
        time.sleep(0.1)
    else:
        print(f"[driver] WARN Fn peak≈{peak:.0f}N after A (may still be hanging high)", flush=True)

    print(f"[driver] settle {settle}s", flush=True)
    t_end = time.time() + settle
    while time.time() < t_end:
        drain(g_fd, g_log, 0.05)
        drain(m_fd, m_log, 0.02)
        time.sleep(0.05)

    print("[driver] step3: X velocity (policy)", flush=True)
    g_key(b"x", "x=velocity")
    time.sleep(0.8)
    drain(g_fd, g_log, 0.5)
    try:
        gtxt = Path("/tmp/oneclick_g1.log").read_text(errors="replace")
        if "from Velocity to Passive" in gtxt:
            print("[driver] WARN: Velocity→Passive right after X", flush=True)
    except Exception:
        pass

    print("[driver] step4: 9 release rope", flush=True)
    m_cmd("9", "9=release hang band")
    time.sleep(1.5)
    # if still band-on (toggle started ON), one more 9 to ensure OFF
    # band default enable_=true; first 9 turns OFF. Good.
    for _ in range(30):
        fn = foot_fn_sum()
        if fn > 120:
            print(f"[driver] after release Fn≈{fn:.0f}N", flush=True)
            break
        time.sleep(0.1)

    def walk_phase(sec: float, label: str, fric_key=None):
        if fric_key is not None:
            m_cmd(fric_key, f"friction {fric_key}")
        Path("/tmp/oneclick_phase.txt").write_text(f"{label} {time.time()}\n")
        t0 = time.time()
        last_w = 0.0
        while time.time() - t0 < sec:
            if g_proc.poll() is not None:
                print("[driver] g1_ctrl exited early", flush=True)
                break
            now = time.time()
            if now - last_w > 0.12:
                os.write(g_fd, b"w")
                last_w = now
            drain(g_fd, g_log, 0.03)
            drain(m_fd, m_log, 0.02)
            time.sleep(0.05)

    print(f"[driver] warm walk {warm}s", flush=True)
    walk_phase(warm, "default", None)

    print(f"[driver] GRIP walk {grip}s", flush=True)
    walk_phase(grip, "grip", "3")

    print(f"[driver] ICE walk {ice}s", flush=True)
    walk_phase(ice, "ice", "1")

    print("[driver] stopping...", flush=True)
    killpg(g_proc)
    killpg(m_proc)
    try:
        log_proc.wait(timeout=20)
    except Exception:
        log_proc.kill()
    m_log.close()
    g_log.close()
    print("[driver] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
