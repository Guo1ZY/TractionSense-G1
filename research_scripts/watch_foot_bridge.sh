#!/usr/bin/env bash
# Live-print /tmp/g1_foot_rl_obs.bin (g1_ctrl foot obs bridge).
# Usage: watch_foot_bridge.sh [Hz]   default 5 Hz
set -euo pipefail
HZ="${1:-5}"
python3 - "$HZ" <<'PY'
import struct, pathlib, sys, time

hz = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
dt = 1.0 / max(hz, 0.5)
path = pathlib.Path("/tmp/g1_foot_rl_obs.bin")
MAGIC = 0x46305431

print(f"watching {path}  @ {1.0/dt:.1f} Hz   Ctrl+C to quit\n")
try:
    while True:
        line = ""
        if not path.exists():
            line = "NO FILE  /tmp/g1_foot_rl_obs.bin  (start MuJoCo with G1_MUJOCO_FOOT_BRIDGE=1)"
        else:
            d = path.read_bytes()
            age = time.time() - path.stat().st_mtime
            if len(d) < 40:
                line = f"BAD size={len(d)} age={age:.2f}s"
            else:
                magic, seq = struct.unpack_from("<II", d, 0)
                c = struct.unpack_from("<ff", d, 16)
                n = struct.unpack_from("<ff", d, 24)  # already *0.01
                t = struct.unpack_from("<ff", d, 32)
                ok = "OK" if magic == MAGIC and age < 0.25 else ("STALE" if age >= 0.25 else "BAD_MAGIC")
                fn = (n[0] / 0.01, n[1] / 0.01)
                ft = (t[0] / 0.01, t[1] / 0.01)
                rho = (
                    ft[0] / (fn[0] + 1.0),
                    ft[1] / (fn[1] + 1.0),
                )
                line = (
                    f"{ok} seq={seq:<7d} age={age:4.2f}s | "
                    f"c L/R={c[0]:.2f}/{c[1]:.2f}  "
                    f"Fn={fn[0]:6.0f}/{fn[1]:6.0f} N  "
                    f"Ft={ft[0]:5.0f}/{ft[1]:5.0f} N  "
                    f"ρ={rho[0]:.2f}/{rho[1]:.2f}"
                )
        # single-line refresh
        sys.stdout.write("\r\033[K" + line)
        sys.stdout.flush()
        time.sleep(dt)
except KeyboardInterrupt:
    print("\nbye")
PY
