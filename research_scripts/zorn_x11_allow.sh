#!/usr/bin/env bash
# Allow Docker/root inside zorn to open GUI windows on the host X display.
# Run this on the **host** (not inside the container) once per login session.
set -euo pipefail

export DISPLAY="${DISPLAY:-:1}"

echo "DISPLAY=$DISPLAY"
# Allow all local clients (docker typically appears as local root)
xhost +local: >/dev/null
xhost +SI:localuser:root >/dev/null 2>&1 || true

echo "[OK] X11 access for local/root enabled."
echo "xhost:"
xhost
echo
echo "Now inside zorn you can re-run:"
echo "  bash /workspace/foot_sensor/scripts/g1fs_one_click_runtime.sh"
