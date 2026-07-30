#!/usr/bin/env bash
# Full-stick speed stress test for foot_full (μ adapt check).
# Stick full → cmd up to deploy.yaml lin_vel_x max (currently 1.5).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CTRL="$ROOT/deploy/robots/g1_29dof"
export LD_LIBRARY_PATH="/usr/local/lib:${CTRL}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib:${LD_LIBRARY_PATH:-}"
# 1.0 = full stick hits yaml max; raise only if stick doesn't reach end travel
export G1_CMD_GAIN_LIN="${G1_CMD_GAIN_LIN:-1.0}"
export G1_CMD_GAIN_YAW="${G1_CMD_GAIN_YAW:-1.0}"
NET="${1:-lo}"
echo "[stress] network=$NET  G1_CMD_GAIN_LIN=$G1_CMD_GAIN_LIN"
echo "[stress] full stick → clamp to deploy lin_vel_x max (see foot/params/deploy.yaml)"
echo "[stress] MuJoCo: key 3=GRIP high μ, key 1=ICE low μ, V=print |v|"
cd "$CTRL"
exec ./run_g1_ctrl.sh --network "$NET"
