#!/usr/bin/env bash
# Start zorn foot ROS → g1_ctrl observation bridge on the host.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRIDGE_PY="${BRIDGE_PY:-$ROOT/deploy/robots/g1_29dof/scripts/foot_ros_bridge.py}"
OUT="${G1_FOOT_BRIDGE_PATH:-/tmp/g1_foot_rl_obs.bin}"
DEMO=0

usage() {
  cat <<EOF
Usage: $0 [--demo] [--out PATH]

  Default: subscribe ROS2 /g1/{left,right}_foot/frame (zorn) and write IPC for g1_ctrl.
  --demo : synthetic forces (no ROS / no zorn needed)

Env:
  G1_FOOT_BRIDGE_PATH  IPC file (default /tmp/g1_foot_rl_obs.bin)
  ROS_DOMAIN_ID        default 0
EOF
}

EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --demo) DEMO=1; shift ;;
    --out) OUT="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

export G1_FOOT_BRIDGE_PATH="$OUT"

if [[ "$DEMO" == "1" ]]; then
  echo "[run_foot_ros_bridge] DEMO → $OUT"
  exec python3 "$BRIDGE_PY" --demo --out "$OUT" "${EXTRA[@]}"
fi

# ROS setup.bash references optional unset vars (e.g. AMENT_TRACE_SETUP_FILES).
# With `set -u` that aborts — disable nounset only while sourcing.
if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  set -u
elif [[ -f /opt/ros/humble/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  set -u
else
  echo "[WARN] No /opt/ros/{jazzy,humble}/setup.bash — trying system python rclpy"
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

echo "============================================================"
echo " Foot ROS → RL bridge"
echo "  script : $BRIDGE_PY"
echo "  out    : $OUT"
echo "  domain : $ROS_DOMAIN_ID"
echo "  topics : /g1/left_foot/frame  /g1/right_foot/frame"
echo "============================================================"
echo "Ensure zorn foot runtime is publishing, then start g1_ctrl with foot policy."
echo ""

exec python3 "$BRIDGE_PY" --out "$OUT" --print-hz 1 "${EXTRA[@]}"
