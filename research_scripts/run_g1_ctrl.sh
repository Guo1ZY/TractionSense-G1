#!/usr/bin/env bash
# Launch the restored G1-29DoF controller with explicit Sim/real safety gates.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_DIR="$ROOT/deploy/robots/g1_29dof"
BIN="$ROBOT_DIR/build/g1_ctrl"
POLICY="$ROBOT_DIR/config/policy/velocity/traction_student_7989/exported/policy.onnx"
BRIDGE="${G1_FOOT_BRIDGE_PATH:-/tmp/g1_foot_rl_obs.bin}"
NETWORK="lo"
MODE="sim"
HARNESS=0
ESTOP=0
DRY_RUN=0
SPEED_LIMIT="1.0"
SPEED_LIMIT_SET=0

usage() {
  echo "Usage:"
  echo "  $0 --sim [--network lo]"
  echo "  $0 --real IFACE --harness-confirmed --estop-confirmed [--speed-limit 0.05]"
  echo "  $0 --dry-run"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sim) MODE="sim"; NETWORK="lo"; shift ;;
    --real) MODE="real"; NETWORK="$2"; shift 2 ;;
    --network) NETWORK="$2"; shift 2 ;;
    --harness-confirmed) HARNESS=1; shift ;;
    --estop-confirmed) ESTOP=1; shift ;;
    --speed-limit) SPEED_LIMIT="$2"; SPEED_LIMIT_SET=1; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$MODE" == "real" && "$SPEED_LIMIT_SET" == "0" ]]; then
  SPEED_LIMIT="0.05"
fi
python3 - "$SPEED_LIMIT" <<'PY'
import math
import sys
try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit("[ERROR] --speed-limit must be numeric")
if not math.isfinite(value) or value <= 0.0 or value > 1.0:
    raise SystemExit("[ERROR] --speed-limit must be in (0, 1.0]")
PY

[[ -x "$BIN" ]] || {
  echo "[ERROR] missing g1_ctrl: $BIN" >&2
  exit 1
}
[[ -f "$POLICY" ]] || {
  echo "[ERROR] missing traction Student ONNX: $POLICY" >&2
  exit 1
}
[[ -d "/sys/class/net/$NETWORK" ]] || {
  echo "[ERROR] network interface does not exist: $NETWORK" >&2
  exit 1
}

if [[ "$MODE" == "real" ]]; then
  [[ "$NETWORK" != "lo" ]] || {
    echo "[ERROR] real mode cannot use loopback" >&2
    exit 1
  }
  [[ "$HARNESS" == "1" && "$ESTOP" == "1" ]] || {
    echo "[BLOCKED] real mode requires --harness-confirmed and --estop-confirmed" >&2
    exit 4
  }
  [[ "$(cat "/sys/class/net/$NETWORK/operstate")" == "up" ]] || {
    echo "[BLOCKED] $NETWORK is not UP" >&2
    exit 4
  }
  if [[ -f "/sys/class/net/$NETWORK/carrier" ]]; then
    [[ "$(cat "/sys/class/net/$NETWORK/carrier")" == "1" ]] || {
      echo "[BLOCKED] $NETWORK has no Ethernet carrier" >&2
      exit 4
    }
  fi
  [[ -f "$BRIDGE" ]] || {
    echo "[BLOCKED] foot bridge packet is missing: $BRIDGE" >&2
    exit 4
  }
  bridge_age="$(python3 - "$BRIDGE" <<'PY'
import sys, time
from pathlib import Path
print(time.time() - Path(sys.argv[1]).stat().st_mtime)
PY
)"
  python3 - "$bridge_age" <<'PY'
import sys
age = float(sys.argv[1])
if age > 0.25:
    raise SystemExit(f"[BLOCKED] foot bridge is stale: {age:.3f}s")
PY
fi

export G1_FOOT_BRIDGE_PATH="$BRIDGE"
export G1_CMD_FORWARD_LIMIT="$SPEED_LIMIT"
export G1_CMD_SLEW_LIN="${G1_CMD_SLEW_LIN:-0.35}"
export LD_LIBRARY_PATH="$ROOT/.unitree_sdk2/lib:$ROOT/.cpp_deps/lib:$ROOT/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "mode=$MODE network=$NETWORK policy=traction_student_7989"
echo "forward_limit=${SPEED_LIMIT}m/s (absolute maximum 1.0m/s) slew=$G1_CMD_SLEW_LIN bridge=$BRIDGE"
echo "Controller always starts in Passive. A=stand, X=velocity, B=passive."

if [[ "$DRY_RUN" == "1" ]]; then
  "$BIN" --version
  echo "DRY_RUN PASS"
  exit 0
fi

cd "$ROBOT_DIR"
exec "$BIN" --network "$NETWORK"
