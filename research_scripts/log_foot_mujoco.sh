#!/usr/bin/env bash
# Log foot-bridge while testing MuJoCo friction (companion to run_mujoco_friction.sh).
#
# Usage:
#   # Terminal 1 — MuJoCo (writes /tmp/g1_foot_rl_obs.bin)
#   export G1_MUJOCO_FOOT_BRIDGE=1
#   ./research_scripts/run_mujoco_friction.sh normal
#
#   # Terminal 2 — this logger
#   ./research_scripts/log_foot_mujoco.sh
#   ./research_scripts/log_foot_mujoco.sh --tag full_10900 --hz 20 --mu-mode normal
#   ./research_scripts/log_foot_mujoco.sh --duration 60 --tag ice_vs_grip
#
# During logging, type + Enter:
#   1 / ice | 2 / normal | 3 / grip | 4 / ultra | note xxx | q
# (MuJoCo physics keys are still pressed in the MuJoCo terminal; retag here for CSV.)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/research_scripts/log_foot_bridge.py"
export G1_FOOT_LOG_DIR="${G1_FOOT_LOG_DIR:-$ROOT/logs/foot_bridge}"
export G1_FOOT_BRIDGE_PATH="${G1_FOOT_BRIDGE_PATH:-/tmp/g1_foot_rl_obs.bin}"

if [[ ! -f "$SCRIPT" ]]; then
  echo "[ERROR] missing $SCRIPT"
  exit 1
fi

mkdir -p "$G1_FOOT_LOG_DIR"
echo "[info] logs → $G1_FOOT_LOG_DIR"
echo "[info] bridge → $G1_FOOT_BRIDGE_PATH"
echo "[info] ensure MuJoCo started with: export G1_MUJOCO_FOOT_BRIDGE=1"
echo

exec python3 "$SCRIPT" "$@"
