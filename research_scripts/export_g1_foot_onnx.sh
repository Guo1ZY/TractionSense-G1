#!/usr/bin/env bash
# Export a foot fine-tune checkpoint to ONNX and install into g1_ctrl velocity policy slot.
#
# Usage:
#   CHECKPOINT=logs/rsl_rl/unitree_g1_29dof_velocity_foot/.../model_XXXX.pt $0
#   $0 --checkpoint /path/to/model.pt [--dest v0|foot]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB_DIR="${UNITREE_RL_LAB_DIR:-$ROOT}"
ISAACLAB_PATH="${ISAACLAB_PATH:-${HOME}/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-isaaclab-v2}"
TASK="${TASK:-Unitree-G1-29dof-Velocity-Foot}"
DEVICE="${DEVICE:-cuda:0}"
DEST_NAME="${DEST_NAME:-foot}"  # under deploy/.../config/policy/velocity/
NUM_ENVS="${NUM_ENVS:-32}"
HEADLESS="${HEADLESS:-1}"

CHECKPOINT="${CHECKPOINT:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint|-c) CHECKPOINT="$2"; shift 2 ;;
    --dest) DEST_NAME="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  # pick latest foot experiment checkpoint
  LATEST_DIR=$(ls -1dt "$LAB_DIR"/logs/rsl_rl/unitree_g1_29dof_velocity_foot/*/ 2>/dev/null | head -1 || true)
  if [[ -n "$LATEST_DIR" ]]; then
    CHECKPOINT=$(ls -1t "$LATEST_DIR"/model_*.pt 2>/dev/null | head -1 || true)
  fi
fi

if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] checkpoint not found. Set CHECKPOINT=... or --checkpoint"
  exit 1
fi

CHECKPOINT=$(realpath "$CHECKPOINT")
CKPT_DIR=$(dirname "$CHECKPOINT")
RUN_DIR=$(dirname "$CKPT_DIR")
# rsl-rl layout: run_dir/model_x.pt OR run_dir/ is log_dir with model files
if [[ "$(basename "$CKPT_DIR")" == "exported" ]]; then
  LOG_DIR=$(dirname "$CKPT_DIR")
else
  LOG_DIR="$CKPT_DIR"
fi

source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export ISAACLAB_PATH
unset PYTHONPATH || true
if [[ -f "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh" || true
  set -u
fi

cd "$LAB_DIR"

echo "============================================================"
echo " Export foot policy → ONNX"
echo "  task       : $TASK"
echo "  checkpoint : $CHECKPOINT"
echo "  dest       : velocity/$DEST_NAME"
echo "============================================================"

PLAY_CMD=(
  python scripts/rsl_rl/play.py
  --task "$TASK"
  --num_envs "$NUM_ENVS"
  --device "$DEVICE"
  --checkpoint "$CHECKPOINT"
  --export_only
)
# play.py may need load_run from relative logs — also support absolute via checkpoint only
# Prefer --resume path style if play supports checkpoint absolute
if [[ "$HEADLESS" == "1" ]]; then
  PLAY_CMD+=(--headless)
fi

# Export happens inside play when it finds exported helpers — unitree play exports ONNX.
# If play does not auto-export, call torch.onnx after load. Detect exported/ after play.
echo "+ ${PLAY_CMD[*]}"
# Short play to trigger export (video off). Many unitree play scripts export at start.
# Use a timeout-friendly approach: play with video length 1 if needed.
set +e
"${PLAY_CMD[@]}" 2>&1 | tee /tmp/export_g1_foot_play.log
PLAY_RC=${PIPESTATUS[0]}
set -e
if [[ "$PLAY_RC" -ne 0 ]]; then
  echo "[ERROR] play.py export failed with rc=$PLAY_RC"
  exit "$PLAY_RC"
fi

# Locate exported onnx near checkpoint
ONNX=""
for cand in \
  "$LOG_DIR/exported/policy.onnx" \
  "$CKPT_DIR/exported/policy.onnx" \
  "$LOG_DIR/policy.onnx"
do
  if [[ -f "$cand" ]]; then
    ONNX="$cand"
    break
  fi
done

if [[ -z "$ONNX" ]]; then
  echo "[WARN] play did not produce ONNX automatically; running export helper..."
  python - <<PY
import os, sys, torch
from pathlib import Path

# Minimal: copy pt and remind user; full ONNX export needs Isaac env + runner.
ckpt = Path("$CHECKPOINT")
out_dir = ckpt.parent / "exported"
out_dir.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Checkpoint ready at {ckpt}")
print(f"[INFO] Expected ONNX at {out_dir / 'policy.onnx'}")
print("[INFO] Re-run play.py which calls export_as_onnx / torch.jit after load.")
sys.exit(0)
PY
fi

if [[ -z "$ONNX" || ! -f "$ONNX" ]]; then
  echo "[ERROR] ONNX export did not produce policy.onnx"
  exit 1
fi

DEST_ROOT="$LAB_DIR/deploy/robots/g1_29dof/config/policy/velocity/$DEST_NAME"
mkdir -p "$DEST_ROOT/params" "$DEST_ROOT/exported"

# Copy deploy.yaml if present from training log
for y in "$LOG_DIR/params/deploy.yaml" "$CKPT_DIR/params/deploy.yaml"; do
  if [[ -f "$y" ]]; then
    cp -v "$y" "$DEST_ROOT/params/deploy.yaml"
    break
  fi
done

if [[ -n "$ONNX" && -f "$ONNX" ]]; then
  cp -v "$ONNX" "$DEST_ROOT/exported/policy.onnx"
  # also mirror to model/rl for convenience
  mkdir -p "$LAB_DIR/model/rl/exported_foot"
  cp -v "$ONNX" "$LAB_DIR/model/rl/exported_foot/policy.onnx"
fi

# Optional: point config.yaml Velocity.policy_dir to this dest
CFG="$LAB_DIR/deploy/robots/g1_29dof/config/config.yaml"
echo ""
echo "Install done → $DEST_ROOT"
echo "To use in g1_ctrl, set Velocity.policy_dir to:"
echo "  config/policy/velocity/$DEST_NAME"
echo "Current config: $CFG"
echo ""
echo "MuJoCo retest:"
echo "  1) Start unitree_mujoco (domain_id 0, network lo)"
echo "  2) g1_ctrl --network lo  (A → stand, X → velocity)"
echo "  Note: foot obs terms return zeros in C++ deploy until real/sim sensors are wired;"
echo "        trained policy must tolerate zero-pad (trained with noise DR helps)."
