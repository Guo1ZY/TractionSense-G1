#!/usr/bin/env bash
# Play latest (or given) G1 velocity checkpoint
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB_DIR="${UNITREE_RL_LAB_DIR:-$ROOT}"
ISAACLAB_PATH="${ISAACLAB_PATH:-${HOME}/IsaacLab}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-isaaclab-v2}"
NUM_ENVS="${NUM_ENVS:-2}"
DEVICE="${DEVICE:-cuda:0}"
# default: record video on play (set VIDEO=0 to disable)
VIDEO="${VIDEO:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-300}"
KEYBOARD="${KEYBOARD:-0}"
VX_MAX="${VX_MAX:-1.0}"

CHECKPOINT="${1:-}"
if [[ -z "$CHECKPOINT" ]]; then
  # default: full official-stack 50k checkpoint if present
  DEFAULT_CKPT="$ROOT/checkpoints/baseline_model_49999.pt"
  if [[ -f "$DEFAULT_CKPT" ]]; then
    CHECKPOINT="$DEFAULT_CKPT"
  fi
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export ISAACLAB_PATH
export DISPLAY="${DISPLAY:-:1}"
source "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh" 2>/dev/null || true
cd "$LAB_DIR"

ARGS=(
  --task Unitree-G1-29dof-Velocity
  --num_envs "$NUM_ENVS"
  --device "$DEVICE"
)

if [[ -n "$CHECKPOINT" ]]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
else
  echo "[WARN] No checkpoint given; play will try latest under logs/"
fi

if [[ "$VIDEO" == "1" ]]; then
  ARGS+=(--video --video_length "$VIDEO_LENGTH")
fi

if [[ "$KEYBOARD" == "1" ]]; then
  NUM_ENVS=1
  ARGS=(
    --task Unitree-G1-29dof-Velocity
    --num_envs "$NUM_ENVS"
    --device "$DEVICE"
    --keyboard
    --vx_max "$VX_MAX"
    --real-time
  )
  if [[ -n "$CHECKPOINT" ]]; then
    ARGS+=(--checkpoint "$CHECKPOINT")
  fi
  if [[ "$VIDEO" == "1" ]]; then
    ARGS+=(--video --video_length "$VIDEO_LENGTH")
  fi
fi

echo "+ python scripts/rsl_rl/play.py ${ARGS[*]}"
python scripts/rsl_rl/play.py "${ARGS[@]}"
EC=$?

# copy latest recorded mp4 to a convenient place
if [[ "$VIDEO" == "1" && $EC -eq 0 && -n "$CHECKPOINT" ]]; then
  CKPT_DIR="$(dirname "$CHECKPOINT")"
  VID="$(ls -t "$CKPT_DIR"/videos/play/*.mp4 2>/dev/null | head -1 || true)"
  if [[ -n "$VID" ]]; then
    VIDEO_OUT_DIR="${VIDEO_OUT_DIR:-$ROOT/video}"
    mkdir -p "$VIDEO_OUT_DIR"
    STAMP="$(date +%Y%m%d_%H%M%S)"
    OUT_MP4="$VIDEO_OUT_DIR/g1_play_${STAMP}.mp4"
    LATEST_MP4="$VIDEO_OUT_DIR/g1_play_latest.mp4"
    cp -f "$VID" "$OUT_MP4"
    cp -f "$VID" "$LATEST_MP4"
    echo "[INFO] Video saved:"
    echo "  $VID"
    echo "  -> $OUT_MP4"
    echo "  -> $LATEST_MP4"
  fi
fi
exit $EC
