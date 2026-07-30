#!/usr/bin/env bash
# Jointly recover 1.0-m/s high-grip tracking and reduce straight-line drift.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB_DIR="${UNITREE_RL_LAB_DIR:-$PROJECT_ROOT}"
ISAACLAB_PATH="${ISAACLAB_PATH:-${HOME}/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-isaaclab-v2}"
TASK="${TASK:-Unitree-G1-29dof-Velocity-Foot-TractionTeacher-SpeedLateral}"
CHECKPOINT="${CHECKPOINT:-$LAB_DIR/logs/imported_remote_20260728/model_8110.pt}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-200}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-8291}"
RUN_NAME="${RUN_NAME:-speed_lateral_from_8110}"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      NUM_ENVS=128
      MAX_ITERS=2
      RUN_NAME="speed_lateral_smoke"
      shift
      ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --max-iterations) MAX_ITERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--smoke] [--task ID] [--checkpoint PT] [--num-envs N] [--max-iterations N]"
      exit 0
      ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

CHECKPOINT="$(realpath "$CHECKPOINT")"
[[ -f "$CHECKPOINT" ]] || {
  echo "[ERROR] checkpoint missing: $CHECKPOINT" >&2
  exit 1
}

source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export ISAACLAB_PATH
unset PYTHONPATH || true
if [[ -f "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" >/dev/null 2>&1 || true
  set -u
fi

cd "$LAB_DIR"
exec python scripts/rsl_rl/train.py \
  --task "$TASK" \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERS" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --run_name "$RUN_NAME" \
  --headless \
  --resume_checkpoint "$CHECKPOINT" \
  "${EXTRA[@]}"
