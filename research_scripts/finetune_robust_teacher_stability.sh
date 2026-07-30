#!/usr/bin/env bash
# Targeted continuation that removes the randomized-domain fall tail.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB_DIR="${UNITREE_RL_LAB_DIR:-$ROOT}"
ISAACLAB_PATH="${ISAACLAB_PATH:-${HOME}/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-isaaclab-v2}"

TASK="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-Robust-Stability"
CHECKPOINT="${CHECKPOINT:-$LAB_DIR/logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_teacher_robust/2026-07-21_13-15-55_robust_from_4999/model_6600.pt}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-500}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-44}"
RUN_NAME="${RUN_NAME:-stability_from_6600}"
HEADLESS="${HEADLESS:-1}"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      echo "Usage: $0 [--smoke] [--checkpoint FILE] [--num-envs N] [--max-iterations N]"
      echo "          [--device DEVICE] [--seed N] [--run-name NAME]"
      exit 0
      ;;
    --smoke) NUM_ENVS=128; MAX_ITERS=2; RUN_NAME="stability_smoke"; shift ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --max-iterations) MAX_ITERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi
CHECKPOINT="$(realpath "$CHECKPOINT")"

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
echo "============================================================"
echo " G1 641-D Robust Oracle Teacher stability fine-tune"
echo "  task             : $TASK"
echo "  strict warm-start: $CHECKPOINT"
echo "  num_envs         : $NUM_ENVS"
echo "  added iterations : $MAX_ITERS"
echo "  device / seed    : $DEVICE / $SEED"
echo "  run_name         : $RUN_NAME"
echo "============================================================"

CMD=(
  python scripts/rsl_rl/train.py
  --task "$TASK"
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERS"
  --device "$DEVICE"
  --seed "$SEED"
  --run_name "$RUN_NAME"
  --resume_checkpoint "$CHECKPOINT"
)
if [[ "$HEADLESS" == "1" ]]; then
  CMD+=(--headless)
fi
if ((${#EXTRA[@]})); then
  CMD+=("${EXTRA[@]}")
fi

echo "+ ${CMD[*]}"
exec "${CMD[@]}"
