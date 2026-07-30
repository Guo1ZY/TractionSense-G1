#!/usr/bin/env bash
# Fine-tune the 641-D Oracle Teacher on high<->low traction transitions.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB="${UNITREE_RL_LAB_DIR:-$ROOT}"
ISAACLAB_PATH="${ISAACLAB_PATH:-${HOME}/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-isaaclab-v2}"
TASK="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-MotionFeedback-Switch"
CHECKPOINT="${CHECKPOINT:-$LAB/logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_teacher_motion_balanced_symmetry/2026-07-29_13-44-07_motion_balanced_symmetry_prod_20260729/model_8900.pt}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-240}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-314159}"
RUN_NAME="${RUN_NAME:-two_surface_switch_$(date +%Y%m%d_%H%M%S)}"
HEADLESS="${HEADLESS:-1}"

usage() {
  echo "Usage: $0 [--smoke] [--checkpoint FILE] [--num-envs N]"
  echo "          [--max-iterations N] [--device DEVICE] [--seed N]"
  echo "          [--run-name NAME]"
}

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --smoke)
      NUM_ENVS=64
      MAX_ITERS=2
      RUN_NAME="two_surface_switch_smoke_$(date +%Y%m%d_%H%M%S)"
      shift
      ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --max-iterations) MAX_ITERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -f "$CHECKPOINT" ]] || {
  echo "[ERROR] checkpoint not found: $CHECKPOINT" >&2
  exit 2
}
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

cd "$LAB"
echo "============================================================"
echo " G1 two-surface transition Teacher"
echo "  task       : $TASK"
echo "  checkpoint : $CHECKPOINT"
echo "  envs/iters : $NUM_ENVS / $MAX_ITERS"
echo "  device/seed: $DEVICE / $SEED"
echo "  run        : $RUN_NAME"
echo "============================================================"

cmd=(
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
  cmd+=(--headless)
fi
echo "+ ${cmd[*]}"
exec "${cmd[@]}"
