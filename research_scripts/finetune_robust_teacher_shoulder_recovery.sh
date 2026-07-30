#!/usr/bin/env bash
# Recover high-grip transient stability without modifying the frozen baselines.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB_DIR="${UNITREE_RL_LAB_DIR:-$PROJECT_ROOT}"
ISAACLAB_PATH="${ISAACLAB_PATH:-${HOME}/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-isaaclab-v2}"
TASK="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-Robust-Shoulder-Recovery"
CHECKPOINT="${CHECKPOINT:-$LAB_DIR/logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_teacher_robust_shoulder/2026-07-21_18-18-09_shoulder_boost_from_7750/model_7775.pt}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-120}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-63}"
RUN_NAME="${RUN_NAME:-high_mu_transient_recovery_from_7775}"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) NUM_ENVS=128; MAX_ITERS=2; RUN_NAME="shoulder_recovery_smoke"; shift ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --max-iterations) MAX_ITERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--smoke] [--checkpoint PT] [--num-envs N] [--max-iterations N]"
      exit 0 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

CHECKPOINT="$(realpath "$CHECKPOINT")"
[[ -f "$CHECKPOINT" ]] || { echo "[ERROR] checkpoint missing: $CHECKPOINT" >&2; exit 1; }

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
echo " G1 robust Teacher high-mu transient recovery"
echo "  checkpoint       : $CHECKPOINT"
echo "  task             : $TASK"
echo "  envs / iters     : $NUM_ENVS / $MAX_ITERS"
echo "  device / seed    : $DEVICE / $SEED"
echo "  run              : $RUN_NAME"
echo "============================================================"

exec python scripts/rsl_rl/train.py \
  --task "$TASK" --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERS" \
  --device "$DEVICE" --seed "$SEED" --run_name "$RUN_NAME" --headless \
  --resume_checkpoint "$CHECKPOINT" "${EXTRA[@]}"
