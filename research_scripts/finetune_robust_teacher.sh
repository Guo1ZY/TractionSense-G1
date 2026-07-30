#!/usr/bin/env bash
# Fine-tune the 641-D Oracle Teacher under conservative Sim2Sim randomization.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB_DIR="${UNITREE_RL_LAB_DIR:-$ROOT}"
ISAACLAB_PATH="${ISAACLAB_PATH:-${HOME}/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-isaaclab-v2}"

TASK="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-Robust"
CHECKPOINT="${CHECKPOINT:-$LAB_DIR/logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_teacher/2026-07-20_17-15-25_traction_teacher_flat_balanced/model_4999.pt}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-2000}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-43}"
RUN_NAME="${RUN_NAME:-robust_from_4999}"
HEADLESS="${HEADLESS:-1}"
LOAD_OPTIMIZER=0
EXTRA=()

usage() {
  echo "Usage: $0 [--smoke] [--checkpoint FILE] [--num-envs N] [--max-iterations N]"
  echo "          [--device DEVICE] [--seed N] [--run-name NAME] [--load-optimizer]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --smoke) NUM_ENVS=128; MAX_ITERS=2; RUN_NAME="robust_smoke"; shift ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --max-iterations) MAX_ITERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --load-optimizer) LOAD_OPTIMIZER=1; shift ;;
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
mkdir -p logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_teacher_robust

echo "============================================================"
echo " G1 641-D Robust Oracle Teacher fine-tune"
echo "  task             : $TASK"
echo "  strict warm-start: $CHECKPOINT"
echo "  optimizer        : $([[ $LOAD_OPTIMIZER == 1 ]] && echo resume || echo fresh)"
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
if [[ "$LOAD_OPTIMIZER" == "1" ]]; then
  CMD+=(--load_optimizer)
fi
if [[ "$HEADLESS" == "1" ]]; then
  CMD+=(--headless)
fi
if ((${#EXTRA[@]})); then
  CMD+=("${EXTRA[@]}")
fi

echo "+ ${CMD[*]}"
exec "${CMD[@]}"
