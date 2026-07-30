#!/usr/bin/env bash
# Train Unitree G1-29dof velocity locomotion (unitree_rl_lab + Isaac Lab 2.3.2)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB_DIR="${UNITREE_RL_LAB_DIR:-$ROOT}"
ISAACLAB_PATH="${ISAACLAB_PATH:-${HOME}/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-isaaclab-v2}"

# Defaults tuned for RTX 5070 Ti 16GB (safe first pipeline)
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-50000}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-pipeline}"
HEADLESS="${HEADLESS:-1}"

usage() {
  cat <<EOF
Usage: $0 [options]

  NUM_ENVS=2048 MAX_ITERS=50000 $0
  $0 --smoke          # 30 iters smoke test
  $0 --num-envs 1024 --max-iterations 10000

Env overrides: ISAACLAB_PATH, CONDA_ENV, NUM_ENVS, MAX_ITERS, DEVICE, SEED, RUN_NAME
EOF
}

SMOKE=0
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --smoke) SMOKE=1; shift ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --max-iterations) MAX_ITERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ "$SMOKE" == "1" ]]; then
  NUM_ENVS="${NUM_ENVS:-512}"
  MAX_ITERS=30
  RUN_NAME="smoke"
fi

source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export ISAACLAB_PATH
# Avoid ROS Python path pollution
unset PYTHONPATH || true
# shellcheck disable=SC1091
if [[ -f "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh" ]]; then
  set +u
  source "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh" || true
  set -u
fi

cd "$LAB_DIR"
mkdir -p logs/rsl_rl

RSL_VER="$(python -c 'import importlib.metadata as m; print(m.version("rsl-rl-lib"))' 2>/dev/null || echo unknown)"

echo "============================================================"
echo " G1 Velocity Training"
echo "  task        : Unitree-G1-29dof-Velocity"
echo "  num_envs    : $NUM_ENVS"
echo "  max_iters   : $MAX_ITERS"
echo "  device      : $DEVICE"
echo "  seed        : $SEED"
echo "  run_name    : $RUN_NAME"
echo "  workdir     : $LAB_DIR"
echo "  rsl-rl      : $RSL_VER"
echo "============================================================"

CMD=(
  python scripts/rsl_rl/train.py
  --task Unitree-G1-29dof-Velocity
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERS"
  --device "$DEVICE"
  --seed "$SEED"
  --run_name "$RUN_NAME"
)
if [[ "$HEADLESS" == "1" ]]; then
  CMD+=(--headless)
fi
if ((${#EXTRA[@]})); then
  CMD+=("${EXTRA[@]}")
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"
