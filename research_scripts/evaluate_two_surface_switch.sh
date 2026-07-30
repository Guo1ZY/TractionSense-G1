#!/usr/bin/env bash
# Evaluate one unchanged command through high->low->high friction phases.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB="${UNITREE_RL_LAB_DIR:-$ROOT}"
ISAACLAB_PATH="${ISAACLAB_PATH:-${HOME}/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-isaaclab-v2}"
TASK="Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-Switch"
SHARED_POLICY="${SHARED_POLICY:-$LAB/logs/evaluations/friction_switch_training/final_sensor_policy_20260729/shared_magnetic_policy.pt}"
LATERAL_ESTIMATOR="${LATERAL_ESTIMATOR:-$LAB/logs/evaluations/traction_magnetic_motion/20260729_lateral_velocity_estimator/lateral_velocity_estimator.pt}"
NUM_ENVS="${NUM_ENVS:-64}"
PHASE_STEPS="${PHASE_STEPS:-150}"
WARMUP_STEPS="${WARMUP_STEPS:-75}"
HIGH_MU="${HIGH_MU:-1.20}"
LOW_MU="${LOW_MU:-0.15}"
VX="${VX:-0.60}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
ABLATE=0
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$LAB/logs/evaluations/friction_switch/$STAMP}"

usage() {
  echo "Usage: $0 [--shared-policy FILE] [--ablate-foot] [--num-envs N]"
  echo "          [--phase-steps N] [--high-mu MU] [--low-mu MU]"
  echo "          [--vx MPS] [--seed N] [--output-dir DIR]"
}

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --shared-policy) SHARED_POLICY="$2"; shift 2 ;;
    --lateral-estimator) LATERAL_ESTIMATOR="$2"; shift 2 ;;
    --ablate-foot) ABLATE=1; shift ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --phase-steps) PHASE_STEPS="$2"; shift 2 ;;
    --high-mu) HIGH_MU="$2"; shift 2 ;;
    --low-mu) LOW_MU="$2"; shift 2 ;;
    --vx) VX="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -f "$SHARED_POLICY" ]] || {
  echo "[ERROR] shared policy not found: $SHARED_POLICY" >&2
  exit 2
}
SHARED_POLICY="$(realpath "$SHARED_POLICY")"
mkdir -p "$OUTPUT_DIR"

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

cmd=(
  python "$LAB/scripts/rsl_rl/eval_friction_matrix.py"
  --task "$TASK"
  --shared_policy "$SHARED_POLICY"
  --num_envs "$NUM_ENVS"
  --warmup_steps "$WARMUP_STEPS"
  --command_ramp_steps "$WARMUP_STEPS"
  --switch_sequence "$HIGH_MU" "$LOW_MU" "$HIGH_MU"
  --switch_phase_steps "$PHASE_STEPS"
  --switch_settle_steps 25
  --switch_max_response_s 1.0
  --vx "$VX"
  --seed "$SEED"
  --output_csv "$OUTPUT_DIR/phases.csv"
  --device "$DEVICE"
  --headless
)
if [[ -f "$LATERAL_ESTIMATOR" ]]; then
  cmd+=(--lateral_estimator "$(realpath "$LATERAL_ESTIMATOR")")
fi
if [[ "$ABLATE" == "1" ]]; then
  cmd+=(--ablate_foot_sensor)
fi

cd "$LAB"
echo "+ ${cmd[*]}"
exec "${cmd[@]}"
