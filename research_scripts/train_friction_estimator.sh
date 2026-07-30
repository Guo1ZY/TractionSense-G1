#!/usr/bin/env bash
# Collect noisy-Teacher rollouts, train a friction estimator, export ONNX, gate it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB="$ROOT"
TASK="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-Noisy"
LOG_ROOT="$LAB/logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_teacher"
CHECKPOINT=""
OUTPUT_DIR=""
DEVICE="cpu"
QUICK=0
STRICT=0
SKIP_COLLECT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint|-c) CHECKPOINT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --quick) QUICK=1; shift ;;
    --strict) STRICT=1; shift ;;
    --skip-collect) SKIP_COLLECT=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--checkpoint PT] [--output-dir DIR] [--device cpu] [--quick] [--strict] [--skip-collect]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT="$(find "$LOG_ROOT" -type f -name 'model_*.pt' -size +1M -printf '%T@ %p\n' \
    | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi
if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] Teacher checkpoint not found" >&2
  exit 1
fi
CHECKPOINT="$(realpath "$CHECKPOINT")"
if [[ -z "$OUTPUT_DIR" ]]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_DIR="$LAB/logs/evaluations/friction_estimator/${STAMP}_$(basename "${CHECKPOINT%.pt}")"
fi
mkdir -p "$OUTPUT_DIR"
# The collection commands run with $LAB as their working directory.  Resolve
# once here so a caller-supplied relative path cannot silently create a nested
# unitree_rl_lab/unitree_rl_lab output tree.
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
TRAIN_NPZ="$OUTPUT_DIR/train_noisy_teacher.npz"
TEST_NPZ="$OUTPUT_DIR/test_noisy_teacher.npz"

source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate isaaclab-v2
export ISAACLAB_PATH="${ISAACLAB_PATH:-${HOME}/IsaacLab}"
unset PYTHONPATH || true
if [[ -f "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" >/dev/null 2>&1 || true
  set -u
fi

if [[ "$SKIP_COLLECT" == "0" ]]; then
  if [[ "$QUICK" == "1" ]]; then
    NUM_ENVS=16
    MAX_STEPS=40
    WARMUP_STEPS=10
    STRIDE=2
    EPOCHS=8
    MUS=(0.08 0.20 0.80 1.20)
    SPEEDS=(0.5 1.5)
  else
    NUM_ENVS=64
    MAX_STEPS=160
    WARMUP_STEPS=40
    STRIDE=4
    EPOCHS=40
    MUS=(0.08 0.15 0.25 0.40 0.60 0.80 1.00 1.20)
    SPEEDS=(0.3 0.8 1.5)
  fi

  cd "$LAB"
  echo "[1/3] collect training set → $TRAIN_NPZ"
  python scripts/rsl_rl/eval_friction_matrix.py \
    --task "$TASK" --checkpoint "$CHECKPOINT" --device "$DEVICE" --headless \
    --num_envs "$NUM_ENVS" --max_steps "$MAX_STEPS" --warmup_steps "$WARMUP_STEPS" \
    --collect_stride "$STRIDE" --collect_npz "$TRAIN_NPZ" \
    --output_csv "$OUTPUT_DIR/train_matrix.csv" --seed 42 \
    --mu_bins "${MUS[@]}" --vx "${SPEEDS[@]}"

  echo "[2/3] collect unseen test set → $TEST_NPZ"
  python scripts/rsl_rl/eval_friction_matrix.py \
    --task "$TASK" --checkpoint "$CHECKPOINT" --device "$DEVICE" --headless \
    --num_envs "$NUM_ENVS" --max_steps "$MAX_STEPS" --warmup_steps "$WARMUP_STEPS" \
    --collect_stride "$STRIDE" --collect_npz "$TEST_NPZ" \
    --output_csv "$OUTPUT_DIR/test_matrix.csv" --seed 314159 \
    --mu_bins "${MUS[@]}" --vx "${SPEEDS[@]}"
else
  EPOCHS=$([[ "$QUICK" == "1" ]] && echo 8 || echo 40)
  if [[ ! -f "$TRAIN_NPZ" || ! -f "$TEST_NPZ" ]]; then
    echo "[ERROR] --skip-collect requires $TRAIN_NPZ and $TEST_NPZ" >&2
    exit 1
  fi
fi

cd "$ROOT"
echo "[3/3] train, gate and export estimator"
EST_ARGS=(
  --train "$TRAIN_NPZ" --test "$TEST_NPZ" --features all
  --epochs "$EPOCHS" --device "$DEVICE" --output-dir "$OUTPUT_DIR/model"
)
if [[ "$STRICT" == "1" ]]; then
  EST_ARGS+=(--strict)
fi
python scripts/evaluate_friction_estimator.py "${EST_ARGS[@]}"

echo "[DONE] $OUTPUT_DIR"
echo "       ONNX: $OUTPUT_DIR/model/friction_estimator.onnx"
