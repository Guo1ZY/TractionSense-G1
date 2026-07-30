#!/usr/bin/env bash
# Reproduce the selected 640-D privileged Teacher distillation run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB="$ROOT"
DATA="$LAB/logs/evaluations/friction_estimator/20260721_model_7750_robust_teacher"
MUJOCO="$LAB/logs/evaluations/mujoco_traction_teacher"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$LAB/logs/evaluations/traction_student/${STAMP}_model_7750_privileged_aux}"

source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate isaaclab-v2

python "$ROOT/research_scripts/distill_traction_student.py" \
  --teacher-onnx "$LAB/deploy/robots/g1_29dof/config/policy/velocity/traction_teacher/exported/policy.onnx" \
  --estimator-pt "$DATA/model_sim2sim_dagger1/friction_estimator.pt" \
  --target-mu privileged \
  --train "$DATA/train_noisy_teacher.npz" \
  --test "$DATA/test_noisy_teacher.npz" \
  --augment-raw-grid "$MUJOCO/final_model_7750_independent_full/policy_obs.bin" \
  --augment-raw-grid "$MUJOCO/final_model_7750_estimated_mu_dagger1_full/policy_obs.bin" \
  --augment-repeat 4 \
  --epochs 120 --batch-size 1024 --device cuda:0 --strict \
  --output-dir "$OUTPUT_DIR" \
  --deploy-template "$LAB/deploy/robots/g1_29dof/config/policy/velocity/traction_teacher/params/deploy.yaml" \
  --install-slot "$LAB/deploy/robots/g1_29dof/config/policy/velocity/traction_student" \
  "$@"

echo "[DONE] Student installed: $LAB/deploy/robots/g1_29dof/config/policy/velocity/traction_student"
