#!/usr/bin/env bash
# Train the 1840->29 dual-foot magnetic-history Student on an 8GB GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB="$ROOT"
PYTHON="${ISAACLAB_PYTHON:-python3}"
DATA="$LAB/logs/evaluations/friction_estimator/20260728_teacher7989_hardened_8030_cap1ms"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$LAB/logs/evaluations/traction_magnetic_student/${STAMP}_mag15x3_hist15_7989}"

exec "$PYTHON" "$ROOT/research_scripts/train_magnetic_traction_student.py" \
  --teacher-onnx "$LAB/deploy/robots/g1_29dof/config/policy/velocity/traction_teacher_7989/exported/policy.onnx" \
  --source-student "$LAB/logs/evaluations/traction_student/20260728_traction_student_7989_dagger1_cap1ms/student_actor.pt" \
  --train "$DATA/train_noisy_teacher.npz" \
  --test "$DATA/test_noisy_teacher.npz" \
  --augment "$LAB/logs/evaluations/mujoco_traction_student/20260728_student7989_dagger1_cap1_full5x3/policy_obs.npz" \
  --augment-repeat 8 \
  --epochs "${EPOCHS:-80}" \
  --batch-size "${BATCH_SIZE:-512}" \
  --device cuda:0 \
  --strict \
  --output-dir "$OUTPUT_DIR" \
  "$@"
