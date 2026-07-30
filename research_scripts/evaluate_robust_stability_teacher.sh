#!/usr/bin/env bash
# Evaluate a stability-pass Teacher checkpoint under the matching randomized task.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STABILITY_ROOT="$ROOT/logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_teacher_robust_stability"
EVAL_ROOT="$ROOT/logs/evaluations/traction_teacher_robust_stability"

exec "$ROOT/research_scripts/evaluate_teacher.sh" \
  --task Unitree-G1-29dof-Velocity-Foot-TractionTeacher-Robust-Stability \
  --experiment-root "$STABILITY_ROOT" \
  --evaluation-root "$EVAL_ROOT" \
  "$@"
