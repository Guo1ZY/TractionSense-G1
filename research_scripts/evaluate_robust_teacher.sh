#!/usr/bin/env bash
# Evaluate a robust Teacher checkpoint inside its randomized Isaac domain.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBUST_ROOT="$ROOT/logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_teacher_robust"
EVAL_ROOT="$ROOT/logs/evaluations/traction_teacher_robust"

exec "$ROOT/research_scripts/evaluate_teacher.sh" \
  --task Unitree-G1-29dof-Velocity-Foot-TractionTeacher-Robust \
  --experiment-root "$ROBUST_ROOT" \
  --evaluation-root "$EVAL_ROOT" \
  "$@"
