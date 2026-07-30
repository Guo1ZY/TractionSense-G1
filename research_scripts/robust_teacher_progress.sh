#!/usr/bin/env bash
# Colored progress for a model_4999 + 2000-iteration robust fine-tune.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBUST_ROOT="$ROOT/logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_teacher_robust"

exec "$ROOT/research_scripts/training_progress.py" \
  --log-root "$ROBUST_ROOT" \
  --start "${START_ITERATION:-4999}" \
  --target "${TARGET_ITERATION:-6999}" \
  "$@"
