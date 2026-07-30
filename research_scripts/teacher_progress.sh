#!/usr/bin/env bash
# One-click colored progress display for the active TractionTeacher run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEACHER_LOG_ROOT="$ROOT/logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_teacher"

exec "$ROOT/research_scripts/training_progress.py" --log-root "$TEACHER_LOG_ROOT" "$@"
