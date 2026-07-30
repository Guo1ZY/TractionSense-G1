#!/usr/bin/env bash
# Re-run the isolated model_7890 DAgger2 Student with its candidate command slew limiter.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB="$ROOT"
SLOT="traction_student_7890_dagger2_candidate"
CHECKPOINT="$LAB/logs/evaluations/traction_student/20260722_model_7890_privileged_aux_dagger2/student_actor.pt"

[[ -f "$CHECKPOINT" ]] || { echo "[ERROR] missing Student: $CHECKPOINT" >&2; exit 1; }

exec python3 "$ROOT/research_scripts/mujoco_friction_speed_matrix.py" \
  --profile adaptive \
  --slot "$SLOT" \
  --checkpoint "$CHECKPOINT" \
  --skip-export \
  --strict \
  "$@"
