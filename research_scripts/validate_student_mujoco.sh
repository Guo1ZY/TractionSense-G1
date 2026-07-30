#!/usr/bin/env bash
# Validate the installed standalone 640-D Student; simulation loopback only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLOT="$ROOT/deploy/robots/g1_29dof/config/policy/velocity/traction_student"
CHECKPOINT="$(tr -d '\r\n' < "$SLOT/checkpoint.txt")"

exec python3 "$ROOT/research_scripts/mujoco_friction_speed_matrix.py" \
  --profile adaptive --slot traction_student --checkpoint "$CHECKPOINT" --skip-export "$@"
