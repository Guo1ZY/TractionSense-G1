#!/usr/bin/env bash
# One-command Oracle MuJoCo validation for the newest 641-D TractionTeacher.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT/research_scripts/mujoco_friction_speed_matrix.py" --profile teacher "$@"
