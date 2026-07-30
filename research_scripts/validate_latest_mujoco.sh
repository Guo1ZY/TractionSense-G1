#!/usr/bin/env bash
# One-command entry point for the latest TractionAdaptive MuJoCo matrix.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT/research_scripts/mujoco_friction_speed_matrix.py" "$@"
