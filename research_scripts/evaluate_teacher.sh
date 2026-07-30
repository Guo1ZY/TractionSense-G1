#!/usr/bin/env bash
# Bootstrap the Isaac Lab environment, then evaluate the newest teacher checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate isaaclab-v2
export ISAACLAB_PATH="${ISAACLAB_PATH:-${HOME}/IsaacLab}"
unset PYTHONPATH || true
if [[ -f "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" >/dev/null 2>&1 || true
  set -u
fi

exec python "$ROOT/research_scripts/evaluate_teacher.py" "$@"
