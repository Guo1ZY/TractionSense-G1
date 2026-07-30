#!/usr/bin/env bash
# One-shot login launcher used only to recover from the pending NVIDIA reboot.
set -euo pipefail

ROOT="${TRACTIONSENSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STATE_DIR="$ROOT/.training_state"
COMPLETE="$STATE_DIR/magnetic_speed_lateral_complete"
mkdir -p "$STATE_DIR"

if [[ -f "$COMPLETE" ]]; then
  echo "[SKIP] magnetic speed/lateral production pipeline already completed"
  exit 0
fi

for _ in $(seq 1 60); do
  if nvidia-smi >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
nvidia-smi >/dev/null

RUN_ID="${RUN_ID:-20260729_production}" \
  "$ROOT/research_scripts/run_magnetic_speed_lateral_pipeline.sh"
date --iso-8601=seconds > "$COMPLETE"
