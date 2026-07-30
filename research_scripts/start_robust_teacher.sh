#!/usr/bin/env bash
# Start robust Teacher training and its periodic evaluator in the background.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT/logs/runtime"
mkdir -p "$RUNTIME_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 [finetune_robust_teacher.sh options]"
  echo "Starts training and periodic Isaac/MuJoCo evaluation in the background."
  exit 0
fi

if pgrep -af 'train.py.*TractionTeacher-Robust' >/dev/null; then
  echo "[ERROR] robust Teacher training is already running:" >&2
  pgrep -af 'train.py.*TractionTeacher-Robust' >&2
  exit 1
fi

stamp="$(date '+%Y%m%d_%H%M%S')"
train_log="$RUNTIME_DIR/robust_teacher_train_$stamp.log"
monitor_log="$RUNTIME_DIR/robust_teacher_monitor_$stamp.log"

nohup setsid "$ROOT/research_scripts/finetune_robust_teacher.sh" "$@" \
  >"$train_log" 2>&1 < /dev/null &
train_pid=$!
echo "$train_pid" > "$RUNTIME_DIR/robust_teacher_train.pid"

nohup setsid "$ROOT/research_scripts/monitor_robust_teacher.sh" "$train_pid" \
  >"$monitor_log" 2>&1 < /dev/null &
monitor_pid=$!
echo "$monitor_pid" > "$RUNTIME_DIR/robust_teacher_monitor.pid"

echo "training pid : $train_pid"
echo "monitor pid  : $monitor_pid"
echo "training log : $train_log"
echo "monitor log  : $monitor_log"
