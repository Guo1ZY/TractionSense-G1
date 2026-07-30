#!/usr/bin/env bash
# Evaluate robust Teacher checkpoints every 500 iterations without stopping training.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="$ROOT/logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_teacher_robust"
STATE_FILE="$LOG_ROOT/.last_periodic_eval"
TRAIN_PID="${1:-}"
BASE_ITER="${BASE_ITER:-4999}"
EVAL_EVERY="${EVAL_EVERY:-500}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MIN_EVAL_ITER=$((BASE_ITER + EVAL_EVERY))

mkdir -p "$LOG_ROOT"
last_eval=-1
if [[ -f "$STATE_FILE" ]]; then
  read -r last_eval < "$STATE_FILE" || last_eval=-1
fi

latest_checkpoint() {
  find "$LOG_ROOT" -type f -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | awk 'NR==1 {sub(/^[^ ]+ /, ""); print; exit}'
}

checkpoint_iteration() {
  local name
  name="$(basename "$1")"
  name="${name#model_}"
  echo "${name%.pt}"
}

evaluate_checkpoint() {
  local checkpoint="$1"
  local iteration="$2"
  echo "[$(date '+%F %T')] periodic evaluation: model_$iteration"
  "$ROOT/research_scripts/evaluate_robust_teacher.sh" \
    --checkpoint "$checkpoint" --quick --device cpu --no-color || true
  "$ROOT/research_scripts/validate_teacher_mujoco.sh" \
    --checkpoint "$checkpoint" --smoke --skip-build || true
  echo "$iteration" > "$STATE_FILE"
  last_eval="$iteration"
}

while true; do
  checkpoint="$(latest_checkpoint)"
  if [[ -n "$checkpoint" ]]; then
    iteration="$(checkpoint_iteration "$checkpoint")"
    if [[ "$iteration" =~ ^[0-9]+$ ]] \
      && (( iteration >= MIN_EVAL_ITER )) \
      && (( iteration > last_eval )) \
      && (( iteration % EVAL_EVERY == 0 )); then
      evaluate_checkpoint "$checkpoint" "$iteration"
    fi
  fi

  if [[ -n "$TRAIN_PID" ]] && ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    checkpoint="$(latest_checkpoint)"
    if [[ -n "$checkpoint" ]]; then
      iteration="$(checkpoint_iteration "$checkpoint")"
      if [[ "$iteration" =~ ^[0-9]+$ ]] && (( iteration > last_eval )); then
        evaluate_checkpoint "$checkpoint" "$iteration"
      fi
    fi
    echo "[$(date '+%F %T')] training stopped; monitor exiting"
    exit 0
  fi
  sleep "$POLL_SECONDS"
done
