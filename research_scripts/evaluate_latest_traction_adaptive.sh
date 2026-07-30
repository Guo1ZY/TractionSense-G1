#!/usr/bin/env bash
# Find the newest healthy TractionAdaptive checkpoint, run a fixed-friction
# matrix, and generate a compact PASS/WARN report. Simulation only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB_DIR="${UNITREE_RL_LAB_DIR:-$ROOT}"
ISAACLAB_PATH="${ISAACLAB_PATH:-${HOME}/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-isaaclab-v2}"
EXP_ROOT="$LAB_DIR/logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_adaptive"
TASK="Unitree-G1-29dof-Velocity-Foot-TractionAdaptive"
DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT="${CHECKPOINT:-}"
MODE="quick"

usage() {
  cat <<EOF
Usage: $0 [--quick|--full] [--checkpoint /path/model_N.pt] [--device cuda:0]

  --quick  cmd=1.5 over five friction levels, 32 envs, 300 measured steps (default)
  --full   cmd=0.5/1.0/1.5 over five friction levels, 64 envs, 500 measured steps

Environment overrides: CHECKPOINT, DEVICE, NUM_ENVS, MAX_STEPS, WARMUP_STEPS, SEED
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) MODE="quick"; shift ;;
    --full) MODE="full"; shift ;;
    --checkpoint|-c) CHECKPOINT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1"; usage; exit 2 ;;
  esac
done

source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export ISAACLAB_PATH
unset PYTHONPATH || true
if [[ -f "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" >/dev/null 2>&1 || true
  set -u
fi

if [[ -z "$CHECKPOINT" ]]; then
  mapfile -t CANDIDATES < <(
    find "$EXP_ROOT" -mindepth 2 -maxdepth 2 -type f -name 'model_*.pt' \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr | cut -d' ' -f2-
  )
  for candidate in "${CANDIDATES[@]}"; do
    # Skip a checkpoint that is still being written and validate its structure.
    if [[ $(stat -c '%s' "$candidate") -lt 1000000 ]]; then
      continue
    fi
    if python - "$candidate" <<'PY' >/dev/null 2>&1
import sys
import torch

obj = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert isinstance(obj, dict)
assert "actor_state_dict" in obj or "model_state_dict" in obj
PY
    then
      CHECKPOINT="$candidate"
      break
    fi
  done
fi

if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] no healthy TractionAdaptive checkpoint found under: $EXP_ROOT"
  exit 1
fi
CHECKPOINT="$(realpath "$CHECKPOINT")"

if [[ "$MODE" == "full" ]]; then
  NUM_ENVS="${NUM_ENVS:-64}"
  MAX_STEPS="${MAX_STEPS:-500}"
  WARMUP_STEPS="${WARMUP_STEPS:-75}"
  VX_VALUES=(0.5 1.0 1.5)
else
  NUM_ENVS="${NUM_ENVS:-32}"
  MAX_STEPS="${MAX_STEPS:-300}"
  WARMUP_STEPS="${WARMUP_STEPS:-50}"
  VX_VALUES=(1.5)
fi
SEED="${SEED:-42}"
MU_VALUES=(0.08 0.20 0.40 0.80 1.20)

ITER="$(basename "$CHECKPOINT" .pt)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$LAB_DIR/logs/evaluations/traction_adaptive/${STAMP}_${ITER}_${MODE}"
CSV="$OUT_DIR/matrix.csv"
RAW_LOG="$OUT_DIR/eval.log"
SUMMARY="$OUT_DIR/summary.md"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo " TractionAdaptive latest-checkpoint evaluation"
echo "  checkpoint : $CHECKPOINT"
echo "  mode       : $MODE"
echo "  envs       : $NUM_ENVS"
echo "  warmup     : $WARMUP_STEPS"
echo "  steps      : $MAX_STEPS"
echo "  output     : $OUT_DIR"
echo "============================================================"

cd "$LAB_DIR"
python scripts/rsl_rl/eval_friction_matrix.py \
  --task "$TASK" \
  --checkpoint "$CHECKPOINT" \
  --num_envs "$NUM_ENVS" \
  --warmup_steps "$WARMUP_STEPS" \
  --max_steps "$MAX_STEPS" \
  --seed "$SEED" \
  --vx "${VX_VALUES[@]}" \
  --mu_bins "${MU_VALUES[@]}" \
  --output_csv "$CSV" \
  --device "$DEVICE" \
  --headless 2>&1 | tee "$RAW_LOG"

python - "$CSV" "$SUMMARY" "$CHECKPOINT" <<'PY'
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
checkpoint = sys.argv[3]

with csv_path.open(newline="") as f:
    rows = list(csv.DictReader(f))

numeric = {
    key
    for key in rows[0]
    if key not in {"steps", "seed"}
}
for row in rows:
    for key in numeric:
        row[key] = float(row[key])

stress = [r for r in rows if abs(r["cmd_vx"] - 1.5) < 1e-6]
low = [r for r in stress if r["mu"] <= 0.20]
high = [r for r in stress if r["mu"] >= 0.80]

def avg(group, key):
    return statistics.fmean(float(r[key]) for r in group)

checks = []

def check(name, value, passed, target):
    checks.append((name, value, "PASS" if passed else "WARN", target))

if low and high:
    low_vx = avg(low, "mean_vx")
    high_vx = avg(high, "mean_vx")
    check("高低摩擦速度差", high_vx - low_vx, high_vx - low_vx >= 0.50, ">= 0.50 m/s")
    check("高摩擦前向速度", high_vx, high_vx >= 0.90, ">= 0.90 m/s")
    check("低摩擦限速", low_vx, low_vx <= 0.45, "<= 0.45 m/s")
    check("低摩擦横向速度", avg(low, "mean_abs_vy"), avg(low, "mean_abs_vy") <= 0.20, "<= 0.20 m/s")
    check("低摩擦偏航率", avg(low, "mean_abs_wz"), avg(low, "mean_abs_wz") <= 0.40, "<= 0.40 rad/s")
    check("高摩擦偏航率", avg(high, "mean_abs_wz"), avg(high, "mean_abs_wz") <= 0.35, "<= 0.35 rad/s")
    check(
        "高摩擦横向路径偏移",
        avg(high, "mean_abs_lateral_pos"),
        avg(high, "mean_abs_lateral_pos") <= 0.75,
        "<= 0.75 m",
    )
    check("低摩擦摔倒次数/环境", avg(low, "fall_per_env"), avg(low, "fall_per_env") <= 0.20, "<= 0.20")
    check("高摩擦摔倒次数/环境", avg(high, "fall_per_env"), avg(high, "fall_per_env") <= 0.10, "<= 0.10")

overall = "PASS" if checks and all(c[2] == "PASS" for c in checks) else "NEEDS_TRAINING"
lines = [
    "# TractionAdaptive evaluation",
    "",
    f"- Checkpoint: `{checkpoint}`",
    f"- Overall: **{overall}**",
    "",
    "## Friction matrix",
    "",
    "| μ | cmd | vx | |vy| | |wz| | contact slip | lateral pos | falls/env |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['mu']:.2f} | {r['cmd_vx']:.2f} | {r['mean_vx']:.3f} | "
        f"{r['mean_abs_vy']:.3f} | {r['mean_abs_wz']:.3f} | "
        f"{r['mean_contact_slip']:.3f} | {r['mean_abs_lateral_pos']:.3f} | "
        f"{r['fall_per_env']:.3f} |"
    )
lines += ["", "## Gates", "", "| Gate | Value | Result | Target |", "|---|---:|:---:|---:|"]
for name, value, result, target in checks:
    lines.append(f"| {name} | {value:.3f} | {result} | {target} |")
lines.append("")
summary_path.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(f"[info] summary: {summary_path}")
PY

echo "[ok] evaluation complete: $SUMMARY"
