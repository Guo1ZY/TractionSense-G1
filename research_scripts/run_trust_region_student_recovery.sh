#!/usr/bin/env bash
# Recover a safe magnetic Student after unconstrained DAgger destabilized ice.
set -euo pipefail

ROOT="${TRACTIONSENSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LAB="$ROOT"
ISAACLAB_PATH="${ISAACLAB_PATH:-${HOME}/IsaacLab}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-isaaclab-v2}"
DEVICE="${DEVICE:-cuda:0}"
EVAL_ENVS="${EVAL_ENVS:-64}"
RUN_ROOT="${RUN_ROOT:-$LAB/logs/evaluations/traction_magnetic_speed_lateral/20260729_recovery}"
TASK="Unitree-G1-29dof-Velocity-Foot-TractionMagneticStudent"
BASE="$LAB/logs/evaluations/traction_shared_magnetic/20260728_shared15x3_lateral8110_blend50/shared_magnetic_policy.pt"
TEACHER_ONNX="$LAB/logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_teacher_speed_lateral/2026-07-29_10-35-30_speed_lateral_prod_20260729_production/exported/policy.onnx"
TRAIN="$LAB/logs/evaluations/traction_magnetic_speed_lateral/20260729_production/dagger/round1_seed314159.npz"
TEST="$LAB/logs/evaluations/traction_magnetic_speed_lateral/20260729_production/dagger/round1_seed271828.npz"
MUJOCO_AUGMENT="$LAB/logs/evaluations/traction_shared_magnetic/20260728_dagger1_mujoco_dagger_data/train.npz"
LOG="$RUN_ROOT/recovery.log"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$LOG") 2>&1
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export ISAACLAB_PATH
unset PYTHONPATH || true
if [[ -f "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" >/dev/null 2>&1 || true
  set -u
fi
nvidia-smi >/dev/null

declare -A models=([base]="$BASE")
for mix in 0.10 0.20 0.30; do
  name="trust${mix/./}"
  output="$RUN_ROOT/candidates/$name"
  python "$ROOT/research_scripts/fine_tune_shared_magnetic_dagger.py" \
    --base "$BASE" \
    --teacher-onnx "$TEACHER_ONNX" \
    --train "$TRAIN" \
    --test "$TEST" \
    --augment "$MUJOCO_AUGMENT" \
    --output-dir "$output" \
    --epochs 10 \
    --batch-size 1024 \
    --learning-rate 5.0e-6 \
    --device "$DEVICE" \
    --seed 8301 \
    --lateral-joint-weight 1.25 \
    --high-mu-command-scale 1.00 \
    --high-mu-sample-weight 3.0 \
    --teacher-mix-low 0.00 \
    --teacher-mix-high "$mix" \
    --base-action-coef 0.75 \
    --actor-head-only \
    --auxiliary-coef-scale 0.0
  models["$name"]="$output/shared_magnetic_policy.pt"
done

evaluate_isaac() {
  local phase="$1"
  local name="$2"
  local model="$3"
  local mus="$4"
  local result_var="$5"
  local csvs=()
  for seed in 314159 271828; do
    local output="$RUN_ROOT/$phase/$name/seed${seed}.csv"
    mkdir -p "$(dirname "$output")"
    (
      cd "$LAB"
      # shellcheck disable=SC2086
      python scripts/rsl_rl/eval_friction_matrix.py \
        --task "$TASK" \
        --shared_policy "$model" \
        --num_envs "$EVAL_ENVS" \
        --max_steps 250 \
        --warmup_steps 50 \
        --command_ramp_steps 50 \
        --seed "$seed" \
        --vx 0.5 1.0 \
        --mu_bins $mus \
        --output_csv "$output" \
        --device "$DEVICE" \
        --headless
    )
    csvs+=("$output")
  done
  printf -v "$result_var" '%s=%s' "$name" "$(IFS=,; echo "${csvs[*]}")"
}

smoke_specs=()
for name in base trust010 trust020 trust030; do
  spec=""
  evaluate_isaac smoke "$name" "${models[$name]}" "0.08 1.20" spec
  smoke_specs+=(--candidate "$spec")
done
python "$ROOT/research_scripts/select_friction_matrix_candidate.py" \
  "${smoke_specs[@]}" \
  --max-high-abs-vy 0.15 \
  --max-high-lateral 0.20 \
  --output "$RUN_ROOT/smoke_selection.json" \
  --selected-name "$RUN_ROOT/smoke_selected.txt"

mapfile -t finalists < <(
  python - "$RUN_ROOT/smoke_selection.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
safe = [item for item in report["candidates"] if item["gates"]["zero_falls"]]
pool = safe if safe else report["candidates"]
for item in pool[:2]:
    print(item["name"])
PY
)
[[ ${#finalists[@]} -gt 0 ]] || {
  echo "[ERROR] no recovery finalist"
  exit 3
}

full_specs=()
declare -A full_isaac_csvs
for name in "${finalists[@]}"; do
  spec=""
  evaluate_isaac full_isaac "$name" "${models[$name]}" \
    "0.08 0.20 0.40 0.80 1.20" spec
  full_specs+=(--candidate "$spec")
  full_isaac_csvs["$name"]="${spec#*=}"
done
python "$ROOT/research_scripts/select_friction_matrix_candidate.py" \
  "${full_specs[@]}" \
  --max-high-abs-vy 0.15 \
  --max-high-lateral 0.20 \
  --output "$RUN_ROOT/isaac_selection.json" \
  --selected-name "$RUN_ROOT/isaac_selected.txt"

cross_specs=()
for name in "${finalists[@]}"; do
  model="${models[$name]}"
  if [[ "$name" == "base" ]]; then
    model_dir="$(dirname "$BASE")"
  else
    model_dir="$(dirname "$model")"
  fi
  slot="traction_magnetic_recovery_${name}"
  python "$ROOT/research_scripts/install_shared_magnetic_policy.py" \
    --model-dir "$model_dir" \
    --slot "$slot"
  output="$RUN_ROOT/mujoco/$name"
  python "$ROOT/research_scripts/mujoco_friction_speed_matrix.py" \
    --skip-export \
    --slot "$slot" \
    --magnetic-bridge \
    --disable-command-slew \
    --mus 0.08 0.20 0.40 0.80 1.20 \
    --speeds 0.1 0.5 1.0 \
    --output-dir "$output"
  cross_specs+=(
    --candidate
    "$name=${full_isaac_csvs[$name]},$output/matrix.csv"
  )
done

python "$ROOT/research_scripts/select_friction_matrix_candidate.py" \
  "${cross_specs[@]}" \
  --max-high-abs-vy 0.20 \
  --max-high-lateral 0.25 \
  --output "$RUN_ROOT/final_selection.json" \
  --selected-name "$RUN_ROOT/selected_name.txt"
selected="$(tr -d '\r\n' < "$RUN_ROOT/selected_name.txt")"
selected_model="${models[$selected]}"
selected_dir="$(dirname "$selected_model")"
best="$RUN_ROOT/best_model"
mkdir -p "$best"
cp -a "$selected_model" "$best/shared_magnetic_policy.pt"
cp -a "$selected_dir/policy.onnx" "$best/policy.onnx"
if [[ -f "$selected_dir/metrics.json" ]]; then
  cp -a "$selected_dir/metrics.json" "$best/distillation_metrics.json"
fi
echo "$selected_model" > "$RUN_ROOT/BEST_MODEL.txt"
python "$ROOT/research_scripts/install_shared_magnetic_policy.py" \
  --model-dir "$best" \
  --slot traction_magnetic_speed_lateral_best
echo "[DONE] recovery best: $selected_model"
