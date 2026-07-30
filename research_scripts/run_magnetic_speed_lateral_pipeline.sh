#!/usr/bin/env bash
# End-to-end Oracle Teacher -> magnetic DAgger Student -> Isaac/MuJoCo selection.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB="$ROOT"
ISAACLAB_PATH="${ISAACLAB_PATH:-${HOME}/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-isaaclab-v2}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-4096}"
EVAL_ENVS="${EVAL_ENVS:-64}"
TEACHER_ITERS="${TEACHER_ITERS:-200}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-$LAB/logs/evaluations/traction_magnetic_speed_lateral/$RUN_ID}"
TEACHER_TASK="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-SpeedLateral"
STUDENT_TASK="Unitree-G1-29dof-Velocity-Foot-TractionMagneticStudent"
TEACHER_ROOT="$LAB/logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_teacher_speed_lateral"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-$LAB/logs/imported_remote_20260728/model_8110.pt}"
BASE_SHARED="${BASE_SHARED:-$LAB/logs/evaluations/traction_shared_magnetic/20260728_shared15x3_lateral8110_blend50/shared_magnetic_policy.pt}"
OLD_ISAAC="${OLD_ISAAC:-$LAB/logs/evaluations/traction_shared_magnetic/20260728_shared15x3_lateral8110_dagger_data/train.npz}"
OLD_MUJOCO="${OLD_MUJOCO:-$LAB/logs/evaluations/traction_shared_magnetic/20260728_dagger1_mujoco_dagger_data/train.npz}"
PIPELINE_LOG="$RESULT_ROOT/pipeline.log"

mkdir -p "$RESULT_ROOT"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

for required in "$BASE_CHECKPOINT" "$BASE_SHARED" "$OLD_ISAAC" "$OLD_MUJOCO"; do
  [[ -f "$required" ]] || {
    echo "[ERROR] missing required input: $required"
    exit 2
  }
done

source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export ISAACLAB_PATH
unset PYTHONPATH || true
if [[ -f "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" >/dev/null 2>&1 || true
  set -u
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] NVIDIA driver is not ready. Reboot once, then rerun this script."
  exit 70
fi
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the Isaac Lab environment")
print(f"[GPU] {torch.cuda.get_device_name(0)}")
PY

echo "[RUN] id=$RUN_ID results=$RESULT_ROOT"

teacher_run_name="speed_lateral_prod_${RUN_ID}"
"$ROOT/research_scripts/finetune_speed_lateral_teacher.sh" \
  --checkpoint "$BASE_CHECKPOINT" \
  --num-envs "$NUM_ENVS" \
  --max-iterations "$TEACHER_ITERS" \
  --device "$DEVICE" \
  --seed 8291 \
  --run-name "$teacher_run_name"

mapfile -t teacher_runs < <(
  find "$TEACHER_ROOT" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${teacher_run_name}" -printf '%T@ %p\n' |
    sort -nr | cut -d' ' -f2-
)
[[ ${#teacher_runs[@]} -gt 0 ]] || {
  echo "[ERROR] Teacher run directory was not created"
  exit 3
}
teacher_run="${teacher_runs[0]}"
base_iter="$(basename "$BASE_CHECKPOINT" | sed -E 's/^model_([0-9]+)\.pt$/\1/')"
[[ "$base_iter" =~ ^[0-9]+$ ]] || {
  echo "[ERROR] cannot infer baseline iteration from $BASE_CHECKPOINT"
  exit 3
}

declare -A teacher_checkpoints
teacher_specs=()
teacher_offsets=(50 100 150)
(( TEACHER_ITERS > 0 )) && teacher_offsets+=("$((TEACHER_ITERS - 1))")
for offset in "${teacher_offsets[@]}"; do
  (( offset >= 0 && offset < TEACHER_ITERS )) || continue
  iteration=$((base_iter + offset))
  checkpoint="$teacher_run/model_${iteration}.pt"
  [[ -f "$checkpoint" ]] || continue
  name="model_${iteration}"
  teacher_checkpoints["$name"]="$checkpoint"
  csvs=()
  for seed in 314159 271828; do
    out="$RESULT_ROOT/teacher/$name/isaac_seed${seed}.csv"
    mkdir -p "$(dirname "$out")"
    (
      cd "$LAB"
      python scripts/rsl_rl/eval_friction_matrix.py \
        --task "$TEACHER_TASK" \
        --checkpoint "$checkpoint" \
        --num_envs "$EVAL_ENVS" \
        --max_steps 250 \
        --warmup_steps 50 \
        --command_ramp_steps 50 \
        --seed "$seed" \
        --vx 0.5 1.0 \
        --mu_bins 0.08 0.20 0.40 0.80 1.20 \
        --output_csv "$out" \
        --device "$DEVICE" \
        --headless
    )
    csvs+=("$out")
  done
  teacher_specs+=(--candidate "$name=$(IFS=,; echo "${csvs[*]}")")
done
[[ ${#teacher_specs[@]} -gt 0 ]] || {
  echo "[ERROR] no Teacher checkpoint was available for evaluation"
  exit 4
}
python "$ROOT/research_scripts/select_friction_matrix_candidate.py" \
  "${teacher_specs[@]}" \
  --max-high-abs-vy 0.15 \
  --max-high-lateral 0.20 \
  --output "$RESULT_ROOT/teacher/selection.json" \
  --selected-name "$RESULT_ROOT/teacher/selected_name.txt"
teacher_name="$(tr -d '\r\n' < "$RESULT_ROOT/teacher/selected_name.txt")"
teacher_checkpoint="${teacher_checkpoints[$teacher_name]}"
echo "$teacher_checkpoint" > "$RESULT_ROOT/teacher/selected_checkpoint.txt"

(
  cd "$LAB"
  python scripts/rsl_rl/play.py \
    --task "$TEACHER_TASK" \
    --checkpoint "$teacher_checkpoint" \
    --export_only \
    --num_envs 16 \
    --device "$DEVICE" \
    --headless
)
teacher_onnx="$(dirname "$teacher_checkpoint")/exported/policy.onnx"
[[ -f "$teacher_onnx" ]] || {
  echo "[ERROR] Teacher ONNX export missing: $teacher_onnx"
  exit 5
}

collect_dagger() {
  local shared="$1"
  local seed="$2"
  local output="$3"
  local csv="${output%.npz}.csv"
  mkdir -p "$(dirname "$output")"
  (
    cd "$LAB"
    python scripts/rsl_rl/eval_friction_matrix.py \
      --task "$STUDENT_TASK" \
      --shared_policy "$shared" \
      --num_envs "$EVAL_ENVS" \
      --max_steps 300 \
      --warmup_steps 60 \
      --command_ramp_steps 60 \
      --collect_stride 4 \
      --collect_dagger_npz "$output" \
      --seed "$seed" \
      --vx 0.30 0.70 1.00 \
      --mu_bins 0.08 0.20 0.40 0.80 1.20 \
      --output_csv "$csv" \
      --device "$DEVICE" \
      --headless
  )
}

dagger_train="$RESULT_ROOT/dagger/round1_seed314159.npz"
dagger_test="$RESULT_ROOT/dagger/round1_seed271828.npz"
collect_dagger "$BASE_SHARED" 314159 "$dagger_train"
collect_dagger "$BASE_SHARED" 271828 "$dagger_test"

declare -A student_models
declare -A student_scales
student_specs=()
for scale in 1.00 1.03 1.06; do
  tag="scale${scale/./}"
  output="$RESULT_ROOT/student_round1/$tag"
  python "$ROOT/research_scripts/fine_tune_shared_magnetic_dagger.py" \
    --base "$BASE_SHARED" \
    --teacher-onnx "$teacher_onnx" \
    --train "$dagger_train" \
    --test "$dagger_test" \
    --augment "$OLD_ISAAC" \
    --augment "$OLD_MUJOCO" \
    --output-dir "$output" \
    --epochs 35 \
    --batch-size 1024 \
    --learning-rate 3.0e-5 \
    --device "$DEVICE" \
    --seed 8291 \
    --mirror-augmentation \
    --lateral-joint-weight 2.0 \
    --symmetry-coef 0.15 \
    --high-mu-command-scale "$scale" \
    --high-mu-sample-weight 6.0
  student_models["$tag"]="$output/shared_magnetic_policy.pt"
  student_scales["$tag"]="$scale"
done

evaluate_student_isaac() {
  local name="$1"
  local model="$2"
  local specs_name="$3"
  local csvs=()
  for seed in 314159 271828; do
    local out="$RESULT_ROOT/student_eval/$name/isaac_seed${seed}.csv"
    mkdir -p "$(dirname "$out")"
    (
      cd "$LAB"
      python scripts/rsl_rl/eval_friction_matrix.py \
        --task "$STUDENT_TASK" \
        --shared_policy "$model" \
        --num_envs "$EVAL_ENVS" \
        --max_steps 250 \
        --warmup_steps 50 \
        --command_ramp_steps 50 \
        --seed "$seed" \
        --vx 0.5 1.0 \
        --mu_bins 0.08 0.20 0.40 0.80 1.20 \
        --output_csv "$out" \
        --device "$DEVICE" \
        --headless
    )
    csvs+=("$out")
  done
  printf -v "$specs_name" '%s=%s' "$name" "$(IFS=,; echo "${csvs[*]}")"
}

for tag in scale100 scale103 scale106; do
  spec=""
  evaluate_student_isaac "$tag" "${student_models[$tag]}" spec
  student_specs+=(--candidate "$spec")
done
python "$ROOT/research_scripts/select_friction_matrix_candidate.py" \
  "${student_specs[@]}" \
  --max-high-abs-vy 0.15 \
  --max-high-lateral 0.20 \
  --output "$RESULT_ROOT/student_round1/selection.json" \
  --selected-name "$RESULT_ROOT/student_round1/selected_name.txt"
round1_name="$(tr -d '\r\n' < "$RESULT_ROOT/student_round1/selected_name.txt")"
round1_model="${student_models[$round1_name]}"
round1_scale="${student_scales[$round1_name]}"

dagger_round2="$RESULT_ROOT/dagger/round2_seed424242.npz"
collect_dagger "$round1_model" 424242 "$dagger_round2"
refined_dir="$RESULT_ROOT/student_round2/refined_${round1_name}"
python "$ROOT/research_scripts/fine_tune_shared_magnetic_dagger.py" \
  --base "$round1_model" \
  --teacher-onnx "$teacher_onnx" \
  --train "$dagger_round2" \
  --test "$dagger_test" \
  --augment "$dagger_train" \
  --augment "$OLD_MUJOCO" \
  --output-dir "$refined_dir" \
  --epochs 20 \
  --batch-size 1024 \
  --learning-rate 1.5e-5 \
  --device "$DEVICE" \
  --seed 8292 \
  --mirror-augmentation \
  --lateral-joint-weight 2.5 \
  --symmetry-coef 0.20 \
  --high-mu-command-scale "$round1_scale" \
  --high-mu-sample-weight 6.0
refined_model="$refined_dir/shared_magnetic_policy.pt"

round1_spec=""
refined_spec=""
evaluate_student_isaac "round1_best" "$round1_model" round1_spec
evaluate_student_isaac "round2_refined" "$refined_model" refined_spec

declare -A finalist_models=(
  [round1_best]="$round1_model"
  [round2_refined]="$refined_model"
)
cross_specs=()
for name in round1_best round2_refined; do
  model="${finalist_models[$name]}"
  slot="traction_magnetic_speed_lateral_${RUN_ID}_${name}"
  model_dir="$(dirname "$model")"
  python "$ROOT/research_scripts/install_shared_magnetic_policy.py" \
    --model-dir "$model_dir" \
    --slot "$slot"
  mujoco_out="$RESULT_ROOT/student_eval/$name/mujoco"
  python "$ROOT/research_scripts/mujoco_friction_speed_matrix.py" \
    --skip-export \
    --slot "$slot" \
    --magnetic-bridge \
    --disable-command-slew \
    --mus 0.08 0.20 0.40 0.80 1.20 \
    --speeds 0.1 0.5 1.0 \
    --output-dir "$mujoco_out"
  isaac_spec="$round1_spec"
  [[ "$name" == "round2_refined" ]] && isaac_spec="$refined_spec"
  isaac_csvs="${isaac_spec#*=}"
  cross_specs+=(--candidate "$name=$isaac_csvs,$mujoco_out/matrix.csv")
done

python "$ROOT/research_scripts/select_friction_matrix_candidate.py" \
  "${cross_specs[@]}" \
  --max-high-abs-vy 0.20 \
  --max-high-lateral 0.25 \
  --output "$RESULT_ROOT/final_selection.json" \
  --selected-name "$RESULT_ROOT/selected_name.txt"
final_name="$(tr -d '\r\n' < "$RESULT_ROOT/selected_name.txt")"
final_model="${finalist_models[$final_name]}"
final_dir="$RESULT_ROOT/best_model"
mkdir -p "$final_dir"
cp -a "$(dirname "$final_model")/policy.onnx" "$final_dir/policy.onnx"
cp -a "$final_model" "$final_dir/shared_magnetic_policy.pt"
cp -a "$(dirname "$final_model")/metrics.json" "$final_dir/distillation_metrics.json"
echo "$teacher_checkpoint" > "$final_dir/teacher_checkpoint.txt"
echo "$teacher_onnx" > "$final_dir/teacher_onnx.txt"
echo "$final_model" > "$RESULT_ROOT/BEST_MODEL.txt"
python "$ROOT/research_scripts/install_shared_magnetic_policy.py" \
  --model-dir "$final_dir" \
  --slot traction_magnetic_speed_lateral_best

echo "[DONE] best model: $final_model"
echo "[DONE] final report: $RESULT_ROOT/final_selection.json"
echo "[SAFE] deploy slot prepared but not activated: traction_magnetic_speed_lateral_best"
