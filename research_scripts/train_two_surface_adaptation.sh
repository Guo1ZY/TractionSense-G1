#!/usr/bin/env bash
# Oracle switch fine-tune -> transition DAgger -> sensor-only candidate evaluation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB="${UNITREE_RL_LAB_DIR:-$ROOT}"
ISAACLAB_PATH="${ISAACLAB_PATH:-${HOME}/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-isaaclab-v2}"
TEACHER_TASK="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-MotionFeedback-Switch"
STUDENT_TASK="Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-Switch"
BASE_TEACHER="${BASE_TEACHER:-$LAB/logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_teacher_motion_balanced_symmetry/2026-07-29_13-44-07_motion_balanced_symmetry_prod_20260729/model_8900.pt}"
BASE_STUDENT="${BASE_STUDENT:-$LAB/logs/evaluations/traction_magnetic_speed_lateral/20260729_recovery/candidates/trust010/shared_magnetic_policy.pt}"
LATERAL_ESTIMATOR="${LATERAL_ESTIMATOR:-$LAB/logs/evaluations/traction_magnetic_motion/20260729_lateral_velocity_estimator/lateral_velocity_estimator.pt}"
FRICTION_ESTIMATOR="${FRICTION_ESTIMATOR:-$LAB/logs/evaluations/traction_magnetic_speed_lateral/20260729_recovery/friction_estimator_1864/friction_estimator.pt}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-4096}"
EVAL_ENVS="${EVAL_ENVS:-64}"
TEACHER_ITERS="${TEACHER_ITERS:-720}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-20}"
PHASE_STEPS="${PHASE_STEPS:-150}"
HIGH_MU="${HIGH_MU:-1.20}"
LOW_MU="${LOW_MU:-0.15}"
VX="${VX:-0.60}"
SEED="${SEED:-314159}"
SMOKE=0
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$LAB/logs/evaluations/friction_switch_training/$RUN_ID}"

usage() {
  echo "Usage: $0 [--smoke] [--base-teacher FILE] [--base-student FILE]"
  echo "          [--lateral-estimator FILE] [--friction-estimator FILE]"
  echo "          [--num-envs N] [--eval-envs N] [--teacher-iters N]"
  echo "          [--student-epochs N] [--phase-steps N]"
  echo "          [--high-mu MU] [--low-mu MU] [--vx MPS]"
  echo "          [--device DEVICE] [--output-dir DIR]"
}

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --smoke) SMOKE=1; shift ;;
    --base-teacher) BASE_TEACHER="$2"; shift 2 ;;
    --base-student) BASE_STUDENT="$2"; shift 2 ;;
    --lateral-estimator) LATERAL_ESTIMATOR="$2"; shift 2 ;;
    --friction-estimator) FRICTION_ESTIMATOR="$2"; shift 2 ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --eval-envs) EVAL_ENVS="$2"; shift 2 ;;
    --teacher-iters) TEACHER_ITERS="$2"; shift 2 ;;
    --student-epochs) STUDENT_EPOCHS="$2"; shift 2 ;;
    --phase-steps) PHASE_STEPS="$2"; shift 2 ;;
    --high-mu) HIGH_MU="$2"; shift 2 ;;
    --low-mu) LOW_MU="$2"; shift 2 ;;
    --vx) VX="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$SMOKE" == "1" ]]; then
  NUM_ENVS=64
  EVAL_ENVS=8
  TEACHER_ITERS=2
  STUDENT_EPOCHS=1
  PHASE_STEPS=30
fi

for required in \
  "$BASE_TEACHER" \
  "$BASE_STUDENT" \
  "$LATERAL_ESTIMATOR" \
  "$FRICTION_ESTIMATOR"; do
  [[ -f "$required" ]] || {
    echo "[ERROR] required input not found: $required" >&2
    exit 2
  }
done
BASE_TEACHER="$(realpath "$BASE_TEACHER")"
BASE_STUDENT="$(realpath "$BASE_STUDENT")"
LATERAL_ESTIMATOR="$(realpath "$LATERAL_ESTIMATOR")"
FRICTION_ESTIMATOR="$(realpath "$FRICTION_ESTIMATOR")"
mkdir -p "$OUTPUT_DIR"
PIPELINE_LOG="$OUTPUT_DIR/pipeline.log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

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

run_name="two_surface_switch_${RUN_ID}"
echo "[1/5] fine-tune transition Teacher"
NUM_ENVS="$NUM_ENVS" MAX_ITERS="$TEACHER_ITERS" DEVICE="$DEVICE" \
  SEED="$SEED" RUN_NAME="$run_name" CHECKPOINT="$BASE_TEACHER" \
  "$ROOT/research_scripts/finetune_friction_switch_teacher.sh"

experiment_root="$LAB/logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_teacher_motion_switch"
mapfile -t run_dirs < <(
  find "$experiment_root" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${run_name}" -printf '%T@ %p\n' | sort -nr | cut -d' ' -f2-
)
[[ ${#run_dirs[@]} -gt 0 ]] || {
  echo "[ERROR] transition Teacher run directory not found" >&2
  exit 3
}
teacher_run="${run_dirs[0]}"
mapfile -t checkpoints < <(
  find "$teacher_run" -maxdepth 1 -type f -name 'model_*.pt' \
    -printf '%T@ %p\n' | sort -nr | cut -d' ' -f2-
)
[[ ${#checkpoints[@]} -gt 0 ]] || {
  echo "[ERROR] no transition Teacher checkpoint found" >&2
  exit 3
}
teacher_checkpoint="${checkpoints[0]}"
echo "$teacher_checkpoint" > "$OUTPUT_DIR/teacher_checkpoint.txt"

echo "[2/5] export transition Teacher"
(
  cd "$LAB"
  python scripts/rsl_rl/play.py \
    --task "$TEACHER_TASK" \
    --checkpoint "$teacher_checkpoint" \
    --export_only --num_envs 8 --device "$DEVICE" --headless
)
teacher_onnx="$teacher_run/exported/policy.onnx"
[[ -f "$teacher_onnx" ]] || {
  echo "[ERROR] exported Teacher missing: $teacher_onnx" >&2
  exit 4
}

collect_switch() {
  local seed="$1"
  local output="$2"
  local csv="${output%.npz}.csv"
  (
    cd "$LAB"
    python scripts/rsl_rl/eval_friction_matrix.py \
      --task "$STUDENT_TASK" \
      --shared_policy "$BASE_STUDENT" \
      --lateral_estimator "$LATERAL_ESTIMATOR" \
      --num_envs "$EVAL_ENVS" \
      --warmup_steps 50 \
      --command_ramp_steps 50 \
      --switch_sequence "$HIGH_MU" "$LOW_MU" "$HIGH_MU" "$LOW_MU" \
      --switch_phase_steps "$PHASE_STEPS" \
      --switch_settle_steps "$((PHASE_STEPS > 25 ? 25 : PHASE_STEPS / 3))" \
      --vx "$VX" \
      --seed "$seed" \
      --collect_stride 3 \
      --collect_dagger_npz "$output" \
      --dagger_execution_teacher_onnx "$teacher_onnx" \
      --output_csv "$csv" \
      --device "$DEVICE" \
      --headless
  )
}

echo "[3/5] collect transition DAgger trajectories"
train_npz="$OUTPUT_DIR/dagger_train.npz"
test_npz="$OUTPUT_DIR/dagger_test.npz"
collect_switch "$SEED" "$train_npz"
collect_switch "$((SEED + 1))" "$test_npz"

echo "[4/5] distill conservative Student and export sensor-only guided candidate"
student_dir="$OUTPUT_DIR/student"
python "$ROOT/research_scripts/fine_tune_shared_magnetic_dagger.py" \
  --base "$BASE_STUDENT" \
  --teacher-onnx "$teacher_onnx" \
  --train "$train_npz" \
  --test "$test_npz" \
  --output-dir "$student_dir" \
  --epochs "$STUDENT_EPOCHS" \
  --batch-size "$([[ "$SMOKE" == "1" ]] && echo 128 || echo 1024)" \
  --learning-rate 5.0e-6 \
  --device "$DEVICE" \
  --seed "$SEED" \
  --motion-feedback \
  --mirror-augmentation \
  --lateral-joint-weight 2.0 \
  --symmetry-coef 0.0 \
  --high-mu-sample-weight 1.0 \
  --high-command-threshold 0.50 \
  --teacher-mix-low 0.25 \
  --teacher-mix-high 0.0 \
  --base-action-coef 5.0 \
  --auxiliary-coef-scale 0.0 \
  --actor-head-only

distilled_policy="$student_dir/shared_magnetic_policy.pt"
[[ -f "$distilled_policy" ]] || {
  echo "[ERROR] distilled Student missing: $distilled_policy" >&2
  exit 5
}

guided_dir="$OUTPUT_DIR/sensor_guided"
python "$ROOT/research_scripts/export_estimator_guided_magnetic_teacher.py" \
  --teacher-onnx "$teacher_onnx" \
  --estimator-pt "$FRICTION_ESTIMATOR" \
  --safe "$BASE_STUDENT" \
  --residual-center 0.06 \
  --residual-sharpness 150 \
  --evidence-center 0.15 \
  --evidence-sharpness 50 \
  --motion-feedback \
  --mu-scale 1.2 \
  --output-dir "$guided_dir"

student_policy="$guided_dir/shared_magnetic_policy.pt"
[[ -f "$student_policy" ]] || {
  echo "[ERROR] sensor-guided candidate missing: $student_policy" >&2
  exit 5
}

echo "[5/5] evaluate sensor-only candidate and foot-sensor ablation"
evaluate_common=(
  python "$LAB/scripts/rsl_rl/eval_friction_matrix.py"
  --task "$STUDENT_TASK"
  --shared_policy "$student_policy"
  --lateral_estimator "$LATERAL_ESTIMATOR"
  --num_envs "$EVAL_ENVS"
  --warmup_steps 50
  --command_ramp_steps 50
  --switch_sequence "$HIGH_MU" "$LOW_MU" "$HIGH_MU"
  --switch_phase_steps "$PHASE_STEPS"
  --switch_settle_steps "$((PHASE_STEPS > 25 ? 25 : PHASE_STEPS / 3))"
  --vx "$VX"
  --seed "$((SEED + 2))"
  --device "$DEVICE"
  --headless
)
(
  cd "$LAB"
  "${evaluate_common[@]}" --output_csv "$OUTPUT_DIR/student_phases.csv"
  "${evaluate_common[@]}" --ablate_foot_sensor \
    --output_csv "$OUTPUT_DIR/student_ablation_phases.csv"
)
for required in \
  "$OUTPUT_DIR/student_phases.csv" \
  "$OUTPUT_DIR/student_phases.summary.md" \
  "$OUTPUT_DIR/student_ablation_phases.csv" \
  "$OUTPUT_DIR/student_ablation_phases.summary.md"; do
  [[ -s "$required" ]] || {
    echo "[ERROR] final evaluation output missing: $required" >&2
    exit 6
  }
done

echo "[DONE] $OUTPUT_DIR"
