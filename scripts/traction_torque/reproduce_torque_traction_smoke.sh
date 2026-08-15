#!/usr/bin/env bash
set -euo pipefail

ISAAC_PYTHON="${ISAAC_PYTHON:-/home/mosense/miniconda3/envs/isaaclab-v2/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPRO_DIR="$(realpath -m "${1:-${PROJECT_ROOT}/artifacts/traction_torque/reproduction_seed20260803}")"
MUJOCO_ROOT="/home/mosense/guo/unitree_mujoco"
SEED=20260803

if [[ -e "${REPRO_DIR}" ]]; then
    echo "Refusing to overwrite existing reproduction directory: ${REPRO_DIR}" >&2
    exit 2
fi
mkdir -p "${REPRO_DIR}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/source/unitree_rl_lab${PYTHONPATH:+:${PYTHONPATH}}"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "${ISAAC_PYTHON}" -m pytest -q \
    scripts/tests/test_torque_force_schema.py \
    scripts/tests/test_torque_force_frames.py \
    scripts/tests/test_inverse_dynamics_force_estimator.py \
    scripts/tests/test_contact_estimator.py \
    scripts/tests/test_torque_filter.py \
    scripts/tests/test_torque_traction_state.py \
    scripts/tests/test_torque_traction_networks.py \
    scripts/tests/test_torque_traction_governor.py \
    scripts/tests/test_no_privileged_observation_leak.py \
    scripts/tests/test_torque_traction_export.py

TEACHER_MARKER="${REPRO_DIR}/teacher_start.marker"
touch "${TEACHER_MARKER}"
"${ISAAC_PYTHON}" scripts/traction_torque/train_torque_traction_teacher.py \
    --num_envs 64 --max_iterations 2 --seed "${SEED}" \
    --partial_checkpoint model/rl/model_49999.pt --headless
TEACHER_CKPT="$(find logs/rsl_rl/g1_29dof_torque_traction_teacher -type f -name model_1.pt -newer "${TEACHER_MARKER}" -print | sort | tail -n 1)"
if [[ -z "${TEACHER_CKPT}" ]]; then
    echo "Teacher RSL checkpoint was not generated" >&2
    exit 3
fi

"${ISAAC_PYTHON}" scripts/traction_torque/collect_torque_force_dataset.py \
    --num_envs 12 --steps 400 --warmup_steps 50 --seed "${SEED}" \
    --randomization_stage 0 --benchmark_latency --headless \
    --output "${REPRO_DIR}/dataset_stage0.npz"
"${ISAAC_PYTHON}" scripts/traction_torque/evaluate_analytical_force_estimator.py \
    "${REPRO_DIR}/dataset_stage0.npz" --output "${REPRO_DIR}/analytical_metrics.json"
"${ISAAC_PYTHON}" scripts/traction_torque/train_temporal_force_corrector.py \
    "${REPRO_DIR}/dataset_stage0.npz" --epochs 300 --seed "${SEED}" \
    --output "${REPRO_DIR}/temporal_force_corrector.pt"
"${ISAAC_PYTHON}" scripts/traction_torque/aggregate_torque_traction_dagger.py \
    "${REPRO_DIR}/dataset_stage0.npz" --output "${REPRO_DIR}/dagger_round0.npz"
"${ISAAC_PYTHON}" scripts/traction_torque/distill_torque_traction_student.py \
    "${REPRO_DIR}/dagger_round0.npz" --teacher_checkpoint "${TEACHER_CKPT}" \
    --teacher_epochs 0 --student_epochs 300 \
    --seed "${SEED}" --output "${REPRO_DIR}/torque_student_distilled.pt"

# Generate a checkpoint with the exact RSL model topology, then replace its
# locomotion and auxiliary branches with the baseline and distilled weights.
TEMPLATE_MARKER="${REPRO_DIR}/student_template.marker"
touch "${TEMPLATE_MARKER}"
"${ISAAC_PYTHON}" scripts/traction_torque/train_torque_traction_student.py \
    --num_envs 64 --max_iterations 1 --seed "${SEED}" \
    --partial_checkpoint model/rl/model_49999.pt --headless
TEMPLATE_CKPT="$(find logs/rsl_rl/g1_29dof_torque_traction_student -type f -name model_0.pt -newer "${TEMPLATE_MARKER}" -print | sort | tail -n 1)"
if [[ -z "${TEMPLATE_CKPT}" ]]; then
    echo "Student RSL template checkpoint was not generated" >&2
    exit 4
fi
"${ISAAC_PYTHON}" scripts/traction_torque/build_torque_student_rsl_warmstart.py \
    --template "${TEMPLATE_CKPT}" --baseline model/rl/model_49999.pt \
    --distilled "${REPRO_DIR}/torque_student_distilled.pt" \
    --output "${REPRO_DIR}/torque_student_rsl_warmstart.pt" --seed "${SEED}"

FINAL_MARKER="${REPRO_DIR}/student_final.marker"
touch "${FINAL_MARKER}"
"${ISAAC_PYTHON}" scripts/traction_torque/train_torque_traction_student.py \
    --num_envs 64 --max_iterations 2 --seed "${SEED}" \
    --partial_checkpoint "${REPRO_DIR}/torque_student_rsl_warmstart.pt" --headless
FINAL_CKPT="$(find logs/rsl_rl/g1_29dof_torque_traction_student -type f -name model_1.pt -newer "${FINAL_MARKER}" -print | sort | tail -n 1)"
if [[ -z "${FINAL_CKPT}" ]]; then
    echo "Final Student RSL checkpoint was not generated" >&2
    exit 5
fi
"${ISAAC_PYTHON}" scripts/traction_torque/export_torque_traction_policy.py \
    --rsl_checkpoint "${FINAL_CKPT}" --seed "${SEED}" \
    --output_dir "${REPRO_DIR}/export"

cd "${MUJOCO_ROOT}"
PYTHONPATH="${PROJECT_ROOT}/source/unitree_rl_lab" "${ISAAC_PYTHON}" \
    simulate_python/run_torque_traction_matrix.py \
    --policy "${REPRO_DIR}/export/torque_traction_student.ts" \
    --duration_s 4 --seed "${SEED}" \
    --output_dir "${REPRO_DIR}/mujoco_matrix" \
    --scenarios high_friction low_friction abrupt_friction_drop asymmetric_friction combined_randomization
PYTHONPATH="${PROJECT_ROOT}/source/unitree_rl_lab" "${ISAAC_PYTHON}" \
    simulate_python/compare_torque_traction_matrices.py \
    "${REPRO_DIR}/mujoco_matrix" --output "${REPRO_DIR}/mujoco_matrix_summary.json"

echo "Torque-traction smoke reproduction completed: ${REPRO_DIR}"
