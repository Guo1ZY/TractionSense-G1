#!/usr/bin/env bash
# Guarded launcher for the inactive r26-recovery + r25-Hall-risk candidate.
set -euo pipefail

ROBOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CANDIDATE_REL="config/policy/velocity/hall_traction_r26_r25_candidate"
CANDIDATE_DIR="${ROBOT_DIR}/${CANDIDATE_REL}"
PREFLIGHT="/home/mosense/guo/scripts/check_real_magnetic_preflight.py"
PYTHON_BIN="${G1_HALL_PYTHON:-/home/mosense/miniconda3/envs/isaaclab-v2/bin/python}"
DEFAULT_HALL_CONFIG="/home/mosense/guo_1/ble_sensor/config.magnetic.json"
if [[ ! -f "${DEFAULT_HALL_CONFIG}" ]]; then
  DEFAULT_HALL_CONFIG="/home/mosense/guo_1/vola_sensor/config/dual_foot_real.json"
fi
HALL_CONFIG="${G1_HALL_CONFIG:-${DEFAULT_HALL_CONFIG}}"
HALL_HEALTH="${G1_HALL_HEALTH:-/tmp/g1_foot_magnetic_health.json}"
HALL_PACKET="${G1_HALL_PACKET:-/tmp/g1_foot_rl_obs.bin}"

usage() {
  echo "Usage:"
  echo "  $0 --preflight-only"
  echo "  G1_HALL_HARNESS_ACK=HALL_B_ONLY_HARNESS $0 --execute --network <interface> [--log]"
  echo
  echo "The live F0M1 bridge must already be running. The launcher never treats"
  echo "Hall Bx/By/Bz as normal or tangential force. First motion must use a"
  echo "load-rated overhead harness and a separate hardware E-stop operator."
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

case "$1" in
  --preflight-only)
    EXECUTE=0
    ;;
  --execute)
    EXECUTE=1
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "[ERROR] first argument must be --preflight-only or --execute"
    usage
    exit 2
    ;;
esac
shift

if [[ ! -f "${CANDIDATE_DIR}/exported/policy.onnx" || \
      ! -f "${CANDIDATE_DIR}/exported/hall_risk.onnx" ]]; then
  echo "[STOP] packaged Hall candidate is incomplete: ${CANDIDATE_DIR}"
  exit 2
fi

# Explicit selection is scoped to this child process. The repository's
# config/config.yaml remains on the existing foot policy for instant rollback.
export G1_POLICY_DIR="${CANDIDATE_REL}"
export G1_TRACTION_HALL_RISK_ONNX="${CANDIDATE_DIR}/exported/hall_risk.onnx"
export G1_TRACTION_HALL_RISK_ALPHA="${G1_TRACTION_HALL_RISK_ALPHA:-0.20}"

"${PYTHON_BIN}" "${PREFLIGHT}" \
  --candidate-slot "${CANDIDATE_DIR}" \
  --config "${HALL_CONFIG}" \
  --health "${HALL_HEALTH}" \
  --packet "${HALL_PACKET}" \
  --require-policy-active

if [[ "${EXECUTE}" -eq 0 ]]; then
  echo "[PASS] Hall candidate preflight only; no robot process was started."
  exit 0
fi

if [[ "${G1_HALL_HARNESS_ACK:-}" != "HALL_B_ONLY_HARNESS" ]]; then
  echo "[STOP] real-robot harness acknowledgement is missing."
  echo "Set G1_HALL_HARNESS_ACK=HALL_B_ONLY_HARNESS only after the robot is"
  echo "secured in an overhead harness and the E-stop operator is ready."
  exit 2
fi

# Prevent an old force/proprio friction estimator from becoming a hidden
# fallback. Full/partial Hall loss is handled inside the r25 model as risk=1.
unset G1_TRACTION_PROPRIO_CLASSIFIER_ONNX
unset G1_TRACTION_PROPRIO_ESTIMATOR_ONNX
unset G1_FRICTION_ESTIMATOR_ONNX
unset G1_TRACTION_FEEDBACK_PATH

export G1_TRACTION_GOVERNOR=1
export G1_TRACTION_MODE=auto
export G1_TRACTION_LOW_SPEED="${G1_TRACTION_LOW_SPEED:-0.22}"
export G1_TRACTION_HIGH_SPEED="${G1_TRACTION_HIGH_SPEED:-0.60}"
export G1_TRACTION_LOW_LATERAL="${G1_TRACTION_LOW_LATERAL:-0.05}"
export G1_TRACTION_HIGH_LATERAL="${G1_TRACTION_HIGH_LATERAL:-0.35}"
export G1_TRACTION_LOW_YAW="${G1_TRACTION_LOW_YAW:-0.15}"
export G1_TRACTION_HIGH_YAW="${G1_TRACTION_HIGH_YAW:-0.80}"
export G1_TRACTION_ACCEL="${G1_TRACTION_ACCEL:-1.50}"
export G1_TRACTION_DECEL="${G1_TRACTION_DECEL:-1.00}"
export G1_TRACTION_PROBE_SPEED="${G1_TRACTION_PROBE_SPEED:-0.50}"

mkdir -p "${ROBOT_DIR}/log"
export G1_TRACTION_GOVERNOR_LOG="${G1_TRACTION_GOVERNOR_LOG:-${ROBOT_DIR}/log/hall_traction_governor.csv}"

echo "=============================================================="
echo " G1 Hall-only traction candidate: r26 recovery + r25 risk"
echo " policy     : ${G1_POLICY_DIR}"
echo " governor   : AUTO, low/high ${G1_TRACTION_LOW_SPEED}/${G1_TRACTION_HIGH_SPEED} m/s"
echo " Hall input : dual-foot 15x3 Bx/By/Bz history; no force inverse"
echo " fault      : invalid/full-foot loss -> risk=1 -> command zero"
echo " status     : simulation safety/adaptation PASS; performance target partial"
echo " emergency  : B -> Passive plus independent hardware E-stop"
echo "=============================================================="

exec "${ROBOT_DIR}/run_g1_ctrl.sh" "$@"
