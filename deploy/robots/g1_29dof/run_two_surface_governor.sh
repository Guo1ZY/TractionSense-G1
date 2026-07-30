#!/usr/bin/env bash
# Launch the existing G1 policy with the conservative two-state speed governor.
set -euo pipefail

ROBOT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  echo "Usage: G1_REAL_TEST_ACK=YES $0 <low|high|auto> --network <interface> [--log]"
  echo
  echo "  low   : cap forward command at 0.15 m/s"
  echo "  high  : cap forward command at 0.35 m/s"
  echo "  auto  : use the 480-D proprioceptive traction estimator"
  echo
  echo "Set G1_GOVERNOR_POLICY=sensorless to use the 480-D v0 policy."
  echo "While running: RB+DOWN=LOW, RB+UP=HIGH, RB+LEFT=AUTO/reset"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

case "$1" in
  low)
    export G1_TRACTION_MODE=manual_low
    ;;
  high)
    export G1_TRACTION_MODE=manual_high
    ;;
  auto)
    export G1_TRACTION_MODE=auto
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "[ERROR] unknown mode: $1"
    usage
    exit 2
    ;;
esac
shift

if [[ "${G1_REAL_TEST_ACK:-}" != "YES" ]]; then
  echo "[STOP] Real-robot safety acknowledgement is missing."
  echo "Use a load-rated overhead harness, clear the test area, hold the"
  echo "hardware emergency stop, then export G1_REAL_TEST_ACK=YES."
  exit 2
fi

export G1_TRACTION_GOVERNOR=1
if [[ -n "${G1_GOVERNOR_POLICY:-}" ]]; then
  POLICY_MODE="${G1_GOVERNOR_POLICY}"
elif [[ "${G1_TRACTION_MODE}" == "auto" ]]; then
  POLICY_MODE="sensorless"
else
  POLICY_MODE="current"
fi
case "${POLICY_MODE}" in
  current)
    DEFAULT_LOW_SPEED="0.15"
    ;;
  sensorless)
    export G1_POLICY_DIR="config/policy/velocity/v0"
    # MuJoCo showed that v0's stop dead-band is too large at 0.15 m/s.
    DEFAULT_LOW_SPEED="0.20"
    ;;
  *)
    echo "[ERROR] G1_GOVERNOR_POLICY must be current or sensorless."
    exit 2
    ;;
esac

export G1_TRACTION_LOW_SPEED="${G1_TRACTION_LOW_SPEED:-${DEFAULT_LOW_SPEED}}"
export G1_TRACTION_HIGH_SPEED="${G1_TRACTION_HIGH_SPEED:-0.35}"
export G1_TRACTION_ACCEL="${G1_TRACTION_ACCEL:-0.15}"
export G1_TRACTION_DECEL="${G1_TRACTION_DECEL:-0.80}"

if [[ "${G1_TRACTION_MODE}" == "auto" && -z "${G1_TRACTION_FEEDBACK_PATH:-}" ]]; then
  # A classifier trained only in MuJoCo must never be selected implicitly on
  # hardware.  The explicit real classifier is installed only after collecting
  # labeled proprioception on the exact two physical floors.
  DEFAULT_CLASSIFIER="${ROBOT_DIR}/config/traction/real_classifier.onnx"
  export G1_TRACTION_PROPRIO_CLASSIFIER_ONNX="${G1_TRACTION_PROPRIO_CLASSIFIER_ONNX:-${DEFAULT_CLASSIFIER}}"
  if [[ ! -f "${G1_TRACTION_PROPRIO_CLASSIFIER_ONNX}" ]]; then
    echo "[ERROR] real-floor-calibrated 480-D traction classifier missing:"
    echo "        ${G1_TRACTION_PROPRIO_CLASSIFIER_ONNX}"
    echo "Collect both floors first with collect_two_surface_proprio.sh,"
    echo "then train/install with train_real_two_surface_classifier.sh."
    echo "The MuJoCo classifier is deliberately not a real-robot default."
    exit 2
  fi
fi

mkdir -p "${ROBOT_DIR}/log"
if [[ -z "${G1_TRACTION_GOVERNOR_LOG:-}" ]]; then
  export G1_TRACTION_GOVERNOR_LOG="${ROBOT_DIR}/log/traction_governor_$(date +%Y%m%d_%H%M%S).csv"
fi

echo "=============================================================="
echo " G1 two-surface governor"
echo " mode       : ${G1_TRACTION_MODE}"
echo " policy     : ${POLICY_MODE}${G1_POLICY_DIR:+ (${G1_POLICY_DIR})}"
echo " low/high   : ${G1_TRACTION_LOW_SPEED} / ${G1_TRACTION_HIGH_SPEED} m/s"
echo " accel/decel: ${G1_TRACTION_ACCEL} / ${G1_TRACTION_DECEL} m/s^2"
echo " classifier : ${G1_TRACTION_PROPRIO_CLASSIFIER_ONNX:-external/none}"
echo " CSV        : ${G1_TRACTION_GOVERNOR_LOG}"
echo " controls   : RB+DOWN LOW | RB+UP HIGH | RB+LEFT AUTO/reset"
echo " emergency  : B -> Passive, plus hardware E-stop operator"
echo "=============================================================="

exec "${ROBOT_DIR}/run_g1_ctrl.sh" "$@"
