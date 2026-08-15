#!/usr/bin/env bash
# Guarded launcher for the R5 transition-retention candidate.
set -euo pipefail

ROBOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CANDIDATE_REL="config/policy/velocity/transition_retention_r5_candidate"
CANDIDATE_DIR="${ROBOT_DIR}/${CANDIDATE_REL}"

usage() {
  echo "Usage:"
  echo "  $0 --preflight-only"
  echo "  G1_HALL_HARNESS_ACK=HALL_B_ONLY_HARNESS $0 --execute --network <interface> [--log]"
  echo
  echo "The live F0M1 dual-foot bridge must already be running (BLE=1/1)."
  echo "First motion requires a load-rated overhead harness and a separate"
  echo "hardware E-stop operator."
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

case "$1" in
  --preflight-only) EXECUTE=0 ;;
  --execute) EXECUTE=1 ;;
  -h|--help) usage; exit 0 ;;
  *) echo "[ERROR] first argument must be --preflight-only or --execute"; usage; exit 2 ;;
esac
shift

if [[ ! -f "${CANDIDATE_DIR}/exported/policy.onnx" || \
      ! -f "${CANDIDATE_DIR}/params/deploy.yaml" || \
      ! -f "${CANDIDATE_DIR}/exported/lateral_velocity_estimator.onnx" ]]; then
  echo "[STOP] R5 candidate package is incomplete: ${CANDIDATE_DIR}"
  exit 2
fi

# Scoped to this child process; config/config.yaml keeps the official policy
# for instant rollback.
export G1_POLICY_DIR="${CANDIDATE_REL}"
# The 1864-D policy's final channels are [body_vy, relative_heading].  body_vy
# has no world-velocity sidecar on hardware, so the contact-aided estimator
# bundled beside policy.onnx is the deployable source.  The C++ preflight is
# fail-closed without it.
export G1_LATERAL_VELOCITY_ESTIMATOR_ONNX="${CANDIDATE_DIR}/exported/lateral_velocity_estimator.onnx"
# EMA smoothing gain for the estimator output; 0.35 is the C++ default.
export G1_LATERAL_VELOCITY_ESTIMATOR_ALPHA="${G1_LATERAL_VELOCITY_ESTIMATOR_ALPHA:-0.35}"
# First-run conservative stick scaling: full stick ≈ 0.5 m/s forward and
# 0.5 rad/s yaw.  Override via environment once the operator is comfortable.
export G1_CMD_GAIN_LIN="${G1_CMD_GAIN_LIN:-0.5}"
export G1_CMD_GAIN_YAW="${G1_CMD_GAIN_YAW:-0.5}"
# R5 has no external risk model; the gate lives inside the ONNX.  Explicitly
# unset every legacy estimator/governor so a stale fallback cannot engage.
unset G1_TRACTION_HALL_RISK_ONNX
unset G1_TRACTION_PROPRIO_CLASSIFIER_ONNX
unset G1_TRACTION_PROPRIO_ESTIMATOR_ONNX
unset G1_FRICTION_ESTIMATOR_ONNX
unset G1_TRACTION_FEEDBACK_PATH
unset G1_TRACTION_GOVERNOR

echo "=============================================================="
echo " G1 R5 transition-retention candidate (1864-D Hall/proprio)"
echo " policy : ${G1_POLICY_DIR}"
echo " gate   : in-policy; no external risk model"
echo " fault  : dual-foot stale -> that foot zeroed -> policy fail-closed"
echo " body_vy: lateral_velocity_estimator.onnx (1862->1, alpha=0.35)"
echo " stick  : gain_lin=${G1_CMD_GAIN_LIN} gain_yaw=${G1_CMD_GAIN_YAW} (full=0.5 m/s, 0.5 rad/s)"
echo " status : simulation acceptance PASS for mu in [0.20,0.28]"
echo " bringup: harness + hardware E-stop REQUIRED; low-speed only"
echo "=============================================================="

if [[ "${EXECUTE}" -eq 0 ]]; then
  echo "[PASS] R5 package preflight only; no robot process was started."
  exit 0
fi

if [[ "${G1_HALL_HARNESS_ACK:-}" != "HALL_B_ONLY_HARNESS" ]]; then
  echo "[STOP] real-robot harness acknowledgement is missing."
  echo "Set G1_HALL_HARNESS_ACK=HALL_B_ONLY_HARNESS only after the robot is"
  echo "secured in an overhead harness and the E-stop operator is ready."
  exit 2
fi

exec "${ROBOT_DIR}/run_g1_ctrl.sh" "$@"
