#!/usr/bin/env bash
# Collect same-condition 480-D proprioception on one known physical floor.
set -euo pipefail

ROBOT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${ROBOT_DIR}/../../.." && pwd)"
PYTHON="${G1_PYTHON:-python3}"
CONVERTER="${ROOT}/research_scripts/convert_g1_obs1_to_labeled_npz.py"

usage() {
  echo "Usage: G1_REAL_TEST_ACK=YES $0 <low|high> --network <interface>"
  echo
  echo "The robot always uses the sensorless v0 policy and the same manual-low"
  echo "cap (default 0.20 m/s). Hold the same forward joystick input on both"
  echo "known floors for at least 20 s, then press B and Ctrl-C."
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 3 || "$1" != "low" && "$1" != "high" ]]; then
  usage
  exit 2
fi
FLOOR="$1"
shift

if [[ "${G1_REAL_TEST_ACK:-}" != "YES" ]]; then
  echo "[STOP] Set G1_REAL_TEST_ACK=YES only after installing a load-rated"
  echo "overhead harness and assigning a hardware E-stop operator."
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
CAP="${G1_COLLECTION_SPEED:-0.20}"
OUT_DIR="${G1_COLLECTION_DIR:-${ROOT}/logs/real/traction_two_surface/${STAMP}_${FLOOR}_cap${CAP}}"
RAW="${OUT_DIR}/policy_obs.bin"
LABELED="${OUT_DIR}/${FLOOR}.npz"
mkdir -p "${OUT_DIR}"

export G1_GOVERNOR_POLICY=sensorless
export G1_TRACTION_LOW_SPEED="${CAP}"
export G1_POLICY_OBS_FILE="${RAW}"
export G1_TRACTION_GOVERNOR_LOG="${OUT_DIR}/governor.csv"

finalize() {
  if [[ -s "${RAW}" ]]; then
    "${PYTHON}" "${CONVERTER}" \
      --input "${RAW}" \
      --output "${LABELED}" \
      --floor "${FLOOR}" \
      --command "${CAP}" || true
    echo "[DATA] ${LABELED}"
  else
    echo "[WARN] no observation data were recorded: ${RAW}"
  fi
}
trap finalize EXIT

echo "[COLLECT] floor=${FLOOR} cap=${CAP} output=${OUT_DIR}"
echo "[SAFETY] A=stand, X=walk, B=Passive. Keep vy/wz at zero."
set +e
"${ROBOT_DIR}/run_two_surface_governor.sh" low "$@"
STATUS=$?
set -e
exit "${STATUS}"
