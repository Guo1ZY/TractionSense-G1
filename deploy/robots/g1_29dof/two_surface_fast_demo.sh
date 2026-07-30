#!/usr/bin/env bash
# One-entry workflow for the quickest automatic two-floor/two-speed G1 demo.
set -euo pipefail

ROBOT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${ROBOT_DIR}/../../.." && pwd)"
DATA_ROOT="${G1_FAST_DEMO_DATA_ROOT:-${REPO_ROOT}/logs/real/two_surface_fast_demo}"
CLASSIFIER_DIR="${ROBOT_DIR}/config/traction"
CLASSIFIER="${CLASSIFIER_DIR}/real_classifier.onnx"
METRICS="${CLASSIFIER_DIR}/real_classifier_metrics.json"
PYTHON="${G1_PYTHON:-python3}"
PROFILE="${G1_FAST_DEMO_PROFILE:-clear}"

case "${PROFILE}" in
  safe)
    PROFILE_LOW_SPEED="0.20"
    PROFILE_HIGH_SPEED="0.35"
    ;;
  clear)
    PROFILE_LOW_SPEED="0.20"
    PROFILE_HIGH_SPEED="0.50"
    ;;
  *)
    echo "[ERROR] G1_FAST_DEMO_PROFILE must be safe or clear."
    exit 2
    ;;
esac

LOW_SPEED="${G1_TRACTION_LOW_SPEED:-${PROFILE_LOW_SPEED}}"
HIGH_SPEED="${G1_TRACTION_HIGH_SPEED:-${PROFILE_HIGH_SPEED}}"

usage() {
  cat <<'EOF'
Usage:
  G1_REAL_TEST_ACK=YES ./two_surface_fast_demo.sh collect-low  --network <iface> [--log]
  G1_REAL_TEST_ACK=YES ./two_surface_fast_demo.sh collect-high --network <iface> [--log]
  ./two_surface_fast_demo.sh train
  G1_REAL_TEST_ACK=YES ./two_surface_fast_demo.sh auto --network <iface> [--log]
  ./two_surface_fast_demo.sh status

Optional safe/manual checks:
  G1_REAL_TEST_ACK=YES ./two_surface_fast_demo.sh manual-low  --network <iface>
  G1_REAL_TEST_ACK=YES ./two_surface_fast_demo.sh manual-high --network <iface>

The final AUTO trial uses one joystick command and maps the detected floors to:
  safe  profile: LOW 0.20 m/s, HIGH 0.35 m/s
  clear profile: LOW 0.20 m/s, HIGH 0.50 m/s (default)

Run G1_FAST_DEMO_PROFILE=safe first. Select clear only after both guarded fixed
speed trials pass. G1_TRACTION_LOW_SPEED/HIGH_SPEED provide explicit overrides.
EOF
}

run() {
  if [[ "${G1_FAST_DEMO_DRY_RUN:-0}" == "1" ]]; then
    printf '[DRY-RUN]'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

datasets() {
  local floor="$1"
  if [[ ! -d "${DATA_ROOT}" ]]; then
    return 0
  fi
  find "${DATA_ROOT}" -type f -name "${floor}.npz" -print0 \
    | sort -z
}

dataset_count() {
  local floor="$1"
  local count=0
  while IFS= read -r -d '' _; do
    count=$((count + 1))
  done < <(datasets "${floor}")
  printf '%s' "${count}"
}

show_status() {
  local low_count high_count
  low_count="$(dataset_count low)"
  high_count="$(dataset_count high)"
  echo "=============================================================="
  echo " G1 automatic two-surface fast demo"
  echo " data root   : ${DATA_ROOT}"
  echo " profile     : ${PROFILE}"
  echo " low/high    : ${LOW_SPEED} / ${HIGH_SPEED} m/s"
  echo " LOW trials  : ${low_count}"
  echo " HIGH trials : ${high_count}"
  echo " classifier  : ${CLASSIFIER}"
  if [[ -f "${CLASSIFIER}" ]]; then
    echo " model state : INSTALLED"
  else
    echo " model state : MISSING"
  fi
  if [[ -f "${METRICS}" ]]; then
    "${PYTHON}" - "${METRICS}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
test = report.get("test", {})
print(f" fit result  : {report.get('overall', 'UNKNOWN')}")
print(f" accuracy    : {test.get('balanced_accuracy', float('nan')):.3f}")
print(f" p_low LOW   : {test.get('low_p_mean', float('nan')):.3f}")
print(f" p_low HIGH  : {test.get('high_p_mean', float('nan')):.3f}")
PY
  else
    echo " metrics     : MISSING"
  fi
  echo "=============================================================="
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

COMMAND="$1"
shift

case "${COMMAND}" in
  collect-low|collect-high)
    FLOOR="${COMMAND#collect-}"
    export G1_COLLECTION_DIR="${DATA_ROOT}/$(date +%Y%m%d_%H%M%S)_${FLOOR}"
    export G1_COLLECTION_SPEED="${G1_COLLECTION_SPEED:-0.20}"
    echo "[STEP] Collecting ${FLOOR^^} floor at the common 0.20 m/s cap."
    echo "[RULE] Keep the same forward stick input; vy=wz=0; record >=20 s."
    run "${ROBOT_DIR}/collect_two_surface_proprio.sh" "${FLOOR}" "$@"
    ;;

  train)
    if [[ $# -ne 0 ]]; then
      echo "[ERROR] train takes no positional arguments."
      usage
      exit 2
    fi
    LOW_FILES=()
    HIGH_FILES=()
    while IFS= read -r -d '' path; do
      LOW_FILES+=("${path}")
    done < <(datasets low)
    while IFS= read -r -d '' path; do
      HIGH_FILES+=("${path}")
    done < <(datasets high)
    if [[ ${#LOW_FILES[@]} -lt 1 || ${#HIGH_FILES[@]} -lt 1 ]]; then
      echo "[ERROR] Need at least one LOW and one HIGH collection."
      show_status
      exit 2
    fi
    echo "[STEP] Training on ${#LOW_FILES[@]} LOW + ${#HIGH_FILES[@]} HIGH trials."
    export G1_CLASSIFIER_DEVICE="${G1_CLASSIFIER_DEVICE:-auto}"
    run "${ROBOT_DIR}/train_real_two_surface_classifier.sh" \
      --install "${LOW_FILES[@]}" "${HIGH_FILES[@]}"
    ;;

  auto)
    if [[ ! -f "${CLASSIFIER}" ]]; then
      echo "[ERROR] No installed real-floor classifier. Run collect-low,"
      echo "        collect-high and train first."
      exit 2
    fi
    export G1_GOVERNOR_POLICY=sensorless
    export G1_TRACTION_LOW_SPEED="${LOW_SPEED}"
    export G1_TRACTION_HIGH_SPEED="${HIGH_SPEED}"
    export G1_TRACTION_P_LOW_ENTER="${G1_TRACTION_P_LOW_ENTER:-0.60}"
    export G1_TRACTION_P_HIGH_ENTER="${G1_TRACTION_P_HIGH_ENTER:-0.40}"
    export G1_TRACTION_LOW_HOLD="${G1_TRACTION_LOW_HOLD:-0.20}"
    export G1_TRACTION_HIGH_HOLD="${G1_TRACTION_HIGH_HOLD:-0.80}"
    export G1_TRACTION_MIN_DETECTION_COMMAND="${G1_TRACTION_MIN_DETECTION_COMMAND:-0.18}"
    echo "[STEP] AUTO: UNKNOWN starts at LOW; classifier promotes HIGH."
    run "${ROBOT_DIR}/run_two_surface_governor.sh" auto "$@"
    ;;

  manual-low)
    export G1_GOVERNOR_POLICY=sensorless
    export G1_TRACTION_LOW_SPEED="${LOW_SPEED}"
    run "${ROBOT_DIR}/run_two_surface_governor.sh" low "$@"
    ;;

  manual-high)
    export G1_GOVERNOR_POLICY=sensorless
    export G1_TRACTION_HIGH_SPEED="${HIGH_SPEED}"
    run "${ROBOT_DIR}/run_two_surface_governor.sh" high "$@"
    ;;

  status)
    show_status
    ;;

  -h|--help|help)
    usage
    ;;

  *)
    echo "[ERROR] Unknown command: ${COMMAND}"
    usage
    exit 2
    ;;
esac
