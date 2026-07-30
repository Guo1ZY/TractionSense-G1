#!/usr/bin/env bash
# Train and optionally install a classifier for the exact two physical floors.
set -euo pipefail

ROBOT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${ROBOT_DIR}/../../.." && pwd)"
PYTHON="${G1_PYTHON:-python3}"
TRAINER="${ROOT}/research_scripts/train_proprio_traction_classifier.py"

usage() {
  echo "Usage: $0 [--install] <low.npz> <high.npz> [more labeled .npz ...]"
  echo
  echo "All supplied trials are intentionally used for fitting and an in-sample"
  echo "sanity check. The decisive validation is a new closed-loop floor-switch"
  echo "trial. --install enables AUTO mode on the real robot."
}

INSTALL=0
if [[ "${1:-}" == "--install" ]]; then
  INSTALL=1
  shift
fi
if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

if ! "${PYTHON}" -c "import numpy, torch, onnx" >/dev/null 2>&1; then
  echo "[ERROR] ${PYTHON} does not provide numpy + torch + onnx."
  echo "Select the project training interpreter, for example:"
  echo "  export G1_PYTHON=/path/to/isaaclab/environment/bin/python"
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${G1_CLASSIFIER_OUTPUT_DIR:-${ROOT}/logs/real/traction_classifier/${STAMP}}"
ARGS=()
for path in "$@"; do
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] missing labeled dataset: ${path}"
    exit 2
  fi
  ARGS+=(--train "${path}")
done
for path in "$@"; do
  ARGS+=(--test "${path}")
done

"${PYTHON}" "${TRAINER}" \
  "${ARGS[@]}" \
  --output-dir "${OUT_DIR}" \
  --epochs "${G1_CLASSIFIER_EPOCHS:-80}" \
  --device "${G1_CLASSIFIER_DEVICE:-auto}"

echo "[MODEL] ${OUT_DIR}/traction_classifier.onnx"
echo "[NOTE] metrics are in-sample; run a fresh physical floor-switch test."
if [[ "${INSTALL}" -eq 1 ]]; then
  INSTALL_DIR="${ROBOT_DIR}/config/traction"
  mkdir -p "${INSTALL_DIR}"
  cp "${OUT_DIR}/traction_classifier.onnx" \
    "${INSTALL_DIR}/real_classifier.onnx"
  cp "${OUT_DIR}/metrics.json" "${INSTALL_DIR}/real_classifier_metrics.json"
  echo "[INSTALLED] ${INSTALL_DIR}/real_classifier.onnx"
else
  echo "[SAFE] not installed. Re-run with --install after reviewing metrics."
fi
