#!/usr/bin/env bash
# Run g1_ctrl with correct library paths (unitree_sdk2 / cyclonedds in /usr/local/lib)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="/usr/local/lib:${ROOT}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib:${LD_LIBRARY_PATH:-}"
cd "${ROOT}/build"
exec ./g1_ctrl "$@"
