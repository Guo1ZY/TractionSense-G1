#!/usr/bin/env bash
set -euo pipefail

network=""
execute=0

usage() {
  echo "Usage: $0 --network INTERFACE [--execute]"
  echo "Without --execute this runs read-only preflight checks only."
}

while (($#)); do
  case "$1" in
    --network)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      network="$2"
      shift 2
      ;;
    --execute)
      execute=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

[[ -n "$network" ]] || { echo "--network is required" >&2; exit 2; }
[[ -d "/sys/class/net/$network" ]] || {
  echo "Network interface does not exist: $network" >&2
  exit 2
}
[[ "$network" != "lo" ]] || {
  echo "Refusing loopback for a real-robot launch." >&2
  exit 2
}

repo_root="${TRACTIONSENSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${ISAACLAB_PYTHON:-python3}"
preflight="$repo_root/research_scripts/check_real_magnetic_preflight.py"
controller="$repo_root/deploy/robots/g1_29dof/build/g1_ctrl"

"$python_bin" "$preflight" --require-policy-active

if ((execute == 0)); then
  echo "Read-only preflight passed. Controller was not started."
  echo "Use --execute only with the robot in a harness and hardware E-stop held."
  exit 0
fi

[[ -x "$controller" ]] || {
  echo "Controller binary is missing or not executable: $controller" >&2
  exit 2
}

echo "Starting G1 controller on $network."
echo "Keep hardware E-stop ready: A=FixStand, X=Velocity, B=Passive."
exec "$controller" --network "$network"
