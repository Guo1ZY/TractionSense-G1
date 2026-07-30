#!/usr/bin/env bash
# One-click MuJoCo Sim2Sim test via KEYBOARD (a/x/w + friction 1/3).
#
# Usage:
#   POLICY_SLOT=foot TAG=full10900_ground ./research_scripts/one_click_muadapt_mujoco_test.sh
#   POLICY_SLOT=foot_mu TAG=muadapt_ground ./research_scripts/one_click_muadapt_mujoco_test.sh
#
# POLICY_SLOT: foot (Full 10900) | foot_mu (MuAdapt)
# Does NOT talk to a real robot (DDS on lo only).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${TRACTIONSENSE_WORKSPACE:-$(dirname "$ROOT")}"
MUJOCO_ROOT="${UNITREE_MUJOCO_ROOT:-$WORKSPACE/unitree_mujoco}"
export DISPLAY="${DISPLAY:-:1}"
export G1_MUJOCO_FOOT_BRIDGE=1
export G1_FOOT_LOG_DIR="${G1_FOOT_LOG_DIR:-$ROOT/logs/foot_bridge}"

MUJOCO_BIN="$MUJOCO_ROOT/simulate/build/unitree_mujoco"
G1_BIN="$ROOT/deploy/robots/g1_29dof/build/g1_ctrl"
CONFIG_YAML="$ROOT/deploy/robots/g1_29dof/config/config.yaml"
PY_HELPER="$ROOT/research_scripts/_pty_muadapt_driver.py"

STAND_SEC="${STAND_SEC:-6}"
WARM_SEC="${WARM_SEC:-12}"
GRIP_SEC="${GRIP_SEC:-20}"
ICE_SEC="${ICE_SEC:-20}"
POLICY_SLOT="${POLICY_SLOT:-foot_mu}"
TAG="${TAG:-${POLICY_SLOT}_oneclick}"
POLICY_ROOT="$ROOT/deploy/robots/g1_29dof/config/policy/velocity/${POLICY_SLOT}"
DEPLOY_YAML="$POLICY_ROOT/params/deploy.yaml"

[[ -x "$MUJOCO_BIN" ]] || { echo "[ERROR] missing $MUJOCO_BIN"; exit 1; }
[[ -x "$G1_BIN" ]] || { echo "[ERROR] missing $G1_BIN"; exit 1; }
[[ -f "$POLICY_ROOT/exported/policy.onnx" ]] || { echo "[ERROR] missing policy.onnx in $POLICY_SLOT"; exit 1; }
[[ -f "$DEPLOY_YAML" ]] || { echo "[ERROR] missing $DEPLOY_YAML"; exit 1; }
[[ -f "$PY_HELPER" ]] || { echo "[ERROR] missing $PY_HELPER"; exit 1; }

mkdir -p "$G1_FOOT_LOG_DIR"

echo "============================================================"
echo " One-click MuJoCo test (KEYBOARD)"
echo "  policy_slot: $POLICY_SLOT"
echo "  onnx       : $POLICY_ROOT/exported/policy.onnx"
echo "  stand=${STAND_SEC}s warm=${WARM_SEC}s grip=${GRIP_SEC}s ice=${ICE_SEC}s"
echo "  tag=$TAG"
echo "============================================================"

# stop previous sim stack (not train)
pkill -x unitree_mujoco 2>/dev/null || true
pkill -f '/build/g1_ctrl' 2>/dev/null || true
pkill -f 'log_foot_bridge.py' 2>/dev/null || true
sleep 1

# scene + hang band ON (user: 8 lower → A/X → 9 release rope)
CFG_MJ="$MUJOCO_ROOT/simulate/config.yaml"
if [[ -f "$CFG_MJ" ]]; then
  grep -q '^robot_scene:' "$CFG_MJ" && \
    sed -i 's|^robot_scene:.*|robot_scene: "scene_29dof_normal.xml"|' "$CFG_MJ"
  if grep -q '^enable_elastic_band:' "$CFG_MJ"; then
    sed -i 's|^enable_elastic_band:.*|enable_elastic_band: 1|' "$CFG_MJ"
  else
    echo 'enable_elastic_band: 1' >> "$CFG_MJ"
  fi
  echo "[info] elastic_band=1 (8/L lower → A → X → 9 release)"
fi

# policy_dir
python3 - <<PY
from pathlib import Path
p = Path("$CONFIG_YAML")
slot = "$POLICY_SLOT"
lines = []
for line in p.read_text().splitlines():
    if "policy_dir: config/policy/velocity/foot" in line and "mimic" not in line:
        indent = line[: len(line) - len(line.lstrip())]
        lines.append(f"{indent}policy_dir: config/policy/velocity/{slot}")
    else:
        lines.append(line)
p.write_text("\n".join(lines) + "\n")
print(f"[info] config.yaml → velocity/{slot}")
PY

# keyboard cmd for auto w
cp -f "$DEPLOY_YAML" "${DEPLOY_YAML}.bak_oneclick"
python3 - <<PY
from pathlib import Path
p = Path("$DEPLOY_YAML")
t = p.read_text()
if "keyboard_velocity_commands:" not in t:
    t2 = t.replace(
        "  velocity_commands:\n    params: {command_name: base_velocity}",
        "  # one-click auto: keyboard w = forward\n"
        "  keyboard_velocity_commands:\n    params: {command_name: base_velocity}",
    )
    if t2 == t:
        print("[WARN] could not rewrite velocity_commands → keyboard")
    else:
        p.write_text(t2)
        print("[info] deploy.yaml → keyboard_velocity_commands")
else:
    print("[info] deploy already keyboard_velocity_commands")
PY

export STAND_SEC WARM_SEC GRIP_SEC ICE_SEC TAG
set +e
python3 "$PY_HELPER"
RC=$?
set -e

# restore joystick deploy
if [[ -f "${DEPLOY_YAML}.bak_oneclick" ]]; then
  cp -f "${DEPLOY_YAML}.bak_oneclick" "$DEPLOY_YAML"
  echo "[info] restored joystick deploy.yaml"
fi

echo "============================================================"
echo " Finished rc=$RC"
echo "  mujoco: /tmp/oneclick_mujoco.log"
echo "  g1    : /tmp/oneclick_g1.log"
LATEST_CSV=$(ls -t "$G1_FOOT_LOG_DIR"/foot_*"${TAG}"*.csv 2>/dev/null | head -1 || true)
if [[ -n "${LATEST_CSV:-}" ]]; then
  echo "  csv   : $LATEST_CSV"
  python3 - <<PY
import csv, statistics
from collections import defaultdict
from pathlib import Path
path = Path("$LATEST_CSV")
rows = list(csv.DictReader(path.open()))
print("rows", len(rows), "t", rows[0]["elapsed_s"] if rows else "-", "->", rows[-1]["elapsed_s"] if rows else "-")

def ff(r, k):
    try: return float(r[k])
    except Exception: return float("nan")

def mode(r):
    s = (r.get("mu_mode_sim") or "").upper()
    if "GRIP" in s: return "GRIP"
    if "ICE" in s or "SLIP" in s: return "ICE"
    if "NORMAL" in s: return "NORMAL"
    return "DEFAULT"

by = defaultdict(list)
for r in rows:
    if r.get("status") != "OK":
        continue
    by[mode(r)].append(r)

print(f"{'mode':<8} {'n':>5} {'|v|':>7} {'|vx|':>7} {'|vy|':>7} {'Fn':>7} {'Ft':>6} move%")
for m, seg in sorted(by.items()):
    if len(seg) < 5:
        continue
    def col(k):
        return [ff(r, k) for r in seg if ff(r, k) == ff(r, k)]
    vxy = [abs(x) for x in col("vxy")]
    vx = [abs(x) for x in col("vx")]
    vy = [abs(x) for x in col("vy")]
    fn = [0.5 * (ff(r, "Fn_N_L") + ff(r, "Fn_N_R")) for r in seg]
    ft = [0.5 * (ff(r, "Ft_N_L") + ff(r, "Ft_N_R")) for r in seg]
    move = sum(1 for x in vxy if x > 0.15)
    def mmean(a):
        a = [x for x in a if x == x]
        return statistics.mean(a) if a else float("nan")
    print(f"{m:<8} {len(seg):5d} {mmean(vxy):7.3f} {mmean(vx):7.3f} {mmean(vy):7.3f} {mmean(fn):7.1f} {mmean(ft):6.1f} {100*move/max(len(vxy),1):5.1f}%")
# contact check
fn_all = [0.5*(ff(r,"Fn_N_L")+ff(r,"Fn_N_R")) for r in rows if r.get("status")=="OK"]
print(f"Fn>50N frames: {sum(1 for x in fn_all if x>50)}/{len(fn_all)}")
PY
fi
echo "============================================================"
exit "$RC"
