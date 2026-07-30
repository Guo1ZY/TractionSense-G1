#!/usr/bin/env bash
# Multi-terminal manual MuJoCo Sim2Sim launcher.
# Opens 4 separate gnome-terminal windows (more reliable than multi-tab).
#
# Default: Full (原版) foot_full 10900 + 手柄摇杆控速 + 挂带
#
# Usage:
#   ./research_scripts/launch_manual_mujoco_test.sh
#   VEL=1.5 ./research_scripts/launch_manual_mujoco_test.sh
#   POLICY=foot_mu VEL=1.2 ./research_scripts/launch_manual_mujoco_test.sh
#   SCENE=grip VEL=1.5 ./research_scripts/launch_manual_mujoco_test.sh
#
# Env:
#   POLICY   foot | foot_mu          default: foot (Full 原版)
#   SCENE    normal|grip|slip|zones  default: normal
#   VEL      full-stick 目标 m/s     default: 1.2  (G1_CMD_GAIN_LIN)
#   TAG      CSV tag
#   NET      lo
#   KILL     1 先杀旧 sim
#   BAND     1 挂带
#   CMD_MODE joystick | keyboard     default: joystick  ← 手柄控移动
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${TRACTIONSENSE_WORKSPACE:-$(dirname "$ROOT")}"
MUJOCO_ROOT="${UNITREE_MUJOCO_ROOT:-$WORKSPACE/unitree_mujoco}"
export DISPLAY="${DISPLAY:-:1}"
LAUNCHER="$ROOT/research_scripts/launch_manual_mujoco_test.sh"

POLICY="${POLICY:-foot}"
SCENE="${SCENE:-normal}"
VEL="${VEL:-1.2}"
NET="${NET:-lo}"
KILL="${KILL:-1}"
BAND="${BAND:-1}"
CMD_MODE="${CMD_MODE:-joystick}"   # joystick (手柄) | keyboard (仅无手柄时)
TAG="${TAG:-${POLICY}_v${VEL}_manual}"
JS_DEV="${G1_JS_DEVICE:-/dev/input/js0}"

MUJOCO_BUILD="$MUJOCO_ROOT/simulate/build"
MUJOCO_BIN="$MUJOCO_BUILD/unitree_mujoco"
G1_DIR="$ROOT/deploy/robots/g1_29dof"
G1_BIN="$G1_DIR/build/g1_ctrl"
CONFIG_YAML="$G1_DIR/config/config.yaml"
CFG_MJ="$MUJOCO_ROOT/simulate/config.yaml"
POLICY_ROOT="$G1_DIR/config/policy/velocity/${POLICY}"
DEPLOY_YAML="$POLICY_ROOT/params/deploy.yaml"
LOG_DIR="${G1_FOOT_LOG_DIR:-$ROOT/logs/foot_bridge}"
ONNX_RT="$ROOT/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib"
RUN_DIR="/tmp/g1_manual_launch_$$"
mkdir -p "$RUN_DIR"

usage() {
  sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage

need() { [[ -x "$1" ]] || { echo "[ERROR] missing executable: $1"; exit 1; }; }
need "$MUJOCO_BIN"
need "$G1_BIN"
[[ -f "$POLICY_ROOT/exported/policy.onnx" ]] || {
  echo "[ERROR] no policy.onnx under $POLICY_ROOT/exported/"
  exit 1
}
[[ -f "$DEPLOY_YAML" ]] || { echo "[ERROR] missing $DEPLOY_YAML"; exit 1; }
command -v gnome-terminal >/dev/null || {
  echo "[ERROR] need gnome-terminal"
  exit 1
}

case "$SCENE" in
  zones|zone) SCENE_XML="scene_29dof_friction_zones.xml"; SCENE_LABEL="zones" ;;
  slip)       SCENE_XML="scene_29dof_slip.xml";           SCENE_LABEL="SLIP μ≈0.2" ;;
  normal)     SCENE_XML="scene_29dof_normal.xml";         SCENE_LABEL="NORMAL μ≈0.8" ;;
  grip)       SCENE_XML="scene_29dof_grip.xml";           SCENE_LABEL="GRIP μ≈1.5" ;;
  default|orig) SCENE_XML="scene_29dof.xml";              SCENE_LABEL="DEFAULT" ;;
  *) echo "[ERROR] unknown SCENE=$SCENE"; exit 1 ;;
esac

# --- stop old sim (never touch train.py) ---
if [[ "$KILL" == "1" ]]; then
  pkill -x unitree_mujoco 2>/dev/null || true
  pkill -f '/build/g1_ctrl' 2>/dev/null || true
  pkill -f 'log_foot_bridge.py' 2>/dev/null || true
  sleep 0.8
fi

# --- detect gamepad ---
HAS_JS=0
if [[ -e "$JS_DEV" ]]; then
  HAS_JS=1
  echo "[info] gamepad found: $JS_DEV"
else
  # try other js*
  for d in /dev/input/js*; do
    if [[ -e "$d" ]]; then
      JS_DEV="$d"
      HAS_JS=1
      echo "[info] gamepad found: $JS_DEV"
      break
    fi
  done
fi
if [[ "$HAS_JS" != "1" ]]; then
  echo "[WARN] 当前没有 $JS_DEV（系统看不到手柄摇杆设备）"
  echo "       飞智等手柄请切到「游戏手柄 / XInput」模式，不要键盘鼠标模式"
  echo "       插上后应出现: ls /dev/input/js0"
  if [[ "$CMD_MODE" == "joystick" ]]; then
    echo "       仍按手柄模式配置 deploy；没有 js 时摇杆无输入，A/X/B 可用键盘 a/x/b 兜底"
  fi
fi

# --- MuJoCo config ---
if [[ -f "$CFG_MJ" ]]; then
  if grep -q '^robot_scene:' "$CFG_MJ"; then
    sed -i "s|^robot_scene:.*|robot_scene: \"$SCENE_XML\"|" "$CFG_MJ"
  else
    echo "robot_scene: \"$SCENE_XML\"" >> "$CFG_MJ"
  fi
  if grep -q '^enable_elastic_band:' "$CFG_MJ"; then
    sed -i "s|^enable_elastic_band:.*|enable_elastic_band: $BAND|" "$CFG_MJ"
  else
    echo "enable_elastic_band: $BAND" >> "$CFG_MJ"
  fi
  # g1_ctrl 自己读 js；MuJoCo 侧也开（若有 js）便于一致
  if [[ "$HAS_JS" == "1" ]]; then
    sed -i 's|^use_joystick:.*|use_joystick: 1|' "$CFG_MJ"
    if grep -q '^joystick_device:' "$CFG_MJ"; then
      sed -i "s|^joystick_device:.*|joystick_device: \"$JS_DEV\"|" "$CFG_MJ"
    fi
  else
    sed -i 's|^use_joystick:.*|use_joystick: 0|' "$CFG_MJ"
  fi
fi

# --- g1 config: policy_dir ---
python3 - <<PY
from pathlib import Path
p = Path(r"$CONFIG_YAML")
slot = "$POLICY"
out = []
for line in p.read_text().splitlines():
    stripped = line.lstrip()
    if (
        stripped.startswith("policy_dir: config/policy/velocity/")
        and "mimic" not in line
        and not stripped.startswith("#")
    ):
        indent = line[: len(line) - len(line.lstrip())]
        out.append(f"{indent}policy_dir: config/policy/velocity/{slot}")
    else:
        out.append(line)
p.write_text("\n".join(out) + "\n")
print(f"[info] config.yaml → policy/velocity/{slot}")
PY

# --- deploy: joystick velocity_commands by default ---
python3 - <<PY
from pathlib import Path
p = Path(r"$DEPLOY_YAML")
t = p.read_text()
mode = "$CMD_MODE"
if mode == "joystick":
    if "keyboard_velocity_commands:" in t:
        t = t.replace("keyboard_velocity_commands:", "velocity_commands:", 1)
        # clean leftover auto comments
        t = t.replace("# one-click auto: keyboard w = forward\n", "")
        t = t.replace("# manual launcher: keyboard w/s/a/d (hold in g1_ctrl terminal)\n", "")
        p.write_text(t)
        print("[info] deploy.yaml → velocity_commands (手柄摇杆)")
    elif "velocity_commands:" in t:
        print("[info] deploy already velocity_commands (手柄)")
    else:
        print("[WARN] no velocity_commands block")
else:
    if "velocity_commands:" in t and "keyboard_velocity_commands:" not in t:
        t = t.replace(
            "  velocity_commands:",
            "  # CMD_MODE=keyboard\n  keyboard_velocity_commands:",
            1,
        )
        p.write_text(t)
        print("[info] deploy.yaml → keyboard_velocity_commands")
    else:
        print("[info] deploy keyboard mode OK")
PY

# bump lin_vel_x max if VEL > current max (skip comments)
python3 - <<PY
from pathlib import Path
import re
p = Path(r"$DEPLOY_YAML")
lines = p.read_text().splitlines()
vel = float("$VEL")
pat = re.compile(r"^(\s*)lin_vel_x:\s*\[([^,]+),\s*([^\]]+)\]\s*$")
found = False
out = []
for line in lines:
    m = pat.match(line)
    if m and not line.lstrip().startswith("#"):
        found = True
        indent, lo_s, hi_s = m.group(1), m.group(2).strip(), m.group(3).strip()
        lo, hi = float(lo_s), float(hi_s)
        if vel > hi + 1e-6:
            out.append(f"{indent}lin_vel_x: [{lo}, {vel}]")
            print(f"[info] deploy lin_vel_x max {hi} → {vel}")
        else:
            out.append(line)
            print(f"[info] deploy lin_vel_x max={hi} (VEL={vel} OK)")
    else:
        out.append(line)
if not found:
    print("[WARN] lin_vel_x not found")
else:
    p.write_text("\n".join(out) + "\n")
PY

mkdir -p "$LOG_DIR"
HELP_FILE="$RUN_DIR/HELP.txt"
cat > "$HELP_FILE" <<EOF
============================================================
 手动测试  POLICY=$POLICY  VEL(满杆)≈${VEL} m/s  CMD=$CMD_MODE
============================================================
模型: foot=Full原版10900 | foot_mu=MuAdapt14100
当前 ONNX: $POLICY_ROOT/exported/policy.onnx
地面: $SCENE ($SCENE_LABEL)
手柄设备: ${JS_DEV}  (存在=$( [[ -e $JS_DEV ]] && echo YES || echo NO ))

【移动 = 手柄左摇杆】 满杆前进 ≈ ${VEL} m/s（G1_CMD_GAIN_LIN）
【转向 = 手柄右摇杆】 偏航
【状态机】
  手柄: A 站立 → X 进策略 → B 回 Passive
  键盘兜底(点 g1_ctrl 窗口): a / x / b

------------------------------------------------------------
四窗口
  1-MuJoCo   仿真 + 挂带/摩擦
  2-g1_ctrl  策略（手柄输入在这里生效）
  3-logger   CSV
  4-HELP     本说明
  (启动脚本终端会停在操作单，不要慌，正常)

------------------------------------------------------------
操作顺序（挂带）
  1. MuJoCo 终端:  L → 多次 8 放低
  2. 手柄:         A 站立 → 等稳 → X 进 Velocity
     (无手柄则 g1_ctrl 键盘 a → x)
  3. MuJoCo:       再 8 → 9 松带
  4. 手柄左摇杆向前推满 → 测 ${VEL} m/s 是否摔
  5. MuJoCo 摩擦:  1=ICE  2=normal  3=GRIP  v=打速度
  6. 停: 手柄 B 或 g1_ctrl 键 b

------------------------------------------------------------
对比
  原版 1.2:  POLICY=foot VEL=1.2 $LAUNCHER
  原版 1.5:  VEL=1.5 $LAUNCHER
  MuAdapt:   POLICY=foot_mu VEL=1.2 $LAUNCHER

若 ls /dev/input/js0 没有设备:
  手柄切到「游戏/XInput」模式后重插接收器，再重跑本脚本
============================================================
EOF

# --- runner scripts ---
cat > "$RUN_DIR/env.sh" <<EOF
export DISPLAY="${DISPLAY}"
export LD_LIBRARY_PATH="/usr/local/lib:${ONNX_RT}:\${LD_LIBRARY_PATH:-}"
export G1_MUJOCO_FOOT_BRIDGE=1
export G1_FOOT_BRIDGE_PATH=/tmp/g1_foot_rl_obs.bin
export G1_FOOT_LOG_DIR="${LOG_DIR}"
export G1_CMD_GAIN_LIN="${VEL}"
export G1_CMD_GAIN_YAW=1.0
export G1_BAD_ORI_LIMIT=1.8
export G1_JS_DEVICE="${JS_DEV}"
EOF

cat > "$RUN_DIR/01_mujoco.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "$RUN_DIR/env.sh"
cd "$MUJOCO_BUILD"
echo "[MuJoCo] POLICY=$POLICY SCENE=$SCENE"
echo "[MuJoCo] 挂带: L 然后多次 8 放低；松带: 9"
echo "[MuJoCo] 摩擦: 1 ICE | 2 normal | 3 GRIP | v 速度"
exec ./unitree_mujoco
EOF

cat > "$RUN_DIR/02_g1_ctrl.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "$RUN_DIR/env.sh"
cd "$G1_DIR"
echo "[g1_ctrl] POLICY=$POLICY  full-stick ≈ \$G1_CMD_GAIN_LIN m/s"
echo "[g1_ctrl] 手柄: A站立 X走路 B停 | 左摇杆走 右摇杆转"
echo "[g1_ctrl] 键盘兜底: a / x / b"
echo "[g1_ctrl] JS=\$G1_JS_DEVICE"
sleep 1.0
exec ./run_g1_ctrl.sh --network "$NET"
EOF

cat > "$RUN_DIR/03_logger.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "$RUN_DIR/env.sh"
echo "[logger] → \$G1_FOOT_LOG_DIR  tag=$TAG"
echo "[logger] 1/2/3 打μ标签 | note xxx | q"
sleep 1.5
exec python3 "$ROOT/research_scripts/log_foot_bridge.py" --tag "$TAG" --hz 20 --mu-mode "$SCENE"
EOF

cat > "$RUN_DIR/04_help.sh" <<EOF
#!/usr/bin/env bash
cat "$HELP_FILE"
echo
echo "本窗口可关。主启动终端也会显示同样说明并保持打开。"
exec bash
EOF

chmod +x "$RUN_DIR"/*.sh

echo "============================================================"
echo " Launch manual multi-terminal MuJoCo test"
echo "  POLICY  : $POLICY"
echo "  ONNX    : $POLICY_ROOT/exported/policy.onnx"
echo "  SCENE   : $SCENE ($SCENE_LABEL)"
echo "  VEL     : $VEL  (满杆 ≈ ${VEL} m/s)"
echo "  CMD_MODE: $CMD_MODE"
echo "  JS      : $JS_DEV  has=$( [[ $HAS_JS == 1 ]] && echo yes || echo NO )"
echo "  TAG     : $TAG"
echo "  LOG     : $LOG_DIR"
echo "  RUNDIR  : $RUN_DIR"
echo "============================================================"
echo
cat "$HELP_FILE"
echo

# --- open 4 separate windows (reliable; multi-tab quoting was broken) ---
echo "[info] opening 4 gnome-terminal windows ..."
open_win() {
  local title="$1" script="$2"
  # Keep window open on exit so errors are visible
  gnome-terminal --title="$title" -- bash -c "'$script'; echo; echo \"[exit \$?] Enter to close\"; read -r _" &
}

open_win "1-MuJoCo"   "$RUN_DIR/01_mujoco.sh"
sleep 0.4
open_win "2-g1_ctrl"  "$RUN_DIR/02_g1_ctrl.sh"
sleep 0.3
open_win "3-logger"   "$RUN_DIR/03_logger.sh"
sleep 0.2
open_win "4-HELP"     "$RUN_DIR/04_help.sh"

echo
echo "[ok] 4 windows launched (background)."
echo "     本终端保持打开，方便对照操作单。"
echo "     顶速: VEL=1.5 $LAUNCHER"
echo "     MuAdapt: POLICY=foot_mu VEL=1.2 $LAUNCHER"
echo
echo "---- 若手柄不生效，先检查 ----"
echo "  ls -l /dev/input/js0"
echo "  应看到 js0；没有则切手柄到游戏模式后重插"
echo
echo "按 Ctrl-C 结束本说明终端（不会自动杀仿真；要杀请再跑一次脚本或手动关窗口）"
# Keep launcher terminal alive with the ops sheet
while true; do
  echo
  read -r -p "命令: [q]退出说明  [s]看 js0  [r]重打操作单 > " ans || break
  case "${ans:-}" in
    q|Q|quit) break ;;
    s|S)
      ls -la /dev/input/js* 2>&1 || true
      ;;
    r|R)
      cat "$HELP_FILE"
      ;;
    *)
      echo "未知: $ans"
      ;;
  esac
done
