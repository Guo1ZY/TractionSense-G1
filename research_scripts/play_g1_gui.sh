#!/usr/bin/env bash
# GUI play for G1 velocity policy (opens Isaac window)
# Fix "Maximum number of clients reached" first if needed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB_DIR="${UNITREE_RL_LAB_DIR:-$ROOT}"
ISAACLAB_PATH="${ISAACLAB_PATH:-${HOME}/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-isaaclab-v2}"
NUM_ENVS="${NUM_ENVS:-1}"
DEVICE="${DEVICE:-cuda:0}"
VIDEO="${VIDEO:-0}"
VIDEO_LENGTH="${VIDEO_LENGTH:-300}"

CHECKPOINT="${1:-$ROOT/checkpoints/baseline_model_49999.pt}"

export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"

echo "=============================================="
echo " G1 GUI Play"
echo "  DISPLAY    = $DISPLAY"
echo "  XAUTHORITY = $XAUTHORITY"
echo "  checkpoint = $CHECKPOINT"
echo "=============================================="

# 1) basic display check
if ! xset q >/dev/null 2>&1; then
  echo "[ERROR] 打不开 DISPLAY=$DISPLAY"
  echo "  请在图形桌面终端里跑，或："
  echo "    export DISPLAY=:1"
  echo "    export XAUTHORITY=/run/user/\$(id -u)/gdm/Xauthority"
  echo "    xhost +local:"
  exit 1
fi

# 2) X client count
N_CLIENTS="$(xlsclients 2>/dev/null | wc -l || echo 999)"
echo "[INFO] 当前 X clients ≈ $N_CLIENTS"
if ! xlsclients >/dev/null 2>&1; then
  echo "[ERROR] Maximum number of clients reached — 必须先释放显示连接"
  echo ""
  echo "请按顺序做："
  echo "  1) 关掉 Chrome / Edge / QQ / 微信 / 向日葵 多余窗口"
  echo "  2) 关掉之前的 Isaac / unitree_mujoco 窗口"
  echo "  3) 或注销/重登桌面（最干净）"
  echo "  4) 再跑本脚本"
  echo ""
  echo "也可先 headless 录视频（不弹窗）："
  echo "  bash $ROOT/research_scripts/play_g1_velocity.sh $CHECKPOINT"
  exit 1
fi

if [[ "$N_CLIENTS" -gt 40 ]]; then
  echo "[WARN] X clients 偏多 ($N_CLIENTS)，Isaac 可能仍会失败，建议关掉浏览器再试"
fi

# 3) allow local connections
xhost +local: >/dev/null 2>&1 || true

# 4) kill leftover play (not browsers)
pkill -f "scripts/rsl_rl/play.py" 2>/dev/null || true
sleep 1

source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export ISAACLAB_PATH
set +u
source "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh" 2>/dev/null || true
set -u

cd "$LAB_DIR"

ARGS=(
  --task Unitree-G1-29dof-Velocity
  --num_envs "$NUM_ENVS"
  --checkpoint "$CHECKPOINT"
  --device "$DEVICE"
)
# GUI play: no --headless
# video optional (GUI 时也可录，但更吃资源)
if [[ "$VIDEO" == "1" ]]; then
  ARGS+=(--video --video_length "$VIDEO_LENGTH")
fi

echo "+ python scripts/rsl_rl/play.py ${ARGS[*]}"
echo "[INFO] 应弹出 Isaac 窗口；关掉窗口即结束。"
exec python scripts/rsl_rl/play.py "${ARGS[@]}"
