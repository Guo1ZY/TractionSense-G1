#!/usr/bin/env bash
# Fine-tune G1 velocity policy with foot-sensor obs + friction DR from model_49999.
# Also supports strict resume after a NaN crash.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB_DIR="${UNITREE_RL_LAB_DIR:-$ROOT}"
ISAACLAB_PATH="${ISAACLAB_PATH:-${HOME}/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-isaaclab-v2}"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-10000}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-foot_ft}"
HEADLESS="${HEADLESS:-1}"
TASK="${TASK:-Unitree-G1-29dof-Velocity-Foot}"
PARTIAL_CKPT="${PARTIAL_CKPT:-$LAB_DIR/model/rl/model_49999.pt}"
RESUME_CKPT="${RESUME_CKPT:-}"

# Recommended safe resume after the ~910 crash (value-loss NaN started ~700+)
DEFAULT_SAFE_RESUME="$LAB_DIR/logs/rsl_rl/unitree_g1_29dof_velocity_foot/2026-07-14_11-34-53_foot_ft/model_700.pt"

# In-place / right-stick turn fine-tune (vx/vy unchanged vs model_4000)
DEFAULT_TURN_RESUME="$LAB_DIR/model/rl/model_foot_4000.pt"
# Combined adaptive (speed+turn+friction): partial warm-start from foot_4000
DEFAULT_ADAPTIVE_PARTIAL="$LAB_DIR/model/rl/model_foot_4000.pt"

usage() {
  cat <<EOF
Usage: $0 [options]

Fine-tune from model_49999 (partial) or resume a foot checkpoint (strict).

  $0 --smoke
  NUM_ENVS=2048 MAX_ITERS=5000 $0
  $0 --resume-safe                 # model_700 from crashed foot_ft run
  $0 --resume-checkpoint /path/to/model_700.pt

  # Right-stick in-place turn only (vx/vy same as 4000):
  $0 --turn
  $0 --turn --max-iterations 6000 --run-name foot_turn

  # RECOMMENDED: fast walk + turn + low-μ slow-stable (partial from 4000):
  $0 --adaptive
  $0 --adaptive --max-iterations 8000 --run-name foot_adaptive

  # After Adaptive NaN: resume model_5400 with forced wider yaw + relaxed ang curriculum:
  $0 --adaptive-yaw
  $0 --adaptive-yaw --resume-checkpoint /path/to/model_5400.pt

  # Fix idle stomping + low-μ slow-down (from adaptive_yaw model_6600):
  $0 --adaptive-stable

  # RECOMMENDED clean rebuild from model_49999 (discard 6600 stack):
  #   foot + μ adapt + turn + stand-still + speed curriculum in one run
  $0 --from-base
  $0 --from-base --max-iterations 12000 --run-name foot_full

  # Adaptive-V2: fix μ-invariant mid-speed (outcome rewards, no actor Ft)
  $0 --v2 --smoke
  $0 --v2 --max-iterations 15000 --run-name foot_adaptive_v2

  # RECOMMENDED MuAdapt: 510 Fn+Ft + full track + stable_speed + lateral skid
  $0 --mu-adapt --smoke
  $0 --mu-adapt --max-iterations 12000 --run-name foot_mu_adapt

  # ★ RECOMMENDED clean line (2026-07-17): high-μ fast straight / low-μ slow-stable
  # Always partial from model_49999; NO turn / NO spin / narrow yaw
  $0 --straight-mu --smoke
  $0 --straight-mu --max-iterations 12000 --run-name foot_straight_mu

  # Traction-Adaptive: default vx<=1.0 + 15% high-speed stress probes;
  # privileged μ teacher, 0.3-s deployable foot-force context
  $0 --traction-adaptive --smoke
  $0 --traction-adaptive --max-iterations 16000 --run-name foot_traction_adaptive

Env: ISAACLAB_PATH, CONDA_ENV, NUM_ENVS, MAX_ITERS, DEVICE, SEED, RUN_NAME,
     PARTIAL_CKPT, RESUME_CKPT, TASK, HEADLESS
EOF
}

SMOKE=0
EXTRA=()
USE_PARTIAL_FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --smoke) SMOKE=1; shift ;;
    --turn)
      # Pure-yaw sampling + wz→±0.6; resume from model_foot_4000
      TASK="Unitree-G1-29dof-Velocity-Foot-Turn"
      if [[ -z "$RESUME_CKPT" ]]; then
        RESUME_CKPT="$DEFAULT_TURN_RESUME"
      fi
      if [[ "$RUN_NAME" == "foot_ft" ]]; then
        RUN_NAME="foot_turn"
      fi
      if [[ "$MAX_ITERS" == "10000" ]]; then
        MAX_ITERS=6000
      fi
      shift
      ;;
    --adaptive)
      # Speed + turn + friction adaptive; critic obs grows → partial load
      TASK="Unitree-G1-29dof-Velocity-Foot-Adaptive"
      RESUME_CKPT=""
      USE_PARTIAL_FORCE=1
      PARTIAL_CKPT="$DEFAULT_ADAPTIVE_PARTIAL"
      if [[ "$RUN_NAME" == "foot_ft" ]]; then
        RUN_NAME="foot_adaptive"
      fi
      if [[ "$MAX_ITERS" == "10000" ]]; then
        MAX_ITERS=8000
      fi
      shift
      ;;
    --adaptive-yaw)
      # Resume NaN-safe Adaptive with wider yaw start + relaxed ang curriculum
      TASK="Unitree-G1-29dof-Velocity-Foot-Adaptive-Yaw"
      DEFAULT_YAW_RESUME="$LAB_DIR/logs/rsl_rl/unitree_g1_29dof_velocity_foot_adaptive/2026-07-14_17-35-06_foot_adaptive/model_5400.pt"
      if [[ -z "$RESUME_CKPT" ]]; then
        RESUME_CKPT="$DEFAULT_YAW_RESUME"
      fi
      if [[ "$RUN_NAME" == "foot_ft" ]]; then
        RUN_NAME="foot_adaptive_yaw"
      fi
      if [[ "$MAX_ITERS" == "10000" ]]; then
        MAX_ITERS=9000
      fi
      shift
      ;;
    --adaptive-stable)
      # Idle stand still + stronger low-μ slow-down; resume yaw model_6600
      TASK="Unitree-G1-29dof-Velocity-Foot-Adaptive-Stable"
      DEFAULT_STABLE_RESUME="$LAB_DIR/logs/rsl_rl/unitree_g1_29dof_velocity_foot_adaptive_yaw/2026-07-14_21-34-03_foot_adaptive_yaw/model_6600.pt"
      if [[ -z "$RESUME_CKPT" ]]; then
        RESUME_CKPT="$DEFAULT_STABLE_RESUME"
      fi
      if [[ "$RUN_NAME" == "foot_ft" ]]; then
        RUN_NAME="foot_adaptive_stable"
      fi
      if [[ "$MAX_ITERS" == "10000" ]]; then
        MAX_ITERS=5000
      fi
      shift
      ;;
    --from-base|--full)
      # Clean multi-objective from model_49999 — do NOT use 6600 stack
      TASK="Unitree-G1-29dof-Velocity-Foot-Full"
      RESUME_CKPT=""
      USE_PARTIAL_FORCE=1
      PARTIAL_CKPT="${PARTIAL_CKPT:-$LAB_DIR/model/rl/model_49999.pt}"
      if [[ "$RUN_NAME" == "foot_ft" ]]; then
        RUN_NAME="foot_full"
      fi
      if [[ "$MAX_ITERS" == "10000" ]]; then
        MAX_ITERS=12000
      fi
      shift
      ;;
    --v2|--adaptive-v2)
      # Outcome rewards + deployable actor (no tangent); partial from 49999
      TASK="Unitree-G1-29dof-Velocity-Foot-Adaptive-V2"
      RESUME_CKPT=""
      USE_PARTIAL_FORCE=1
      PARTIAL_CKPT="${PARTIAL_CKPT:-$LAB_DIR/model/rl/model_49999.pt}"
      if [[ "$RUN_NAME" == "foot_ft" ]]; then
        RUN_NAME="foot_adaptive_v2"
      fi
      if [[ "$MAX_ITERS" == "10000" ]]; then
        MAX_ITERS=15000
      fi
      shift
      ;;
    --mu-adapt|--mu)
      # RECOMMENDED: 510-dim Fn+Ft (same as Foot-Full) + outcome rewards + lateral skid
      # Warm-start Foot-Full / model_foot_4000 if same 510 actor (prefer resume).
      TASK="Unitree-G1-29dof-Velocity-Foot-MuAdapt"
      if [[ -z "${RESUME_CKPT:-}" ]]; then
        # Prefer latest healthy MuAdapt ckpt, else Full 10900
        MU_LATEST=""
        if compgen -G "$LAB_DIR/logs/rsl_rl/unitree_g1_29dof_velocity_foot_mu_adapt/*/model_*.pt" > /dev/null; then
          MU_LATEST=$(ls -t "$LAB_DIR"/logs/rsl_rl/unitree_g1_29dof_velocity_foot_mu_adapt/*/model_*.pt 2>/dev/null | head -1 || true)
        fi
        FULL_CKPT="$LAB_DIR/logs/rsl_rl/unitree_g1_29dof_velocity_foot_full/2026-07-15_10-40-20_foot_full/model_10900.pt"
        if [[ -n "$MU_LATEST" && -f "$MU_LATEST" ]]; then
          RESUME_CKPT="$MU_LATEST"
          echo "[info] MuAdapt continue from latest: $RESUME_CKPT"
        elif [[ -f "$FULL_CKPT" ]]; then
          RESUME_CKPT="$FULL_CKPT"
        elif [[ -f "$LAB_DIR/model/rl/model_foot_4000.pt" ]]; then
          # foot_4000 critic may lack ρ/slip → use partial expand
          RESUME_CKPT=""
          USE_PARTIAL_FORCE=1
          PARTIAL_CKPT="$LAB_DIR/model/rl/model_foot_4000.pt"
        else
          USE_PARTIAL_FORCE=1
          PARTIAL_CKPT="${PARTIAL_CKPT:-$LAB_DIR/model/rl/model_49999.pt}"
        fi
      fi
      if [[ "$RUN_NAME" == "foot_ft" ]]; then
        RUN_NAME="foot_mu_adapt"
      fi
      if [[ "$MAX_ITERS" == "10000" ]]; then
        MAX_ITERS=12000
      fi
      shift
      ;;
    --straight-mu|--straight)
      # ★ Clean line: high-μ fast straight / low-μ slow-stable from 49999 only
      TASK="Unitree-G1-29dof-Velocity-Foot-StraightMu"
      RESUME_CKPT=""
      USE_PARTIAL_FORCE=1
      PARTIAL_CKPT="${PARTIAL_CKPT:-$LAB_DIR/model/rl/model_49999.pt}"
      if [[ "$RUN_NAME" == "foot_ft" ]]; then
        RUN_NAME="foot_straight_mu"
      fi
      if [[ "$MAX_ITERS" == "10000" ]]; then
        MAX_ITERS=12000
      fi
      shift
      ;;
    --traction-adaptive|--traction)
      TASK="Unitree-G1-29dof-Velocity-Foot-TractionAdaptive"
      RESUME_CKPT=""
      USE_PARTIAL_FORCE=1
      PARTIAL_CKPT="${PARTIAL_CKPT:-$LAB_DIR/model/rl/model_49999.pt}"
      if [[ "$RUN_NAME" == "foot_ft" ]]; then
        RUN_NAME="foot_traction_adaptive"
      fi
      if [[ "$MAX_ITERS" == "10000" ]]; then
        MAX_ITERS=16000
      fi
      shift
      ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --max-iterations) MAX_ITERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --partial-checkpoint) PARTIAL_CKPT="$2"; USE_PARTIAL_FORCE=1; RESUME_CKPT=""; shift 2 ;;
    --resume-checkpoint) RESUME_CKPT="$2"; shift 2 ;;
    --resume-safe)
      RESUME_CKPT="$DEFAULT_SAFE_RESUME"
      RUN_NAME="${RUN_NAME:-foot_ft_resume}"
      shift
      ;;
    --task) TASK="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ "$SMOKE" == "1" ]]; then
  NUM_ENVS=256
  MAX_ITERS=20
  RUN_NAME="foot_smoke"
fi

if [[ -n "$RESUME_CKPT" ]]; then
  if [[ ! -f "$RESUME_CKPT" ]]; then
    echo "[ERROR] resume checkpoint not found: $RESUME_CKPT"
    exit 1
  fi
  RESUME_CKPT=$(realpath "$RESUME_CKPT")
elif [[ ! -f "$PARTIAL_CKPT" ]]; then
  echo "[ERROR] partial checkpoint not found: $PARTIAL_CKPT"
  exit 1
fi

source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export ISAACLAB_PATH
unset PYTHONPATH || true
if [[ -f "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh" || true
  set -u
fi

cd "$LAB_DIR"
mkdir -p logs/rsl_rl

echo "============================================================"
echo " G1 Foot-Sensor Fine-tune"
echo "  task            : $TASK"
if [[ -n "$RESUME_CKPT" ]]; then
  echo "  mode            : RESUME strict (no optimizer by default)"
  echo "  resume_ckpt     : $RESUME_CKPT"
else
  echo "  mode            : PARTIAL warm-start from 49999"
  echo "  partial_ckpt    : $PARTIAL_CKPT"
fi
echo "  num_envs        : $NUM_ENVS"
echo "  max_iters       : $MAX_ITERS"
echo "  device          : $DEVICE"
echo "  run_name        : $RUN_NAME"
echo "============================================================"

CMD=(
  python scripts/rsl_rl/train.py
  --task "$TASK"
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERS"
  --device "$DEVICE"
  --seed "$SEED"
  --run_name "$RUN_NAME"
)
if [[ -n "$RESUME_CKPT" ]]; then
  CMD+=(--resume_checkpoint "$RESUME_CKPT")
else
  CMD+=(--partial_checkpoint "$PARTIAL_CKPT")
fi
if [[ "$HEADLESS" == "1" ]]; then
  CMD+=(--headless)
fi
if ((${#EXTRA[@]})); then
  CMD+=("${EXTRA[@]}")
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"
