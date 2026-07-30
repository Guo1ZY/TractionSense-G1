#!/usr/bin/env bash
# Launch unitree_mujoco G1 with a chosen floor friction scene for Sim2Sim tests.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${TRACTIONSENSE_WORKSPACE:-$(dirname "$ROOT")}"
MUJOCO_ROOT="${UNITREE_MUJOCO_ROOT:-$WORKSPACE/unitree_mujoco}"
MUJOCO_BUILD="${MUJOCO_BUILD:-$MUJOCO_ROOT/simulate/build}"
CFG="$MUJOCO_ROOT/simulate/config.yaml"
MODE="${1:-normal}"

usage() {
  cat <<EOF
Usage: $0 [zones|slip|normal|grip|default]

  zones   ★ recommended: ONE scene, 3 colored strips (red slip | blue normal | green grip)
          walk -X / +X to change ground; keys 1/2/3 hot-switch ALL floors (need rebuild)

  slip    whole floor low μ (~0.2)  red
  normal  whole floor mid μ (~0.8)  blue
  grip    whole floor high μ (~1.5) green
  default original scene_29dof.xml

Friction switch (RECOMMENDED — type in the unitree_mujoco TERMINAL + Enter):
  1 = ICE/slip   2 = normal   3 = grip   4 = ultra-ice   0 = zones
  f = cycle   v = speed print   h = help
  (3D window keys often stolen by UI focus — prefer terminal input)

Examples:
  $0 zones
  $0 slip
  # other terminal: g1_ctrl --network lo → A stand, X walk
EOF
}

case "$MODE" in
  -h|--help) usage; exit 0 ;;
  zones|zone|all|tri|two|2col) SCENE="scene_29dof_friction_zones.xml"; LABEL="2 COLUMNS left RED slip≈0.2 | right GREEN grip≈1.5" ;;
  slip)   SCENE="scene_29dof_slip.xml";   LABEL="SLIP μ_slide≈0.2" ;;
  normal) SCENE="scene_29dof_normal.xml"; LABEL="NORMAL μ_slide≈0.8" ;;
  grip)   SCENE="scene_29dof_grip.xml";   LABEL="GRIP μ_slide≈1.5" ;;
  default|orig|original) SCENE="scene_29dof.xml"; LABEL="DEFAULT scene (implicit μ)" ;;
  *) echo "Unknown mode: $MODE"; usage; exit 1 ;;
esac

if [[ ! -x "$MUJOCO_BUILD/unitree_mujoco" ]]; then
  echo "[ERROR] unitree_mujoco not found at $MUJOCO_BUILD/unitree_mujoco"
  exit 1
fi

# Patch config.yaml robot_scene (keep other keys)
if [[ -f "$CFG" ]]; then
  if grep -q '^robot_scene:' "$CFG"; then
    sed -i "s|^robot_scene:.*|robot_scene: \"$SCENE\"|" "$CFG"
  else
    echo "robot_scene: \"$SCENE\"" >> "$CFG"
  fi
  # Joystick required for A/X/B FSM via DDS. Prefer enable when js0 exists.
  if [[ -e /dev/input/js0 ]]; then
    sed -i 's|^use_joystick:.*|use_joystick: 1|' "$CFG"
    echo "[INFO] /dev/input/js0 found → use_joystick: 1 (handle A/X/B)"
  else
    sed -i 's|^use_joystick:.*|use_joystick: 0|' "$CFG"
    echo "[WARN] no js0 → use_joystick: 0; use g1_ctrl terminal keys a/x/b"
  fi
fi

echo "============================================================"
echo " MuJoCo friction test"
echo "  mode   : $MODE  ($LABEL)"
echo "  scene  : $SCENE"
echo "  config : $CFG"
echo "============================================================"
if [[ "$SCENE" == *friction_zones* ]]; then
  echo "Layout (top-down, white line = center):"
  echo "   LEFT  RED   y<0  SLIP  μ≈0.2"
  echo "   RIGHT GREEN y>0  GRIP  μ≈1.5"
  echo "   Walk with lateral stick into red vs green; compare speed."
fi
echo "Hotkeys: 1 slip | 2 normal | 3 grip | 0 restore columns | V velocity print"
echo "Velocity: terminal shows |v_xy| m/s (sensor frame_vel). Key V toggles."
echo "Same policy (foot ONNX). Other terminal: g1_ctrl --network lo → A, X"
echo "============================================================"

cd "$MUJOCO_BUILD"
# -r g1 -s scene: some builds take CLI; config.yaml is authoritative for this tree
exec ./unitree_mujoco
