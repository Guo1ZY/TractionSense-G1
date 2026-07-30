#!/usr/bin/env bash
# Full model MuJoCo multi-friction test (hang workflow + file cmds).
# Requires rebuilt unitree_mujoco with /tmp/mujoco_rl_cmd polling.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${TRACTIONSENSE_WORKSPACE:-$(dirname "$ROOT")}"
MUJOCO_ROOT="${UNITREE_MUJOCO_ROOT:-$WORKSPACE/unitree_mujoco}"
export TRACTIONSENSE_ROOT="$ROOT"
export DISPLAY="${DISPLAY:-:1}"
export G1_MUJOCO_FOOT_BRIDGE=1
export G1_BAD_ORI_LIMIT="${G1_BAD_ORI_LIMIT:-1.8}"
export G1_FOOT_LOG_DIR="${G1_FOOT_LOG_DIR:-$ROOT/logs/foot_bridge}"
export LD_LIBRARY_PATH="/usr/local/lib:$ROOT/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib:${LD_LIBRARY_PATH:-}"

POLICY_SLOT="${POLICY_SLOT:-foot}"
TAG="${TAG:-${POLICY_SLOT}_friction_ab}"
STAND_SEC="${STAND_SEC:-8}"
SETTLE_SEC="${SETTLE_SEC:-4}"
WARM_SEC="${WARM_SEC:-12}"
GRIP_SEC="${GRIP_SEC:-18}"
ICE_SEC="${ICE_SEC:-18}"

MUJOCO="$MUJOCO_ROOT/simulate/build/unitree_mujoco"
G1="$ROOT/deploy/robots/g1_29dof/build/g1_ctrl"
CFG="$MUJOCO_ROOT/simulate/config.yaml"
CFG_YAML="$ROOT/deploy/robots/g1_29dof/config/config.yaml"
DEP="$ROOT/deploy/robots/g1_29dof/config/policy/velocity/${POLICY_SLOT}/params/deploy.yaml"
CMD=/tmp/mujoco_rl_cmd

echo "==== MuJoCo friction test slot=$POLICY_SLOT tag=$TAG ===="

# stop old sim by exact name
for p in $(pgrep -x unitree_mujoco || true); do kill "$p" 2>/dev/null || true; done
for p in $(pgrep -f 'g1_29dof/build/g1_ctrl' || true); do kill "$p" 2>/dev/null || true; done
for p in $(pgrep -f 'log_foot_bridge.py' || true); do kill "$p" 2>/dev/null || true; done
sleep 1

# config
sed -i 's|^robot_scene:.*|robot_scene: "scene_29dof_normal.xml"|' "$CFG" || true
sed -i 's|^enable_elastic_band:.*|enable_elastic_band: 1|' "$CFG" || true
sed -i 's|^use_joystick:.*|use_joystick: 0|' "$CFG" || true
python3 -c "
from pathlib import Path
p=Path('$CFG_YAML'); slot='$POLICY_SLOT'; o=[]
for line in p.read_text().splitlines():
  if 'policy_dir: config/policy/velocity/foot' in line and 'mimic' not in line:
    ind=line[:len(line)-len(line.lstrip())]
    o.append(f'{ind}policy_dir: config/policy/velocity/{slot}')
  else: o.append(line)
p.write_text(chr(10).join(o)+chr(10))
print('policy', slot)
"
cp -f "$DEP" "${DEP}.bak_fric"
python3 -c "
from pathlib import Path
p=Path('$DEP'); t=p.read_text()
if 'keyboard_velocity_commands:' not in t:
  t=t.replace('  velocity_commands:\n    params: {command_name: base_velocity}',
              '  keyboard_velocity_commands:\n    params: {command_name: base_velocity}')
  p.write_text(t); print('keyboard on')
"

# start mujoco (normal, not fifo)
cd "$MUJOCO_ROOT/simulate/build"
./unitree_mujoco > /tmp/full_fric_mujoco.log 2>&1 &
echo $! > /tmp/full_fric_mj.pid
echo "mujoco $(cat /tmp/full_fric_mj.pid)"
sleep 5

mj(){ printf '%s\n' "$1" > "$CMD"; echo "mujoco <- $1"; sleep 0.6; }

# start g1 pty feeder
python3 - <<'PY' &
import os, pty, select, subprocess, time, signal
from pathlib import Path
env=os.environ.copy()
repo=Path(os.environ["TRACTIONSENSE_ROOT"])
ort=str(repo / "deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib")
env["LD_LIBRARY_PATH"]=f"/usr/local/lib:{ort}:{env.get('LD_LIBRARY_PATH','')}"
env["G1_BAD_ORI_LIMIT"]=os.environ.get("G1_BAD_ORI_LIMIT","1.8")
env["G1_CMD_GAIN_LIN"]="1.0"
master,slave=pty.openpty()
logf=open("/tmp/full_fric_g1.log","w",buffering=1)
proc=subprocess.Popen(
  [str(repo / "deploy/robots/g1_29dof/build/g1_ctrl"),"--network","lo"],
  cwd=str(repo / "deploy/robots/g1_29dof/build"),
  stdin=slave,stdout=slave,stderr=slave,env=env,preexec_fn=os.setsid)
os.close(slave)
cmdf=Path("/tmp/full_fric_g1_cmd"); cmdf.write_text("")
def drain(t=0.1):
  end=time.time()+t
  while time.time()<end:
    r,_,_=select.select([master],[],[],0.05)
    if master in r:
      try:
        c=os.read(master,8192); logf.write(c.decode("utf-8","replace"))
      except Exception: break
time.sleep(4); drain(0.5)
print("g1 ready", flush=True)
while True:
  drain(0.05)
  raw=cmdf.read_text().strip() if cmdf.exists() else ""
  if not raw:
    if proc.poll() is not None: break
    time.sleep(0.05); continue
  cmdf.write_text("")
  if raw=="QUIT": break
  if raw=="W": os.write(master,b"w")
  else:
    os.write(master, raw[:1].encode()); print("g1 key", raw[:1], flush=True)
  drain(0.12)
try: os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
except Exception: proc.terminate()
logf.close()
PY
echo $! > /tmp/full_fric_g1h.pid
sleep 5

g1(){ echo "$1" > /tmp/full_fric_g1_cmd; echo "g1 <- $1"; sleep 0.4; }

TOTAL=$((STAND_SEC+SETTLE_SEC+WARM_SEC+GRIP_SEC+ICE_SEC+15))
python3 "$ROOT/research_scripts/log_foot_bridge.py" --tag "$TAG" --hz 20 --duration "$TOTAL" --no-console \
  > /tmp/full_fric_logger.out 2>&1 &
echo $! > /tmp/full_fric_log.pid
echo "logger $(cat /tmp/full_fric_log.pid) ${TOTAL}s"

echo "== L lower =="
mj L
sleep 2
echo "== A stand ${STAND_SEC}s =="
g1 a
sleep "$STAND_SEC"
sleep "$SETTLE_SEC"
echo "== X =="
g1 x
sleep 1.5
if grep -q 'from Velocity to Passive' /tmp/full_fric_g1.log 2>/dev/null; then
  echo "WARN Velocity->Passive, retry A/X"
  g1 a; sleep 6; g1 x; sleep 1
fi
echo "== 9 release =="
mj 9
sleep 2

walk(){ local n=$(( $1 * 7 )); echo "walk $2 ${1}s"; for ((i=0;i<n;i++)); do echo W > /tmp/full_fric_g1_cmd; sleep 0.14; done; }

walk "$WARM_SEC" warm
echo "== GRIP 3 =="; mj 3; walk "$GRIP_SEC" GRIP
echo "== ICE 1 =="; mj 1; walk "$ICE_SEC" ICE

echo "== stop =="
echo QUIT > /tmp/full_fric_g1_cmd || true
sleep 1
kill "$(cat /tmp/full_fric_log.pid)" 2>/dev/null || true
kill "$(cat /tmp/full_fric_mj.pid)" 2>/dev/null || true
kill "$(cat /tmp/full_fric_g1h.pid)" 2>/dev/null || true
sleep 1
[[ -f "${DEP}.bak_fric" ]] && cp -f "${DEP}.bak_fric" "$DEP" || true

CSV=$(ls -t "$G1_FOOT_LOG_DIR"/foot_*"${TAG}"*.csv 2>/dev/null | head -1 || true)
echo "CSV=$CSV"
python3 - <<PY
import csv, statistics
from collections import defaultdict, Counter
from pathlib import Path
p=Path("$CSV") if "$CSV" else None
if not p or not p.exists():
  print("no csv"); raise SystemExit
rows=list(csv.DictReader(p.open()))
print("rows",len(rows),"status",Counter(r["status"] for r in rows))
def ff(r,k):
  try: return float(r[k])
  except: return float("nan")
def mode(r):
  s=(r.get("mu_mode_sim") or "").upper()
  if "GRIP" in s: return "GRIP"
  if "ICE" in s or "SLIP" in s: return "ICE"
  return "DEFAULT"
print("mu", Counter(mode(r) for r in rows))
by=defaultdict(list)
for r in rows:
  if r["status"]=="OK": by[mode(r)].append(r)
print(f"{'mode':<8} n |v| |vx| |vy| Fn Ft move%")
for m,seg in sorted(by.items()):
  if len(seg)<5: continue
  vxy=[abs(ff(r,"vxy")) for r in seg if ff(r,"vxy")==ff(r,"vxy")]
  vx=[abs(ff(r,"vx")) for r in seg if ff(r,"vx")==ff(r,"vx")]
  vy=[abs(ff(r,"vy")) for r in seg if ff(r,"vy")==ff(r,"vy")]
  fn=[0.5*(ff(r,"Fn_N_L")+ff(r,"Fn_N_R")) for r in seg]
  ft=[0.5*(ff(r,"Ft_N_L")+ff(r,"Ft_N_R")) for r in seg]
  def mm(a):
    a=[x for x in a if x==x]; return statistics.mean(a) if a else float("nan")
  move=sum(1 for x in vxy if x>0.15)
  print(f"{m:<8} {len(seg):4d} {mm(vxy):.3f} {mm(vx):.3f} {mm(vy):.3f} {mm(fn):.1f} {mm(ft):.1f} {100*move/max(len(vxy),1):.0f}%")
fn=[0.5*(ff(r,"Fn_N_L")+ff(r,"Fn_N_R")) for r in rows if r["status"]=="OK"]
print(f"Fn>50 {sum(1 for x in fn if x==x and x>50)}/{len(fn)} max={max((x for x in fn if x==x), default=0):.1f}")
print("--- FSM ---")
for line in Path("/tmp/full_fric_g1.log").read_text(errors="replace").splitlines():
  if "FSM" in line or "G1_BAD" in line: print(line)
print("--- band/fric ---")
for line in Path("/tmp/full_fric_mujoco.log").read_text(errors="replace").splitlines():
  if "[band" in line or "KEY OK" in line: print(line)
PY
ARCH="$ROOT/logs/full_fric_result_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCH"
[[ -n "${CSV:-}" ]] && cp -a "$CSV" "$ARCH/" || true
cp -a /tmp/full_fric_mujoco.log /tmp/full_fric_g1.log "$ARCH/" 2>/dev/null || true
echo "archive $ARCH"
echo DONE
