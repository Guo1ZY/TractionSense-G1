# Zorn 足底 ROS ↔ g1_ctrl 足底策略桥接

把 docker **zorn** 的足底 `frame` topic 接到 **unitree_rl_lab** 的 foot 微调策略（510 维 ONNX）。

## 数据流

```text
zorn (docker Isaac)
  ContactView → ROS2
    /g1/left_foot/frame[35]
    /g1/right_foot/frame[35]
        │
        ▼  host: foot_ros_bridge.py
  /tmp/g1_foot_rl_obs.bin   (40B: contact×2, normal×2, tangent×2)
        │
        ▼  g1_ctrl (C++ foot_bridge.h)
  obs: foot_contact / foot_normal_force / foot_tangent_force
        │  (+ history_length=5 in deploy.yaml)
        ▼
  policy.onnx (510 → 29)  @ config/policy/velocity/foot
```

映射（与训练 `mdp/foot_sensor.py` 一致）：

| frame35 | RL 项 | 处理 |
|---------|-------|------|
| `[0]` normal_force_mag | `foot_normal_force` | `×0.01`，clip `[0,5]` |
| `[1]` tangent_force_mag | `foot_tangent_force` | `×0.01`，clip `[0,5]` |
| 由 `[0]` | `foot_contact` | `sigmoid((Fn-5)*2)`，clip `[0,1]` |

**不用** `sensor15` 进当前 ONNX（维数固定 510）。

## 配置

- `config/config.yaml` → `Velocity.policy_dir: config/policy/velocity/foot`
- `config/policy/velocity/foot/exported/policy.onnx`（model_3000 导出）
- `config/policy/velocity/foot/params/deploy.yaml`（含 foot_* 项）

无桥接文件时：足底项为 **0**，策略仍可跑（降级）。

## 启动顺序

### 1. zorn 发 topic

```bash
docker start zorn
docker exec -u root -it zorn bash
# 容器内: g1fs / one-click + Script Editor 起 foot runtime
```

### 2. 宿主机桥接

```bash
<repo>/research_scripts/run_foot_ros_bridge.sh
# 无 zorn 时自测:
<repo>/research_scripts/run_foot_ros_bridge.sh --demo
```

检查：

```bash
python3 -c "import struct,pathlib; d=pathlib.Path('/tmp/g1_foot_rl_obs.bin').read_bytes(); print(len(d), struct.unpack('<IIQffffff', d))"
cat /tmp/g1_foot_rl_obs.bin.json
```

### 3. 编译 g1_ctrl（改了 observations 后必须）

```bash
cd <repo>/deploy/robots/g1_29dof
mkdir -p build && cd build
cmake .. && make -j$(nproc)
```

### 4. MuJoCo + g1_ctrl

```bash
# 终端1: unitree_mujoco
# 终端2:
cd <repo>/deploy/robots/g1_29dof
export G1_FOOT_BRIDGE_PATH=/tmp/g1_foot_rl_obs.bin   # 可选，默认即此路径
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}
./run_g1_ctrl.sh --network lo
# A 站立 → X Velocity
```

## 文件

| 路径 | 作用 |
|------|------|
| `scripts/foot_ros_bridge.py` | ROS→bin 桥 |
| `../../../../scripts/run_foot_ros_bridge.sh` | 一键起桥 |
| `include/.../foot_bridge.h` | C++ 读 bin |
| `include/.../observations.h` | `foot_*` 注册 |
| `config/config.yaml` | `policy_dir: .../foot` |

## 注意

- `ROS_DOMAIN_ID` 与 zorn 一致（默认 0）。
- 包超过 **0.25 s** 未更新 → C++ 当 stale，足底回 0。
- 回退基线策略：config 里改 `policy_dir: config/policy/velocity/v0`。
