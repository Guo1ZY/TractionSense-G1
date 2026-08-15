# G1 足底传感器 → 仿真训练链路

将 zorn `foot_sensor` 语义接到 `unitree_rl_lab` 速度行走任务，支持：

1. 接口对齐（Contact / 力 / L-R / 单位 / 频率 / topic）
2. 可开关 Observation（默认关 = 49999 兼容）
3. 从 `model_49999.pt` 微调（足底 obs + 传感器噪声 DR）
4. 摩擦自适应（随机 μ + soft anti-slip，无 if-μ 规则）
5. 导出 ONNX → MuJoCo / g1_ctrl 复测

---

## 1. 对齐 zorn foot_sensor

| 项 | 仿真训练 (Isaac Lab) | zorn 采集 / ROS2 |
|----|----------------------|------------------|
| 左右脚 body | `left_ankle_roll_link`, `right_ankle_roll_link` | 同 prim 路径 |
| 接触/力源 | `ContactSensor` → `net_forces_w` | RigidPrim ContactView + ground filter |
| 15 点 | 不进入 RL 主环（并行成本高） | `sensor15` RBF 分布，schema 对齐 |
| 单位 | N, m, s | 同 |
| 物理 dt | `0.005` s | ContactView `DT=0.005` |
| 策略步 | `0.02` s (`decimation=4`) | collector ~20 Hz / ROS ~50 Hz |
| Topic | （训练无 ROS） | `/g1/{left,right}_foot/frame` [35]<br>`/g1/{left,right}_foot/sensor15` [15] |

**frame35 布局**（zorn `build_frame35`）：

```
[0]     normal_force_mag
[1]     tangent_force_mag
[2]     total_force_mag
[3:6]   cop_local
[6:9]   force_local_total
[9:12]  normal_force_local
[12:15] tangent_force_local
[15:18] torque_local
[18]    contact_count
[19]    friction_count
[20:35] sensor15
```

RL 使用的聚合量对应 frame 的 `[0]` / `[1]` / 接触状态，顺序恒为 **left → right**。

源码常量：`mdp/foot_sensor.py` → `ZORN_FOOT_SCHEMA`。

---

## 2. Observation 接口（可开关）

| Term | Dim (单步) | 含义 | 噪声 DR（policy） |
|------|------------|------|-------------------|
| `foot_contact` | 2 | soft 接触 ∈[0,1] | ±0.05 |
| `foot_normal_force` | 2 | scale·\|Fz\| | ±0.05 |
| `foot_tangent_force` | 2 | scale·\|Fxy\| | ±0.05 |

- `scale = 0.01`（约 100 N → 1.0）
- Policy / Critic 组 `history_length = 5` → 足底短历史
- Critic 另含 `foot_force_history`（18 维，传感器 T=3）

### 任务注册

| Gym ID | 行为 |
|--------|------|
| `Unitree-G1-29dof-Velocity` | **基线不变**，obs = 49999 |
| `Unitree-G1-29dof-Velocity-Foot` | 足底 obs ON + μ DR + anti-slip |
| `Unitree-G1-29dof-Velocity-Foot-Off` | 同 foot 场景类但全部开关 OFF → 降级 49999 |

开关（`RobotFootEnvCfg`）：

```python
enable_foot_policy_obs = True
enable_foot_critic_obs = True
enable_friction_dr = True
enable_anti_slip = True
```

全部 `False` 时去掉足底 term / 额外 reward / 宽摩擦，行为对齐基线。

---

## 3. 在 49999 上微调

```bash
# 默认 PARTIAL_CKPT = unitree_rl_lab/model/rl/model_49999.pt
/home/mosense/guo/scripts/finetune_g1_foot.sh

# smoke
/home/mosense/guo/scripts/finetune_g1_foot.sh --smoke

# 自定义
NUM_ENVS=2048 MAX_ITERS=8000 RUN_NAME=foot_ft1 \
  /home/mosense/guo/scripts/finetune_g1_foot.sh
```

原理：

- 新网络 obs 维更大 → 不能 strict resume
- `--partial_checkpoint` 兼容 **rsl-rl 5**（`actor_state_dict` / `critic_state_dict`，键名 `mlp.0.weight`）
- 拷贝匹配权重；对 actor/critic **第一层 Linear** 做输入维扩展（旧列保留，新列保留初始化）
  - baseline actor 输入 480 → foot ~510；critic 495 → 更大（含 privileged force history）
- **不**加载 optimizer（维度变了 Adam 状态无效）
- 学习率 `3e-4`，默认 10k iter（`FootPPORunnerCfg`）

等价手动命令：

```bash
cd /home/mosense/guo/unitree_rl_lab
python scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Velocity-Foot \
  --headless --num_envs 4096 --max_iterations 10000 \
  --partial_checkpoint model/rl/model_49999.pt \
  --run_name foot_ft
```

日志：`logs/rsl_rl/unitree_g1_29dof_velocity_foot/<timestamp>_*/`

---

## 4. 摩擦自适应

| 机制 | 实现 |
|------|------|
| 随机 μ | `static/dynamic_friction_range = (0.1, 1.2)`，startup **与** reset |
| anti-slip | `feet_anti_slip`：soft contact × (脚平面速度 + 软切向/法向比) |
| 加强 slide | `feet_slide` weight `-0.35` |
| 力平滑 | `feet_force_rate`（极小权重） |

**没有** `if μ < x: then ...` 规则；策略从足底 obs + 奖励自己学「滑就慢 / 稳」。

---

## 5. 导出 ONNX → MuJoCo

```bash
# play 会 export policy.onnx 到 checkpoint 同级 exported/
cd /home/mosense/guo/unitree_rl_lab
python scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Velocity-Foot \
  --checkpoint logs/rsl_rl/unitree_g1_29dof_velocity_foot/<run>/model_XXXX.pt \
  --num_envs 16 --headless

# 安装到 g1_ctrl 策略槽
/home/mosense/guo/scripts/export_g1_foot_onnx.sh \
  --checkpoint logs/rsl_rl/unitree_g1_29dof_velocity_foot/<run>/model_XXXX.pt \
  --dest foot
```

在 `deploy/robots/g1_29dof/config/config.yaml` 将 Velocity 的 `policy_dir` 改为：

```yaml
policy_dir: config/policy/velocity/foot
```

并保证 `params/deploy.yaml` 含足底 observation 项（训练 `export_deploy_cfg` 自动写出）。

### MuJoCo 复测

```bash
# 终端 1
cd /home/mosense/guo/unitree_mujoco/simulate/build
./unitree_mujoco   # domain 0 / network 按 config

# 终端 2
cd /home/mosense/guo/unitree_rl_lab/deploy/robots/g1_29dof
./run_g1_ctrl.sh --network lo
# A 站立 → X 进入 Velocity
```

**降级说明**：C++ `observations.h` 已为 `foot_*` 注册 **零填充**。无真实足底时 ONNX 仍可跑，等价「传感器全零」。微调时的噪声 DR 有助于容忍该降级；接上 zorn ROS/真实传感器后可替换为真实读数。

---

## 文件索引

| 路径 | 作用 |
|------|------|
| `tasks/locomotion/mdp/foot_sensor.py` | obs 函数 + schema |
| `tasks/locomotion/mdp/rewards.py` | `feet_anti_slip`, `feet_force_rate` |
| `robots/g1/29dof/velocity_foot_env_cfg.py` | foot 环境 / 开关 |
| `robots/g1/29dof/__init__.py` | gym 注册 |
| `agents/rsl_rl_ppo_cfg.py` | `FootPPORunnerCfg` |
| `utils/partial_checkpoint.py` | 49999 → foot 热启动 |
| `scripts/rsl_rl/train.py` | `--partial_checkpoint` |
| `guo/scripts/finetune_g1_foot.sh` | 一键微调 |
| `guo/scripts/export_g1_foot_onnx.sh` | 导出安装 |
| `deploy/.../observations.h` | C++ foot 零填充 |

---

## 建议训练顺序

1. 确认基线：`Unitree-G1-29dof-Velocity` + `model_49999.pt` play 正常  
2. Smoke：`finetune_g1_foot.sh --smoke`（20 iter）  
3. 正式微调：10k–20k iter，盯 TensorBoard `Episode_Reward/*`、`feet_anti_slip`  
4. play + 导出 ONNX → 装 `velocity/foot`  
5. MuJoCo Sim2Sim；有 zorn 后再做 Real2Sim 力分布标定  
