# Foot-Adaptive：快走 / 小跑式 + 低 μ 稳慢 + 原地转弯

## 目标（现实预期）

| 地面 | 期望行为 |
|------|----------|
| 正常 / 高 μ | 跟大速度指令（约到 **1.2 m/s** 快走/小跑式），少滑 |
| 很滑 / 低 μ | **稳但慢**：可跟不住 1.2，自动降实际速度，少侧移/少急转 |
| 右摇杆 | 原地 / 行走中转弯（`wz` 到 **±0.6**） |

**不做**：强制「低 μ 也必须 1.2 m/s」（会摔）。

---

## 任务

`Unitree-G1-29dof-Velocity-Foot-Adaptive`

| 模块 | 设定 |
|------|------|
| 指令 limit | `vx∈[-0.5,1.2]`, `vy∈[-0.3,0.3]`, `wz∈[-0.6,0.6]` |
| 原地转 | `rel_spin_envs=0.25`, `min_spin_ang_vel=0.18` |
| 摩擦 DR | μ ∈ **[0.08, 1.2]**，startup + reset 重采样 |
| Policy obs | 与 foot_4000 相同：contact / normal / tangent（部署兼容） |
| Critic 特权 | + `foot_friction_ratio` (ρ) + `foot_slip_proxy` |
| 奖励 | slip-aware track（滑了减弱跟踪）+ 更强 anti-slip + yaw 权重 1.0 + gait period 0.72 |
| 课程 | `lin_vel_cmd_levels` + `ang_vel_cmd_levels` |
| 起点 | `model/rl/model_foot_4000.pt`（**partial**，critic 维变大） |

---

## 训练

```bash
# 推荐一键（partial 从 model_foot_4000）
<repo>/research_scripts/finetune_g1_foot.sh --adaptive

# 自定义
<repo>/research_scripts/finetune_g1_foot.sh \
  --adaptive \
  --max-iterations 8000 \
  --run-name foot_adaptive \
  --num-envs 4096
```

日志：

`logs/rsl_rl/unitree_g1_29dof_velocity_foot_adaptive/<timestamp>_foot_adaptive/`

**不要**对 Adaptive 用 strict `--resume-checkpoint` 从 4000 起训（critic 输入维更大）；必须 **partial**。
同 Adaptive run 中断后，可用同维 checkpoint **strict resume**。

---

## 部署

训完导出 ONNX 后改 `deploy/.../velocity/foot/params/deploy.yaml`：

```yaml
commands:
  base_velocity:
    ranges:
      lin_vel_x: [-0.5, 1.2]
      lin_vel_y: [-0.3, 0.3]
      ang_vel_z: [-0.6, 0.6]
```

验收：

1. 高摩擦地面 / 默认 μ：左摇杆前推 → 明显快于 4000
2. 低 μ 分区：同指令下更慢、更少摔
3. 左摇杆中位 + 右摇杆 → 原地转

---

## 与 `--turn` 的关系

| 模式 | 用途 |
|------|------|
| `--turn` | **只**加大 yaw，速度包络与 4000 相同 |
| `--adaptive` | **推荐**：速度 + 转弯 + 摩擦自适应一体 |
| `--adaptive-yaw` | NaN 后从 `model_5400` 续训：强制 `wz` 起点 ±0.4、课程门槛 0.5、到 ±0.6 |

### 角速度课程曾卡住（±0.2）的修复

第一轮 Adaptive 里 `track_ang≈0.63` 但门槛是 `weight×0.8=0.8`，yaw 永不扩。

已改（两手一起上）：

1. **强制抬起点**：`ang_vel_z` ranges 从 `±0.2` → **`±0.4`**，`rel_spin_envs=0.32`
2. **降低门槛**：`ang_vel_cmd_levels` 用 `reward_threshold_frac=0.5`（约 `0.5×1.15` 即可扩）
3. limit 仍 **`±0.6`**

```bash
# 从崩溃前安全 ckpt 续训（默认 model_5400，不加载 optimizer）
<repo>/research_scripts/finetune_g1_foot.sh --adaptive-yaw
```

TensorBoard 看 `Curriculum/ang_vel_cmd_levels` 应从 0.4 升到 0.6。

---

## 相关代码

- `velocity_foot_env_cfg.py` → `RobotFootAdaptiveEnvCfg`, `AdaptiveCommandsCfg`, `FootAdaptiveRewardsCfg`
- `mdp/rewards.py` → `track_lin_vel_xy_slip_aware`, `track_ang_vel_z_slip_aware`
- `mdp/foot_sensor.py` → `foot_friction_ratio`, `foot_slip_proxy`
- `rsl_rl_ppo_cfg.py` → `FootAdaptivePPORunnerCfg`
