# 原地转弯 / 右摇杆 yaw 微调（速度不变）

## 现象

- 部署端 **右摇杆已经映射到 `ang_vel_z`**（`LX` 左右、`LY` 前后、`RX` 转弯）。
- 但 `model_foot_4000` 训练包络只有 **`wz ∈ [-0.2, 0.2]`**，右摇杆顶满也只有很慢的转弯。
- 标准均匀采样 `(vx, vy, wz)` 几乎采不到 **「左摇杆中位 + 右摇杆转」**（`vx≈0, vy≈0, |wz|大`）的 **原地转**；`rel_standing` 会把 **三轴全置 0**，更不会练 yaw。

## 方案（已接好）

任务：`Unitree-G1-29dof-Velocity-Foot-Turn`

| 项 | 值 |
|----|----|
| 线速度 limit | `vx∈[-0.5,1.0]`, `vy∈[-0.3,0.3]`（与 4000 相同） |
| 角速度 limit | `wz∈[-0.6,0.6]`（约 3×） |
| 原地转采样 | `rel_spin_envs=0.30`，`min_spin_ang_vel=0.18`（纯 yaw） |
| 站立 | `rel_standing_envs=0.05`（全零） |
| 奖励 | `track_ang_vel_z` 权重 1.0 |
| 课程 | `ang_vel_cmd_levels` 从 ±0.2 扩到 ±0.6 |
| 起点 | `model/rl/model_foot_4000.pt` |

## 训练

```bash
# 推荐一键
<repo>/research_scripts/finetune_g1_foot.sh --turn

# 等价写法
<repo>/research_scripts/finetune_g1_foot.sh \
  --task Unitree-G1-29dof-Velocity-Foot-Turn \
  --resume-checkpoint <repo>/model/rl/model_foot_4000.pt \
  --run-name foot_turn \
  --max-iterations 6000
```

日志目录示例：

`unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity_foot_turn/<timestamp>_foot_turn/`

## 导出与部署

训练结束后：

1. 用 `play.py` 导出 ONNX（或沿用现有 export 流程）。
2. **只改 yaw clamp**，线速度保持 4000：

```yaml
# deploy/.../velocity/foot/params/deploy.yaml
commands:
  base_velocity:
    ranges:
      lin_vel_x: [-0.5, 1.0]
      lin_vel_y: [-0.3, 0.3]
      ang_vel_z: [-0.6, 0.6]   # 训练完再改；未训练前保持 ±0.2
```

3. MuJoCo + `g1_ctrl`：左摇杆中位、只拧右摇杆 → 应能原地转。

可选灵敏度（一般保持默认 1.0）：

```bash
export G1_CMD_GAIN_YAW=1.0   # 仅在 turn 模型部署后可略调
export G1_CMD_GAIN_LIN=1.0
```

## 不要做的事

- **不要**在未 finetune 前把 deploy 的 `ang_vel_z` 拉到 ±0.6（OOD 易摔）。
- **不要**用 `WideCommandsCfg` 除非还要大侧移；侧移包络变大比只加大 yaw 更难、更易不稳。
- 原地转靠 **`rel_spin_envs`**，不是把 `rel_standing` 调高。

## 相关代码

- 命令：`mdp/commands/velocity_command.py`（`rel_spin_envs` / `min_spin_ang_vel`）
- 任务 cfg：`velocity_foot_env_cfg.py` → `TurnCommandsCfg` / `RobotFootTurnEnvCfg`
- 摇杆：`deploy/.../observations.h` → `velocity_commands`（`RX → wz`）
