# Straight-Mu（干净主线）

**目标：** 高 μ 走直线快；低 μ 降速稳。  
**母本：** `model/rl/model_49999.pt` partial。  
**不做：** turn / spin / 扩 yaw / 从 Full·MuAdapt 歪栈 resume。

## 任务

`Unitree-G1-29dof-Velocity-Foot-StraightMu`

| 项 | 设定 |
|----|------|
| Actor | 510 = 基线 + contact/Fn/Ft × history 5 |
| Critic | + ρ + slip_proxy（特权） |
| 指令 limit | **`vx≤1.5`**（对齐满杆），`vy±0.2`，**`wz±0.2`** |
| Spin | **`rel_spin_envs=0`** |
| 课程 | 只开 **lin_vel** → 逐步到 1.5；**不开 ang_vel** |
| μ DR | 约 `[0.08, 1.2]` 每 episode |
| 奖励 | full track + `stable_speed_bonus` + `lateral_slip` + `slip_under_command` |

### 满杆 1.5 与 μ 行为（设计意图）

- **部署**：`G1_CMD_GAIN_LIN=1.0` + clamp `lin_vel_x max=1.5` → 推满 = cmd 1.5  
- **训练**：`limit_ranges.lin_vel_x` 必须到 **1.5**，否则高 μ「跟上满杆」是 OOD  
- **高 μ**：脚不滑 → track + stable_speed_bonus 鼓励跟上 1.5、走直  
- **低 μ**：滑 → bonus 消失 + slip 惩罚 ↑ → **允许比 cmd 慢**，优先稳与少侧滑

## 训练

```bash
# smoke
/home/mosense/guo/scripts/finetune_g1_foot.sh --straight-mu --smoke

# 正式
/home/mosense/guo/scripts/finetune_g1_foot.sh --straight-mu \
  --max-iterations 12000 --run-name foot_straight_mu --num-envs 4096
```

日志：`logs/rsl_rl/unitree_g1_29dof_velocity_foot_straight_mu/`

崩溃后续训（同维 strict，默认不带 optimizer）：

```bash
/home/mosense/guo/scripts/finetune_g1_foot.sh --straight-mu \
  --resume-checkpoint logs/rsl_rl/unitree_g1_29dof_velocity_foot_straight_mu/<run>/model_XXXX.pt \
  --max-iterations 8000 --run-name foot_straight_mu_cont
```

## 验收（MuJoCo）

1. 满杆**只推前后**，左右回中（cmd ≈ **1.5**）  
2. 键 **3 GRIP**：希望 `|vx|` **尽量靠近 1.5**、`|vy|` 低  
3. 键 **1 ICE**：希望 `|vx|` **明显 < 1.5**、少摔、少侧滑（不必硬跟满杆）  
4. Logger 看 `|vx|` vs `|vy|`，不要只看 `|v|`

Deploy clamp 与训练一致：`lin_vel_x: [-0.5, 1.5]`，`ang_vel_z: [-0.2, 0.2]`。
