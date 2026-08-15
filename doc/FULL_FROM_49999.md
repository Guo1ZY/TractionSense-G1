# 从 model_49999 干净重训（推荐，替代 6600 叠训）

## 为什么不要继续用 6600 链

| 路径 | 问题 |
|------|------|
| 49999→foot→4000→adaptive→yaw→6600 | 多次目标叠加；出现 **松杆跺脚**、**高低 μ 速度差不多** |
| **49999 → Full（本任务）** | 一张奖励/指令表同时定义：走、转、站、μ 适应 |

从 49999 **partial** 热启动：旧 480 维权重保留，足底新列初始化；**不加载** 6600。

---

## 优先级（足底已稳，不主攻“修足底”）

| 优先级 | 目标 | 实现 |
|--------|------|------|
| **P0** | **μ 适应** | 每 episode 随机 μ 0.08–1.2；打滑时 track 变软 + 强 anti-slip |
| **P0** | **多速度** | curriculum 开到 vx **1.2**；track_lin 权重最高 |
| P1 | 转弯 | spin + wz→±0.6（次要） |
| P2 | 站立 | 轻量禁跺脚（继承 49999，不作为主目标） |
| 传感 | 足底 | contact/Fn/Ft 作**摩擦线索**，不是单独任务 |

### 策略怎么改（相对 49999 基线）

1. **观测**：+ `foot_contact/normal/tangent`（×history）；critic 再加 ρ、slip_proxy  
2. **物理**：摩擦 domain randomization 加宽，reset 重采样 μ  
3. **指令**：limit `vx∈[-0.5,1.2]`，课程从慢速长大速度；附带 wz±0.6  
4. **奖励核心**  
   - `track_lin_vel_xy_slip_aware`（主）：不滑 → 跟满指令；滑 → 几乎不强求速度  
   - `feet_anti_slip` / `feet_slide`（主）：罚打滑  
   - `track_ang`：中等  
   - 轻量 idle 脚动惩罚  
5. **不**从 6600 加载（避免跺脚等坏习惯）

---

## 训练

```bash
/home/mosense/guo/scripts/finetune_g1_foot.sh --from-base
```

- 任务：`Unitree-G1-29dof-Velocity-Foot-Full`
- 起点：`model/rl/model_49999.pt`（**partial**）
- 默认 **12000** iter（约 7–9 小时量级，视机器而定）
- 日志：`logs/rsl_rl/unitree_g1_29dof_velocity_foot_full/`

### 防梯度 / value 爆炸（已写进 `FootFullPPORunnerCfg`）

| 项 | 设置 | 目的 |
|----|------|------|
| learning_rate | **4e-5** | 低于早期 foot 的 1e-4 |
| max_grad_norm | **0.35** | 硬裁梯度 |
| desired_kl | **0.006** + adaptive LR | KL 过大自动降 LR |
| clip_param | **0.15** | 更小策略步 |
| num_learning_epochs | **4** | 少过拟合单 batch |
| value loss | clipped | 限制 value 更新 |
| feet_anti_slip | 输出 clamp ≤20 | 防接触尖峰 |
| feet_force_rate | ΔF clip | 历史 NaN 根因之一 |

**盯 TB：** `Loss/value` 若突然变大 / `nan`，立刻用最近 `model_XXXX.pt` **strict resume（默认不加载 optimizer）**。

中断 / NaN 后安全续训：

```bash
/home/mosense/guo/scripts/finetune_g1_foot.sh \
  --task Unitree-G1-29dof-Velocity-Foot-Full \
  --resume-checkpoint /path/to/foot_full/model_XXXX.pt \
  --run-name foot_full_resume \
  --max-iterations 12000
```

（不要加 `--load_optimizer`，除非确定未炸过。）

---

## 部署

导出 ONNX 后：

```yaml
lin_vel_x: [-0.5, 1.2]
lin_vel_y: [-0.3, 0.3]
ang_vel_z: [-0.6, 0.6]
```

MuJoCo：足底 bridge ON；键 **1/3** 测低/高 μ；松杆应站稳。

---

## 和旧入口对比

| 命令 | 起点 | 用途 |
|------|------|------|
| `--from-base` | **49999** | **推荐新主线** |
| `--adaptive-stable` | 6600 | 仅救急，不推荐 |
| `--adaptive` / `--turn` | 4000/旧 | 历史路径 |
