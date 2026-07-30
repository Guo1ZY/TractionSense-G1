# Unitree G1 Velocity 训练 TensorBoard 指标报告

- **任务**: `Unitree-G1-29dof-Velocity`
- **Run**: `2026-07-13_15-39-29_pipeline`
- **日志目录**:
  `<repo>/logs/rsl_rl/unitree_g1_29dof_velocity/2026-07-13_15-39-29_pipeline`
- **截图数据范围**: 约 **iter 0 → 1811**（训练进行中，数值会继续变化）
- **配置**: 4096 envs，目标 50000 iterations，`rsl-rl-lib 5.0.1`，Isaac Lab 2.3.2

---

## 1. 曲线图（导出文件）

| 文件 | 说明 |
|------|------|
| [00_dashboard.png](./00_dashboard.png) | **总览仪表盘（汇报优先）** |
| [01_core_metrics.png](./01_core_metrics.png) | 核心 4 指标 |
| [02_task_gait.png](./02_task_gait.png) | 任务 / 步态相关 reward |
| [03_penalties.png](./03_penalties.png) | 惩罚项 |
| [04_loss_curriculum_perf.png](./04_loss_curriculum_perf.png) | Loss / 课程 / FPS |
| [metrics_snapshot.txt](./metrics_snapshot.txt) | 数值快照（可能略旧于本报告） |

### 总览

![Dashboard](./00_dashboard.png)

### 核心指标

![Core Metrics](./01_core_metrics.png)

### 任务与步态

![Task & Gait](./02_task_gait.png)

### 惩罚项

![Penalties](./03_penalties.png)

### Loss / 课程 / 性能

![Loss Curriculum Perf](./04_loss_curriculum_perf.png)

> **读图注意**：曲线末尾若突然掉一截，可能是最后若干 step 日志未写全或滑动平均边界效应；判断趋势以主体区间（如 200–1600）为准。

---

## 2. 当前数值快照（约 iter 1811）

| 指标 | 开训附近 | 当前 | 变化方向 |
|------|----------|------|----------|
| Mean Reward | ≈ −0.86 | ≈ **18.1** | ↑ 明显改善 |
| Episode Length | ≈ 12 | ≈ **735**（满局约 1000） | ↑ 存活变长 |
| Track Lin Vel XY | ≈ 0.008 | ≈ **0.62** | ↑ 开始跟速 |
| Bad Orientation 终止比例 | 早期接近 1.0 | ≈ **0.50** | ↓ 摔倒减少 |
| Gait Reward | ≈ 0.004 | ≈ **0.36** | ↑ 步态成形 |
| Vel Command Curriculum | 0.10 | ≈ **0.20** | ↑ 指令速度开始扩 |
| Value Loss | ≈ 0.36 | ≈ **0.018** | ↓ Critic 更稳 |
| Entropy | ≈ 41 | ≈ **19.4** | ↓ 探索收敛中 |

**一句话结论**：训练健康，处于「能撑更久、开始跟速、尚未完全稳走」的中早期；约 **1.8k / 50k** iterations。

---

## 3. 指标词典（TensorBoard 名 → 含义）

### 3.1 核心（汇报必看）

| TensorBoard 名称 | 中文含义 | 好坏方向 | 说明 |
|------------------|----------|----------|------|
| `Train/mean_reward` | 平均总回报 | **越高越好** | 一局内所有 reward 加权求和的平均；从负到正说明策略在变好 |
| `Train/mean_episode_length` | 平均 episode 长度 | **越高越好** | 存活控制步数；本任务满局约 **1000**（20s episode，dt=0.02） |
| `Episode_Reward/track_lin_vel_xy` | 水平速度跟踪 | **越高越好** | 主任务：是否跟上 \(v_x, v_y\) 指令；权重设计约 1.0 |
| `Episode_Termination/bad_orientation` | 姿态失败终止比例 | **越低越好** | 身体倾角过大导致结束，近似“摔倒率” |

### 3.2 任务 / 步态

| TensorBoard 名称 | 中文含义 | 好坏方向 | 说明 |
|------------------|----------|----------|------|
| `Episode_Reward/track_ang_vel_z` | 偏航角速度跟踪 | 越高越好 | 转弯 / 原地转向能力 |
| `Episode_Reward/gait` | 步态奖励 | 越高越好 | 鼓励双足交替支撑（period≈0.8s） |
| `Episode_Reward/feet_clearance` | 抬脚净空 | 越高越好 | 摆动腿抬高，减少绊脚 |
| `Episode_Reward/alive` | 存活奖励 | 越高越好 | 活着就给分，与 length 相关 |
| `Metrics/base_velocity/error_vel_xy` | 速度误差 | 越低越好 | 跟踪误差的度量（若存在） |
| `Metrics/base_velocity/error_vel_yaw` | 偏航速度误差 | 越低越好 | 转向误差 |

### 3.3 惩罚 / 正则（多为负值）

| TensorBoard 名称 | 中文含义 | 说明 |
|------------------|----------|------|
| `Episode_Reward/action_rate` | 动作变化率惩罚 | 抑制关节指令剧烈抖动 |
| `Episode_Reward/feet_slide` | 脚滑惩罚 | 抑制支撑脚在地面滑动 |
| `Episode_Reward/flat_orientation_l2` | 姿态倾斜惩罚 | 鼓励躯干保持直立 |
| `Episode_Reward/base_height` | 机身高度偏差惩罚 | 高度偏离目标（约 0.78m） |
| `Episode_Reward/joint_vel` / `joint_acc` | 关节速度 / 加速度惩罚 | 鼓励更平滑运动 |
| `Episode_Reward/energy` | 能耗相关惩罚 | 抑制过大力矩/做功 |
| `Episode_Reward/joint_deviation_*` | 关节偏离默认姿态 | 手臂/腰/腿姿态整形 |
| `Episode_Reward/dof_pos_limits` | 关节限位惩罚 | 靠近极限位姿时惩罚 |
| `Episode_Reward/undesired_contacts` | 非期望接触 | 膝/手等非脚部碰地 |

> 惩罚项绝对值变大不一定坏：策略更“敢动”时惩罚常变重；关键看 **总 reward** 与 **是否还摔**。

### 3.4 终止项

| TensorBoard 名称 | 含义 |
|------------------|------|
| `Episode_Termination/bad_orientation` | 姿态过差终止（主要失败模式） |
| `Episode_Termination/base_height` | 高度过低终止（倒地） |
| `Episode_Termination/time_out` | 正常撑满 episode 结束（希望升高） |

### 3.5 优化器 / 算法健康度

| TensorBoard 名称 | 含义 | 正常现象 |
|------------------|------|----------|
| `Loss/value` | Critic 价值损失 | 训练中下降并趋稳 |
| `Loss/surrogate` | PPO 策略 surrogate loss | 在 0 附近小幅波动 |
| `Loss/entropy` | 策略熵（探索程度） | 逐渐下降，过早崩到 0 需警惕 |
| `Loss/learning_rate` | 学习率 | 自适应 schedule 时会变化 |
| `Policy/mean_std` | 动作高斯噪声标准差 | 探索强度相关 |

### 3.6 课程学习 / 性能

| TensorBoard 名称 | 含义 | 说明 |
|------------------|------|------|
| `Curriculum/lin_vel_cmd_levels` | 线速度指令课程水平 | 跟踪够好时扩大指令范围；已从 0.1 → 0.2 |
| `Curriculum/terrain_levels` | 地形难度课程 | 本配置地形较简单时变化可能不明显 |
| `Perf/total_fps` | 训练吞吐（env steps/s） | 约 4–5 万；仅反映效率 |
| `Perf/collection_time` | 采样耗时 | 越短越好 |
| `Perf/learning_time` | 反传更新耗时 | 越短越好 |

---

## 4. 如何解读“训练好不好”

建议按优先级看：

1. **Mean Reward 是否上升并趋稳**
2. **Episode Length 是否接近 800–1000**（少摔、能走完）
3. **Track Lin Vel 是否继续升高**（真正会跟指令）
4. **Bad Orientation 是否持续下降**
5. **Curriculum 是否继续上抬**（允许更快速度）

| 阶段（经验） | 大致 iter | 现象 |
|--------------|-----------|------|
| 刚起步 | 0–1k | reward 转正，length 拉长 |
| 中早期（当前） | 1k–5k | 跟速、步态出现，仍易倒 |
| 可 play 验收 | 5k–10k | 慢走可见，摔减少 |
| 较可用基线 | 2 万+ | 更稳、指令范围更大 |
| 完整默认 | 5 万 | 官方默认完整训练 |

---

## 5. 本地打开 TensorBoard

```bash
source <conda-root>/etc/profile.d/conda.sh
conda activate isaaclab-v2

tensorboard \
  --logdir <repo>/logs/rsl_rl/unitree_g1_29dof_velocity/2026-07-13_15-39-29_pipeline \
  --port 6006 --bind_all
```

浏览器访问：`http://127.0.0.1:6006`

优先勾选：

- `Train/mean_reward`
- `Train/mean_episode_length`
- `Episode_Reward/track_lin_vel_xy`
- `Episode_Termination/bad_orientation`
- `Curriculum/lin_vel_cmd_levels`

---

## 6. 重新导出图片（训练更新后）

```bash
# 可在 guo 下用之前的导出逻辑，或直接用 TensorBoard 截图
ls <workspace>/tb_plots/
```

---

## 7. 汇报可用摘要（复制即用）

> G1 velocity locomotion 训练 run `2026-07-13_15-39-29_pipeline` 曲线健康。约 1.8k/50k iter 时：Mean Reward 从负升至约 18，Episode Length 从约 12 升至约 735（满局 1000），速度跟踪约 0.62，摔倒相关终止从接近 100% 降至约 50%，速度指令课程已由 0.1 提升至 0.2。当前为中早期：能撑更久并开始跟速，尚未完全稳走。详细曲线见 `tb_plots/00_dashboard.png` 与本报告。

---

*报告生成位置：`<workspace>/tb_plots/TRAINING_METRICS_REPORT.md`*
