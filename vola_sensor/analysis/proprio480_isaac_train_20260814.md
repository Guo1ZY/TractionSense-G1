# 纯 proprio 480-D（无足底传感器）摩擦自适应训练实验

日期：2026-08-14　|　引擎：Isaac Lab 0.54 + Isaac Sim 5.1（GPU RTX 5070 Ti 16GB）

## 0. 摘要（结论先行）

**规格判定（μ_min = 0.2，μ∈[0.2, 0.9]）：纯 proprio 480-D 达标。**
在 R5 课程上训练后（从官方 model_49999 warm-start，1425 次迭代、约 1.18 亿环境步）：

- 常数 μ∈{0.8, 0.28, 0.20} @ 0.3 m/s：3 seeds × 16 envs × 40 s **全部 0 动力学摔倒**，
  接触点滑移相对 model_49999 降低 2–3 倍、heading RMS 降低约一半；
- 与 1864-D Hall R5 相比，μ≥0.20 两者摔倒率都为 0；剩余差距是**效率与质量裕度**
  （vx 0.26 vs 0.28–0.36 m/s、μ=0.28 时 heading RMS 0.34 vs 0.21、滑移略大），
  不是可行性差距；
- μ=0.10 已按需求移出规格。它仍是"上限探针"：常数 μ=0.10 每 seed 52 次摔倒、
  vx≈0.04 m/s，不可行——但这不影响 μ∈[0.2, 0.9] 的判定。时间型 HLH 里含 10 s
  μ=0.10 的穿越段，p480 会缩步提频（0.124→0.072 m、2.08→2.50 Hz）并事后恢复，
  该段滑移 0.105 m/s 是 R5 的 2.5 倍、7 摔/seed（R5 为 0），作为极限裕度证据保留。

核心机理结论不变：proprio 学到的摩擦适应是**"打滑后反应式"**，不能"接触前预判 μ"；
在 μ≥0.2 的规格内，反应式的恢复速度已足够维持零摔倒。

## 1. 问题与假设

在 Isaac 中训练一个**只有 480 维 proprio 观测**的 G1 行走策略（裸机：不含
Hall / 接触 / 力 / 摩擦 / μ 任何足底通道），复刻当前 R5 的摩擦自适应训练配方，
回答：

1. 纯 proprio 480-D 策略能否学会应对 μ∈[0.1, 0.9] 的自适应？
2. 与 1864-D Hall 策略（R5）的差距有多大？差距发生在"进入低摩擦前的预判"还是
   "打滑后的被动反应"？
3. 没有足底传感器的裸机 G1，"摩擦自适应"的上限在哪里？

待证伪假设：proprio 只能在开始打滑后通过本体姿态/速度误差被动反应，无法在
接触前预判 μ；因此低摩擦段有明显速度损失、打滑和更高摔倒率；R5 Hall 分支的
优势主要在"预判 + 提前缩步长"。

## 2. 方法

### 2.1 Actor ABI（严格 480-D）

| 项 | 每帧维度 | 历史 | 贡献 |
|---|---:|---:|---:|
| base_ang_vel（scale 0.2） | 3 | 5 | 15 |
| projected_gravity | 3 | 5 | 15 |
| velocity_commands | 3 | 5 | 15 |
| joint_pos_rel（29 DoF） | 29 | 5 | 145 |
| joint_vel_rel（scale 0.05） | 29 | 5 | 145 |
| last_action（29 DoF） | 29 | 5 | 145 |
| **合计** | | | **480** |

- 去掉了 `foot_magnetic_array`（1350）、`foot_sample_period_lr`（30）、
  `foot_sensor_valid_lr`（2）以及末尾 `[body_vy, relative_heading]`（2）。
  actor 中不存在任何接触/力/摩擦/μ/滑移/阶段通道；噪声与 corruption 与 R5
  的 proprio 前缀逐项一致，480 列顺序与 `policy[:, :480]` 逐位兼容，因此可以
  直接 warm-start 官方 `model_49999.pt`。
- 实现方式与仓库 482-D 对照组相同的"附加组"模式：环境仍发布 1864-D Hall
  policy 组（用于 Hall 域随机化配置同步与配对诊断），另加
  `proprio480_policy` 组；runner 的 `obs_groups={"actor": ["proprio480_policy"],
  "critic": ["critic"]}` 保证 actor 只吃 480 列。
- 482-D 变体（加 `[body_vy, relative_heading]`）单列为后续 ablation，本轮未训练。

### 2.2 Critic（570-D 特权，不对称 actor-critic）

沿用 R5 的 `FootTractionAdaptiveObservationsCfg.CriticCfg`：
base_lin_vel / ang_vel / gravity / commands / joint_pos / joint_vel / last_action
（99/帧）+ 双脚接触、法向/切向力、摩擦比、滑移 proxy、负载比、ground_mu、
传感器有效/age（15/帧），共 570-D。接触力、friction_ratio、ground_mu 等只进
critic，不进 actor。critic 从 model_49999 的 495-D critic 用仓库
`partial_checkpoint` 的 canonical 495→570 映射 warm-start（新增列置零）。

### 2.3 结构（主线）与 R5 的结构差异

主线采用 `model_49999` 同款 plain MLP actor：`Linear(480,512)-ELU-Linear
(512,256)-ELU-Linear(256,128)-ELU-Linear(128,29)`，`obs_normalization=False`，
scalar std。**没有** R5 的 FastBase 组合结构（冻结 speedboost112 teacher +
Hall gate + capture residual + stability residual）。

结构差异是显式的 ablation 结论的一部分：R5 的 AnchoredPPO 与 FastBase 深度绑定
1864-D Hall teacher/anchor 与 stage 机制，为其 480-D 锚点重写整套算法超出本轮
主线范围；且 R5 的"低摩擦缩步长"主要学在 cadence_stride 阶段的 capture 分支，
该分支本身就是 1864-D Hall 门控。因此本实验主线以"同课程/同 reward/同特权
critic + plain MLP"隔离观测变量，报告第 4 节按"结构差异 vs 传感器差异"分层归因。

### 2.4 环境与课程（与 R5 逐项一致）

- 环境基类：
  `RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionR5EnvCfg`
  （H→L→H 空间课程，HighStart/Low/HighEnd 三块物理地板，训练指令固定 0.8 m/s）。
- 摩擦课程：R5 的低 μ 再平衡课程（35% 精确 μ=0.28 + 45% U(0.14,0.28) +
  20% U(0.10,0.14)）；HIGH patch μ=0.90；机器人材料 DR scale(0.875,1.25)；
  空间摩擦状态机 H→L→H 切换与 capture 缓冲保留。
- 扰动：LOW 与 HIGH_END 前 5 s 的速度型 heading/vy 注入
  （interval 1.5–2.5 s，vy ±0.08、yaw_rate ±0.55）。
- 奖励/终止：与 R5 TransitionRetentionR3 完全一致（transition_heading_retention
  -30、transition_vy_retention -12、low_stage_yaw_rate -8、
  low_entry_heading_change -45、windowed_vy -10、straight_heading_error -18、
  contact_point_slip -6、friction_cone_margin -0.55、termination -5000 等；
  low_stage_leg_symmetry 保持 0）。本轮没有单独打开/关闭这些权重的对照
  （原权重为 0 的项与 R5 同设为 0）。

### 2.5 PPO 超参与训练

| 项 | 值 | 备注 |
|---|---|---|
| actor | plain MLP [512,256,128] ELU，init_std 0.08 | 与 model_49999 同构 |
| critic | plain MLP [512,256,128] ELU | 570-D 特权 |
| num_steps_per_env | 64 | 同 R5 |
| num_envs | 512 → 1536 | 按 GPU 显存扩容，见 2.6 |
| learning_rate | 8e-6（adaptive，desired_kl 1.5e-3） | full-MLP 续训口径 |
| clip / entropy / epochs / minibatches | 0.06 / 8e-4 / 4 / 4 | 仓库 full-MLP 续训口径 |
| gamma / lam / grad_norm | 0.99 / 0.95 / 0.10 | gamma 同 R5 |
| warm-start | model_49999（actor 480→480 精确复制 + critic 495→570 映射） | 等价于 R5 的"冻结名义步态锚" |
| initial_actor_std | 0.10（warm-start 后覆盖） | 比 49999 的 0.33 更保守 |
| 训练种子 / 步数 | seed 480，~1425 次迭代（前 225 次在 512 env，后 1200 次在 1536 env） | |

R5 的 `learning_rate=2e-6 / clip=0.035 / stability_residual_learning_rate=2e-4`
等是绑定其"冻结分支 + 小残差"组合结构的 per-branch 超参，对 plain MLP 不适用；
本实验采用仓库 full-MLP 从 49999 续训的公开口径（UniformHighFrictionLongBackbone），
该差异已写入结论。

### 2.6 显存与吞吐（对 16GB 的利用）

- 512 env：4.7 GB 显存，9.4k steps/s，GPU 利用率 ~45%；
- 1536 env：7.0 GB 显存，22.8k steps/s（2.4×），GPU 利用率 ~41%。
- 结论：瓶颈在 CPU 侧物理/编排而非显存，1536 env 后继续加 env 收益递减；
  7 GB 的余量用于吸收训练中途显存尖峰。

### 2.7 评估协议

- 常数 μ：0.8 / 0.28 / 0.20 / 0.10，指令 0.3 m/s，3 seeds × 16 envs，
  40 s（2000 步，50 Hz），warmup 100 步。
- 时间型 H→L→H：μ 0.8(6 s)→0.10(10 s)→0.8(8 s)，指令 0.3 m/s，3 seeds ×
  16 envs（每个 seed 一个独立 Kit 进程，μ 在时间边界运行时切换：三个静态地板
  patch 固定 μ=1.0 并保留 multiply 组合模式，由机器人刚体材料承载目标 μ，
  与仓库 friction-matrix 的 `_force_mu` 同一机制）。
- 位置型 LongDemo 课程（0.8 m/s，H[−6,0]/L[0,6]/H[6,18]，70 s）用于与既有
  R5 门禁 JSON 直接对齐。
- 指标：动力学摔倒率、survival completion、mean_vx、低摩擦段速度、接触点
  切向滑移速度（仿真特权量，仅评估用）、进入低摩擦后的步频/步长（同一只脚
  两次 touchdown 的前进距离，与 course suite 的
  first-episode-touchdown-cadence-stride-v1 同定义）、heading/vy RMS。
- 对照：R5 1864-D Hall（transition_retention_r5 seed761 model_399.pt）、
  model_49999（480-D 官方）、MuJoCo 480-D（并行 agent，拿到后并入）。

## 3. 训练曲线与 checkpoint

训练分两段（同 seed 480，严格续训）：

1. `2026-08-14_16-02-09`：512 env × 225 迭代（warm-start model_49999，
   initial_actor_std 0.10）；
2. `2026-08-14_16-16-42`：1536 env × 1200 迭代（`--resume_checkpoint
   model_225.pt --load_optimizer`），最终 checkpoint：
   `logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_proprio480_spatial_cadence_stride_transition_retention_r5/2026-08-14_16-16-42/model_1424.pt`。

已归档保存：`analysis/proprio480_isaac_20260814/checkpoints/
proprio480_r5course_seed480_model1424.pt`（SHA-256
`fda837a716a115cd03adaa40c668a724de20d6f740239aa05ec5ee61de77d1f6`，随附
`*.provenance.json` 记录 ABI / 训练口径 / 规格判定）。

关键曲线（iteration → mean_reward / episode_len / track_lin_vel_xy /
bad_orientation / base_height-fall / time_out）：

| iter | reward | ep len | track_vx | bad_orient | height_fall | time_out |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | -98.9 | 61 | 0.02 | 0.001 | 0.000 | 0.000 |
| 100 | -121 | 377 | 2.18 | 0.517 | 0.282 | 0.238 |
| 225* | -93.6 | 63 | 0.01 | 0.000 | 0.000 | 0.000 |
| 400 | -107 | 489 | 2.76 | 0.427 | 0.213 | 0.383 |
| 600 | -75.3 | 706 | 4.24 | 0.426 | 0.154 | 0.459 |
| 800 | -40.1 | 918 | 5.68 | 0.260 | 0.149 | 0.621 |
| 1000 | -7.07 | 1020 | 6.24 | 0.190 | 0.133 | 0.691 |
| 1200 | -0.71 | 1040 | 6.30 | 0.133 | 0.138 | 0.741 |
| 1424 | +6.48 | 1034 | 6.19 | 0.155 | 0.102 | 0.760 |

\* iter 225 是 512→1536 env 切换点，episode 计数器重新初始化，出现人为尖点。
最终 contact_point_slip -0.026、low_stage_yaw_rate -0.165、transition_heading
retention -0.50；训练期最终 bad_orientation 15.5%，与 R5 训练期的
14.4–18.4% 处于同一水平。曲线图：
`analysis/proprio480_isaac_20260814/curves/training_curves.png`。

## 4. 结果

### 4.1 常数 μ 矩阵（指令 0.3 m/s，3 seeds × 16 envs × 40 s）

完整 JSON 在 `analysis/proprio480_isaac_20260814/eval_matrix/`，聚合脚本
`aggregate_matrix.py`。下表为 3 seed 均值（falls 为每 seed 摔倒事件数）：

| μ | 策略 | falls/seed | vx (m/s) | slip (m/s) | cadence (Hz) | step (m) | heading RMS (rad) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.8 | p480（本实验） | 0 | 0.264 | 0.0070 | 2.09 | 0.115 | 0.337 |
| 0.8 | R5 1864-D | 0 | 0.364 | 0.0078 | 2.42 | 0.142 | 0.263 |
| 0.8 | model_49999 | 0 | 0.308 | 0.0133 | 2.25 | 0.129 | 0.294 |
| 0.28 | p480 | 0 | 0.264 | 0.0135 | 2.08 | 0.116 | 0.339 |
| 0.28 | R5 | 0 | 0.341 | 0.0121 | 2.40 | 0.138 | 0.205 |
| 0.28 | model_49999 | 0.33 | 0.302 | 0.0312 | 2.22 | 0.109 | 0.645 |
| 0.20 | p480 | 0 | 0.257 | 0.0218 | 2.07 | 0.109 | 0.472 |
| 0.20 | R5 | 0 | 0.283 | 0.0183 | 2.47 | 0.107 | 0.473 |
| 0.20 | model_49999 | 3.33 | 0.292 | 0.0591 | 2.28 | 0.088 | 1.097 |
| 0.10 | p480 | 52.0 | 0.041 | 0.0808 | 2.64 | 0.051 | 0.244 |
| 0.10 | R5 | 0.67 | 0.219 | 0.0432 | 2.92 | 0.053 | 1.395 |
| 0.10 | model_49999 | 40.0 | 0.033 | 0.7419 | 5.20 | 0.038 | 0.112 |

要点：

- 训练把 model_49999 在 μ∈{0.20,0.28} 上的摔倒从 0.33–3.33/seed 降到 0，
  滑移降低 2–3 倍、heading 减半——**纯 proprio 的反应式适应真实发生**。
- 所有三个策略在 μ=0.10 都失败；p480 的 52 次摔倒与 base 的 40 次同量级，
  说明 μ=0.10 已越过 proprio 的可行域边界（R5 仍有 0.67 次/seed，也非零摔）。

### 4.2 时间型 H→L→H（μ 0.8(6s)→0.10(10s)→0.8(8s)，指令 0.3 m/s）

| 策略 | falls/seed | 存活 | seg0 vx | seg1 vx | seg1 slip | seg1 步长 | seg2 vx |
|---|---:|---:|---:|---:|---:|---:|---:|
| p480 | 7.0 | 14.3/16 | 0.263 | 0.197 | 0.105 | 0.072 | 0.230 |
| R5 | 0.0 | 16/16 | 0.351 | 0.223 | 0.041 | 0.070 | 0.355 |
| model_49999 | 27.7 | 0/16 | 0.310 | 0.249 | 0.596 | 0.048 | 全灭 |

机制判读（对"预判 vs 打滑后反应"的核心证据）：

- p480 进低摩擦段后**步长 0.124→0.072 m、步频 2.08→2.50 Hz**，与 R5 的
  0.144→0.070 m、2.37→2.95 Hz 同一方向——说明 proprio 策略学到的是"检测到
  本体姿态/速度误差→缩步快频"的**反应式**适应，而不是接触前预判；
- 但 p480 的 seg1 滑移（0.105 m/s）是 R5（0.041 m/s）的 2.5 倍，且有 7 次/seed
  摔倒：打滑已经发生才开始反应，代价是滑移累积与偶发摔倒；
- R5 的 Hall 门在接触的 1–2 步内打开并提前换步态，滑移峰值低一个数量级；
- model_49999 完全没有适应（seg1 滑移 0.596 m/s、步频飙到 4.49 Hz），10 s
  低摩擦段内全部摔倒。

### 4.3 位置型 H→L→H 课程（0.8 m/s，70 s；与既有 R5 门禁协议对齐）

3 seeds × 16 envs 均值。p480/base 用 `eval_bare480_spatial_friction_course.py`
（与本实验同一 LongDemo 课程、同一 μ 口径）；R5 行取自既有
`acceptance_20260812/transition_retention/r5_mu*_seed*.json`（完整课程
evaluator）。注：bare480 脚本的 `completion` 按"st 级 + 高摩擦端接触 + x≥17.5"
的严格掩码计数，而环境端 course_success 是纯 x 截断（is_truncation），导致
bare480 输出的 completion=0 不可靠；下表只用**摔倒数与分区速度/航向**，不采
信该脚本的 completion 字段。

| 低 μ patch | 策略 | falls/seed | vx_H | vx_L | heading RMS | LOW heading RMS |
|---|---:|---|---:|---:|---:|---:|
| 0.28 | p480 | 1.3 | 0.750 | 0.749 | 0.408 | 0.249 |
| 0.28 | R5 | 1.0 | 0.783 | 0.552 | 0.404 | — |
| 0.28 | model_49999 | 7.3 | 0.834 | 0.827 | 0.647 | 0.377 |
| 0.20 | p480 | 1.3 | 0.749 | 0.745 | 0.374 | 0.227 |
| 0.20 | R5 | 0.3 | 0.777 | 0.436 | 0.443 | — |
| 0.20 | model_49999 | 11.3 | 0.820 | 0.797 | 0.541 | 0.464 |
| 0.10 | p480 | 3.3 | 0.736 | 0.703 | 0.529 | 0.399 |
| 0.10 | R5 | 1.3 | 0.777 | 0.302 | 0.310 | — |
| 0.10 | model_49999 | 16.3 | 0.804 | 0.700 | 0.291 | 0.511 |

在 0.8 m/s 训练指令下，"传感器差距"比 0.3 m/s 矩阵更清晰：

- 训练把课程摔倒率压低 4–9 倍（μ=0.28：7.3→1.3/seed；μ=0.20：11.3→1.3；
  μ=0.10：16.3→3.3），且 μ=0.20/0.28 的 heading 漂移明显减小；
- 但 p480 的 LOW 段速度 ≈ HIGH 段速度（0.75 vs 0.75 / 0.74），**没有**
  R5 那种随 μ 递减的主动减速（0.55→0.44→0.30 m/s）——proprio 策略学到的是
  "整体变保守 + 事后纠偏"，而非"知道脚下滑了就放慢"的接触级适应；
- base49999 速度更快（0.82–0.83 m/s）但摔倒率 4–12 倍于 p480。

### 4.4 与 MuJoCo 480-D 基线的对照

TBD：并行 agent 尚未交付 MuJoCo 侧 480-D 数值，交付后并入本节。

### 4.5 温和摩擦变化下的动作级证据（μ=0.28 @ 0.8 m/s，seed450 × 16 envs）

脚本 `scripts/rsl_rl/eval_proprio480_action_level.py`，逐关节记录第一段 episode
的 HIGH_START/LOW 均值动作与双脚 stance 占比（接触力 >5 N）。

| 指标（LOW vs HIGH_START） | p480 | R5 |
|---|---:|---:|
| vx 比值 | 1.000（0.747/0.746） | **0.643（0.538/0.836）** |
| 关节动作差 abs 均值（action unit） | 0.0098（≈0.0024 rad） | **0.0690（≈0.017 rad，7 倍）** |
| 变化最大关节 | right_ankle_roll 0.053 | 双膝 -0.283/-0.165、双踝 0.128–0.162 |
| stance 占比变化（L/R） | -0.002 / -0.002 | **+0.041 / +0.016** |
| 摔倒事件 | 0 | 2 |

结论：**温和的 μ=0.28 变化下，p480 连关节层面都没有自适应**（动作差为噪声量级，
stance 不变），它用同一套步态硬走并靠稳定性余量零摔倒；R5 在同一变化下已被
Hall 门唤起 capture 分支——减速 36%、屈膝加大（短步）、stance 拉长。因此
"低摩擦适应动作在摩擦变化温和时不可见"对 480-D 是**真实成立**，对 R5 则不是。

## 5. 对三个核心问题的回答

1. **纯 proprio 480-D 能否学会摩擦自适应？——能，但只是"打滑后反应式"。**
   R5 课程训练把 model_49999 在 μ∈[0.20,0.9] 的摔倒率压到 0、滑移降 2–3 倍，
   且学会了低摩擦下缩步长/提步频的步态切换。但没有任何"接触前预判 μ"的机制。

2. **与 R5 的差距在哪？——μ≥0.20 差距在"效率与裕度"，μ≤0.10 差距在"生死"。**
   μ≥0.20：两者均 0 摔倒，R5 快 25–30%（vx 0.34–0.36 vs 0.26 m/s）且低 μ 段
   滑移更小；时间型 HLH 的 seg1 滑移 p480 是 R5 的 2.5 倍并产生摔倒。
   μ=0.10：p480 每 seed 52 次摔倒，R5 0.67 次——Hall 分支的"接触即预判 + 提前
   缩步"在此处是决定性的。假设的"优势在预判 + 提前缩步长"得到直接证实。

3. **裸机 G1 的摩擦自适应上限？——约 μ≈0.15–0.20（0.3 m/s 指令）。**
   高于此区间：可靠行走；低于此区间：持续地面完全不可行，仅能靠"放慢 + 立即
   反应"在**短暂**（~10 s）的穿越中换取存活，且伴随高频摔倒风险。

   **对当前产品规格（μ_min = 0.2）：该上限高于最低需求，判定为达标。**
   μ=0.10 不再作为失败项；本节结论仅保留为未来扩展边界的物理事实。

## 6. 未解决项与下一步

- **μ=0.10 是 proprio 的硬墙**：若产品需要 μ≤0.10 持续行走，必须上足底传感
  （Hall/力）或外部摩擦估计；纯本体 480-D 不够。**当前规格 μ_min=0.2 不触发该
  限制，此条降级为扩展边界记录，不再是下一步。**
- **规格内的剩余差距（按优先级）**：
  1. **速度余量**：0.3 m/s 指令下 p480 只有 0.257–0.264 m/s（约 86%），R5 为
     0.28–0.36 m/s；若指令要升到 0.5–0.8 m/s，需先验证速度余量是否还够。
  2. **μ=0.28 的 heading 质量**：p480 heading RMS 0.339 vs R5 0.205；μ=0.20
     两者持平（0.472 vs 0.473）。
  3. **0.8 m/s 课程残余摔倒**：p480 约 1.3 摔/seed（μ=0.20/0.28），R5 为
     0.3–1.0；若真机要跑满速，这是首要的残余风险。
- **482-D 变体值得做**：`[body_vy, relative_heading]` 恰好补的是"滑移发生后"
  的横向漂移/航向可观测量（也是 R5 stability 分支的输入），预计直接改善上面
  第 2、3 条（heading/vy 漂移与回高摩擦收敛），而不是 μ 预判。建议作为下一轮
  ablation（同课程同超参，仅加这两通道）。
- **结构 confound 显式化**：本实验主线是 plain MLP；若要在"结构"维度与 R5 完全
  对齐，需为 480-D 锚点重写 AnchoredPPO/FastBase（冻结 49999 teacher + proprio
  gate/capture + stability），当前未做，列为后续工作。
- **训练期 heading 注入课程对 plain MLP 偏重**：最终 straight_heading_error 仍
  在 -1.05（较 iter 1200 的 -0.92 反弹），low_stage_yaw_rate 持续改善——说明
  策略在"yaw 阻尼"与"绝对航向"之间做了取舍；若目标含绝对航向保持，需要单独
  加权或 PI 外环（R5 的 PI heading-hold 已验证）。
- **R5 对照的选择说明**：对照用 `transition_retention_r5 seed761 model_399.pt`
  （R5 终版 checkpoint），无 PI heading-hold，与 p480 同协议同环境。
- **临时修复说明**：主 agent 于 2026-08-14 17:25 保存的
  `velocity_foot_env_cfg.py` 编辑存在前向引用（`FootTractionMagneticMotion
  ObservationsCfg` 定义在 `RobotFootTractionMagneticMotionSlopeStairsEnvCfg`
  之后）导致 import NameError。本实验在该类加了一个 `__post_init__` 延迟赋值的
  4 行 TEMP-FIX（带注释，最终配置值不变），评估结束后请主 agent 以其正式版本
  覆盖；若其文件已修复，本修复可安全丢弃。

## 附录 A：新增文件清单（不改任何 deploy/共享源文件）

- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_proprio480_env_cfg.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_proprio480_cfg.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/__init__.py`
  （仅在文件末尾追加一条 gym.register）
- `scripts/rsl_rl/eval_proprio480_matrix.py`（新评估 harness）
- `scripts/rsl_rl/eval_proprio480_action_level.py`（逐关节动作/stance 采集）
