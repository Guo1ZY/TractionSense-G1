# Motor-Torque-Based Traction-Adaptive G1 Locomotion

日期：2026-08-03  
主仓库：`/home/mosense/guo/unitree_rl_lab`  
MuJoCo 仓库：`/home/mosense/guo/unitree_mujoco`  
分支：两个仓库均为 `feature/traction-adaptive-joint-torque-policy`

## 1. 结论与状态边界

本次已实现一个与 TractionCanonical、柔性磁足底 Teacher–Student 和已有 checkpoint 并存的独立 motor-torque traction 版本。部署 Student 的唯一输入是 G1 原生可获得的关节位置、关节速度、电机估计力矩、IMU、足端运动学、命令和上一动作。Isaac ContactSensor 力、MuJoCo 接触真值、仿真摩擦系数、terrain label 和 privileged slip label 均未进入 Student schema、Student 网络、governor API 或导出图。

状态定义如下：

| 项目 | 状态 | 证据或限制 |
|---|---|---|
| 29-DOF schema、动力学、解析力估计、接触/牵引状态机 | **implemented + tested** | 30 个专项测试通过；Isaac/MuJoCo 均实际运行 |
| temporal force correction | **trained candidate** | Stage-0，环境隔离验证集 1,200 samples |
| privileged Teacher | **PPO medium candidate** | 128 env × 24 step × 100 iter，共 307,200 transitions；不是收敛 upper bound |
| temporal Student | **distilled + PPO medium selected candidate** | 离线蒸馏头已合并进 RSL checkpoint，再完成 307,200-transition PPO；Student encoder/heads 在该 PPO 阶段冻结 |
| governor runtime | **implemented + tested** | Isaac-independent runtime、离线 replay、MuJoCo 固定权重矩阵 |
| ONNX/TorchScript | **exported + parity tested** | 固定 `[B,15,125]` 输入，可变 batch 1/2/3 实测 |
| Isaac evaluation | **3-seed short-horizon candidate evaluated** | Stage-5，3 seeds × 16 env × 300 steps；不是独立长时 test set |
| MuJoCo Sim2Sim | **pipeline validated; low-traction goal not validated** | nominal μ=0.1 单 seed 完成 4 s；Stage-5 μ=0.1 三种子全部跌倒 |
| DAgger | **implemented + one round executed; candidate rejected** | 3 seeds、7,200 Student-visited samples、PPO Teacher action/latent labels、bounded residual、随后 307,200-transition PPO；MuJoCo nominal low-μ 回归，未替换 selected candidate |
| governor 联合 on-policy fine-tuning | **not executed** | 当前 Student PPO 与 governor runtime 分离 |
| 全部 A0–A15 消融 | **registered, partially executed** | A0/A2/A4/A5/A6/A7/A13 有实际记录，其余未填虚构结果 |
| 真机 G1 | **not hardware validated; no control performed** | 未启动任何真机控制程序 |

最重要的负结果是：当前候选不能宣称已实现可靠低摩擦行走。selected candidate 在名义 μ=0.1、seed 20260803 下由 governor 完成 4.00 s，而 no-governor 在 2.32 s 跌倒；但加入 Stage-5 torque/IMU/model 随机化后，三个 seed 的 governor 生存时间为 2.86/2.48/1.56 s，均跌倒，平均 2.30±0.67 s，反而低于 no-governor 的 2.53±0.62 s。随后实际完成的一轮 DAgger+PPO 将 Stage-5 三种子 governor 平均生存时间提高到 2.56±0.49 s，且相对自身 no-governor 的 2.06±0.31 s 有改善，但三个 seed 仍全部跌倒，并且名义 low-μ 从 4.00 s 退化到 2.02 s。因此 DAgger round-1 被保留为实验候选而未选用。正式 multi-stage 训练、更多 DAgger 迭代和 governor 联合 on-policy fine-tuning 仍是进入真机前的必要工作。

## 2. Git 审计、分支与 commits

开始时保留了两个仓库中既有的触觉/部署未提交改动和全部 checkpoint、日志、模型及摩擦场景资产。本任务未使用 `git add -A`，只显式暂存新 torque 文件。

主仓库 commits：

1. `e244459 feat: add native torque foot force estimator`
2. `846e53f feat: add torque traction teacher student pipeline`
3. `2f13eac test: add torque traction evaluation and validation`
4. `cb6a945 feat: add reproducible torque traction smoke workflow`
5. `0428eb1 docs: report torque traction implementation and results`
6. `a3e51d1 feat: merge distilled torque student into on-policy training`
7. `4ae6fae fix: make torque traction reproduction environment-safe`
8. `a80a2e0 docs: finalize torque traction medium evaluation`
9. `ae9db43 fix: isolate torque task truth sensors`
10. `ed52316 docs: record self-contained torque task validation`
11. `1637691 docs: record clean snapshot validation`
12. `69c9be1 feat: add PPO teacher DAgger torque distillation`
13. `31d18b3 docs: record torque DAgger candidate evaluation`

MuJoCo commit：

1. `34def35 feat: add fixed torque traction mujoco sim2sim`

`/home/mosense/guo_1/vola_sensor` 和 `/home/mosense/docker/zorn` 不是 Git worktree；motor-torque 版本未修改这两个目录，也未修改或删除原柔性磁足底实现。

## 3. 修改文件

核心模块 `source/unitree_rl_lab/unitree_rl_lab/traction_torque/`：

- `schema.py`：唯一 125-D frame、15-frame history、12 个腿关节力矩顺序和 `EstimatedDualFootForce`。
- `dynamics.py`：35-velocity-DOF 浮动基座 inverse-dynamics residual。
- `torque_filter.py`：因果 dq/qdd/tau 滤波和 reset。
- `contact_estimator.py`：部署信号混合接触概率和滞回状态机。
- `analytical_force_estimator.py`：双脚批量正则最小二乘、物理约束和 confidence。
- `temporal_force_corrector` 位于 `networks.py`：共享双腿 GRU/TCN 修正网络。
- `traction_estimator.py`：traction utilization、margin、lower bound、event μ 和六状态滑移机。
- `history.py`：time-major oldest-to-newest history 和 episode reset。
- `randomization.py`：Stage 0–5 provisional torque/dynamics observation randomization。
- `networks.py`：Teacher、Student、GRU/TCN、辅助 heads、零门控残差。
- `isaac_observations.py`：只读取原生状态的 Isaac 125-D adapter。
- `isaac_teacher.py`、`teacher_schema.py`：与 Student 隔离的 privileged truth 路径。
- `governor.py`、`deployment.py`：两遍固定策略 runtime、安全 fallback 和 joint target。
- `rewards.py`、`rsl_models.py`、`evaluation.py`：训练监督、RSL-RL adapter、指标与消融注册。

任务与 agent：

- `velocity_torque_traction_teacher_env_cfg.py`
- `velocity_torque_traction_student_env_cfg.py`
- `tasks/traction_torque/__init__.py`
- `tasks/locomotion/agents/torque_traction_rsl_cfg.py`
- `scripts/list_envs.py` 增加独立 package discovery。

脚本：

- `collect_torque_force_dataset.py`
- `aggregate_torque_traction_dagger.py`
- `evaluate_analytical_force_estimator.py`
- `train_temporal_force_corrector.py`
- `distill_torque_traction_student.py`
- `train_torque_traction_teacher.py`
- `train_torque_traction_student.py`
- `evaluate_torque_traction.py`
- `export_torque_traction_policy.py`
- `replay_torque_traction_policy.py`
- `plot_torque_traction_results.py`
- `build_torque_student_rsl_warmstart.py`
- `evaluate_isaac_torque_rollout.py`
- `reproduce_torque_traction_smoke.sh`

MuJoCo：

- `simulate_python/torque_force_estimator.py`
- `simulate_python/torque_contact_truth.py`
- `simulate_python/run_torque_traction_sim2sim.py`
- `simulate_python/run_torque_traction_matrix.py`
- `simulate_python/compare_torque_traction_matrices.py`

自动测试：

- `test_torque_force_schema.py`
- `test_torque_force_frames.py`
- `test_inverse_dynamics_force_estimator.py`
- `test_contact_estimator.py`
- `test_torque_filter.py`
- `test_torque_traction_state.py`
- `test_torque_traction_networks.py`
- `test_torque_traction_governor.py`
- `test_no_privileged_observation_leak.py`
- `test_torque_traction_export.py`

## 4. 机器人、控制频率和 joint/action 顺序

- action dimension：29
- action scale：0.25 rad
- physics dt：0.005 s
- decimation：4
- policy dt：0.02 s（50 Hz）
- 名义质量：35.2793 kg
- checkpoint：`model/rl/model_49999.pt`
- MuJoCo/Isaac default pose、PD、effort limits 均从现有 canonical deployment constants 复用；未替换资产、惯量或主要碰撞配置。

准确的 29-D action/joint 顺序：

1. `left_hip_pitch_joint`
2. `right_hip_pitch_joint`
3. `waist_yaw_joint`
4. `left_hip_roll_joint`
5. `right_hip_roll_joint`
6. `waist_roll_joint`
7. `left_hip_yaw_joint`
8. `right_hip_yaw_joint`
9. `waist_pitch_joint`
10. `left_knee_joint`
11. `right_knee_joint`
12. `left_shoulder_pitch_joint`
13. `right_shoulder_pitch_joint`
14. `left_ankle_pitch_joint`
15. `right_ankle_pitch_joint`
16. `left_shoulder_roll_joint`
17. `right_shoulder_roll_joint`
18. `left_ankle_roll_joint`
19. `right_ankle_roll_joint`
20. `left_shoulder_yaw_joint`
21. `right_shoulder_yaw_joint`
22. `left_elbow_joint`
23. `right_elbow_joint`
24. `left_wrist_roll_joint`
25. `right_wrist_roll_joint`
26. `left_wrist_pitch_joint`
27. `right_wrist_pitch_joint`
28. `left_wrist_yaw_joint`
29. `right_wrist_yaw_joint`

12-D 腿部 `tau_est` 顺序是左腿六关节后右腿六关节：hip pitch、hip roll、hip yaw、knee、ankle pitch、ankle roll。它在 canonical action 中的显式 indices 为 `(0,3,6,9,13,17,1,4,7,10,14,18)`，不假设连续存储。

## 5. 部署 Student observation schema

单帧 125-D：

| term | dim | 处理 |
|---|---:|---|
| base angular velocity | 3 | base frame，×0.2 |
| projected gravity | 3 | dimensionless |
| raw command | 3 | vx、vy、yaw rate |
| joint position relative to default | 29 | rad |
| joint velocity | 29 | ×0.05 |
| previous action | 29 | canonical action order |
| 12 leg-joint `tau_est` | 12 | 各关节除以 effort limit |
| estimated dual-foot force | 6 | 除以 `mass*9.81`，clip ±3 |
| contact probability | 2 | left、right |
| analytical force confidence | 2 | left、right |
| foot planar rigid-body velocity | 4 | world ground tangent；明确是 ankle/foot proxy，不是接触点真值 |
| IMU linear acceleration | 3 | base frame specific force，除以 9.81 |

history 为 15 帧、0.30 s、50 Hz，shape `[batch,15,125]`，严格 time-major、oldest-to-newest；等价 flatten 为 `[batch,1875]`。reset 将对应环境全部历史、滤波器、接触状态机、力估计器和滑移状态清零。

Student critic 保留 baseline 495-D、5-frame term-major privileged critic，不作为导出输入。Teacher frame 为 `96 proprio + 3 command + 149 privilege = 248`，5 帧后为 1,240-D。

明确禁止进入 Student/导出模型的字段：ContactSensor force、ground-truth contact force、ground friction μ、terrain friction label、privileged slip label、future friction。

## 6. 足端力坐标和统一 schema

唯一顺序：`[L_Fx,L_Fy,L_Fz,R_Fx,R_Fy,R_Fz]`。

- frame：对应左右 `ankle_roll_link` 局部坐标。
- `+x`：脚尖方向。
- `+y`：机器人左侧。
- `+z`：向上。
- 解析输出单位：N。
- 策略输入：`F_hat_N / (robot_mass_kg * 9.81)`。

`EstimatedDualFootForce` 同时包含 timestamp、左右三轴力、左右接触概率、confidence、residual norm 和 Jacobian condition score。Isaac、MuJoCo 与离线数据集使用同一字段语义。

## 7. inverse dynamics 与解析力估计

Isaac 实际 PhysX tensor 已运行确认：mass matrix `[N,35,35]`、Jacobian `[N,30,6,35]`、gravity/Coriolis `[N,35]`，因此没有忽略浮动基座。

实现计算：

```text
tau_model_full = M(q) qdd + h(q,dq)
r_requested = tau_est - tau_model_joint
contact_generalized_force = tau_model_joint - tau_est = -r_requested
r_leg ≈ Jv_leg(q)^T Ffoot
```

代码同时保留用户要求的 `tau_est_minus_model` 诊断和作用在机器人上的 physical contact generalized force，避免隐藏符号反转。`Jv` 先由世界坐标旋转到对应 ankle-roll local，再选取该腿六关节列。

每脚求解：

```text
min_F ||W(Jv^T F-r_leg)||²
    + 0.04 ||F-F_previous||²
    + 2e-4 ||F||²
```

默认约束：active contact 时 `Fz>=0`；inactive 以 0.06 s time constant 衰减；最大力 900 N；最大变化率 8,000 N/s；奇异值条件和残差共同形成 confidence。整个 Isaac 求解是 GPU batched，无 per-environment Python loop。

MuJoCo 使用 `mj_fullM`、`qfrc_bias`、free-base qacc、`qfrc_actuator` 和 `mj_jacBody`。MuJoCo 内部 joint order 通过 semantic joint id 显式重排到 canonical 29-D order；policy estimator 文件不调用 `mj_contactForce`。

## 8. 接触估计器

输入包括 foot height、vertical/planar velocity、每腿六关节 torque、estimated Fz、joint configuration、IMU acceleration 和 recent state；可选 gait phase。默认 on/off threshold 为 0.62/0.38，debounce 0.04 s，minimum hold 0.08 s，probability low-pass time constant 0.04 s。

特征权重为 force 0.40、height 0.20、velocity 0.18、torque 0.14、history 0.08；IMU 作为动态 gate。仿真真值只用于 reward、label 和 metric。

## 9. traction 与 slip estimator

`rho = sqrt(Fx²+Fy²)/(|Fz|+1 N)` 始终命名为 `traction_utilization`，未滑动时不称为精确摩擦系数。

状态机：no contact、stable contact、high utilization、slip candidate、confirmed slip、recovery。默认 speed on/off 为 0.12/0.06 m/s；candidate threshold 0.58；确认 0.06 s；recovery 0.20 s；low-pass 0.06 s。

多帧风险融合 estimated force、foot planar velocity/acceleration、force growth、torque residual、IMU motion、contact probability 和 estimator confidence。`friction_lower_bound` 只由未确认滑移时已利用 traction 更新；`slip_event_mu_estimate` 只在 confirmed-slip 起点记录 `Ft/Fn`。

## 10. temporal correction、Teacher 和 Student

### Analytical + temporal correction

每腿输入 26-D/帧：tau6、analytical force3、q6、dq6、planar velocity2、IMU3。左右腿共享 48-hidden GRU（也实现 TCN 选项），输出 delta-F、confidence 和 gated correction。ground-truth force 只作为监督 target。

### Teacher

RSL Teacher privilege 149-D，包含 canonical simulated truth（ground μ、理想力、contact、slip diagnostics、base velocity、随机化信息）与 analytical-force residual/condition diagnostics。encoder 为 `149→128→64→16`。actor 输入为 baseline480+latent16=496，critic 为 baseline495+latent16=511，action 保持 29。

### Student

共享 per-leg 28-D temporal encoder：q6、dq6、tau6、force3、contact1、confidence1、planar velocity2、IMU3。另有 96-D proprio GRU；融合 `48+48+96→128→latent16`。heads 输出 corrected force6、contact2、slip2、utilization2、margin2、confidence2、latent16。

保留 baseline actor，traction residual action branch 和 force correction branch 均零初始化/近零 gate。初始 Stage-0 candidate 的 action 与旧 baseline bit-exact，实际离线验证的 max/mean action absolute error 均为 0。DAgger round-1 允许 residual 学习 PPO Teacher 动作，但通过 `limit*tanh` 将每个 action component 的残差显式限制到 ±1.0，并增加自动测试；未受训的随机新分支仍不会破坏 baseline。

RSL Student PPO head 使用 `480 baseline proprio + 16 latent + 3 command = 499`，其新 19 列 warm-start 为零。

## 11. dynamics/signal randomization

所有范围都标记为 provisional engineering priors，不声称来自 G1 真机统计。Stage：0 ideal；1 torque scale/bias/noise；2 delay/filter；3 mass/inertia/COM/Jacobian/model mismatch；4 drift/dropout/quantization/saturation；5 combined。

主要默认范围：tau scale 0.94–1.06、episode bias ±1.5 Nm、noise 0.45 Nm、quantization 0.05 Nm、saturation 120 Nm、delay 0–2 frames、state dropout 0.002/step；q noise 0.0015 rad、dq noise 0.025 rad/s、qdd noise 0.8 rad/s²；IMU noise 0.08 m/s²、bias ±0.12 m/s²、delay 0–2 frames；mass scale 0.94–1.06、inertia 0.92–1.08、COM ±0.008 m、gear efficiency 0.92–1.0、Jacobian scale 0.97–1.03/coupling std 0.008。

训练场景事件继承现有 G1 baseline，同时使用现有 per-foot friction event：startup μ 0.05–1.2、50% 左右不对称、1.5–3.0 s interval friction transitions。

## 12. Traction-adaptive governor

Governor API 只接受 Student slip probability、utilization、margin、contact、force confidence、foot relative velocity、slip duration 和 current velocity。它没有 ground μ、ContactSensor force 或 privileged latent 参数。

默认 normal limits：vx 1.5 m/s、vy 0.6 m/s、yaw 1.2 rad/s、acceleration 2.0 m/s²、deceleration 2.5 m/s²。风险时 acceleration 可降至 0.35 m/s²、deceleration 至 0.8 m/s²、push-off scale 至 0.28；persistent slip speed scale 0.25；low-confidence fallback scale 0.45。

risk enter/exit 为 0.48/0.28，debounce 0.06 s，hold 0.20 s，persistent slip 0.12 s。fast-down time constant 0.08 s，slow recovery 0.90 s。单脚支撑 risk gain 1.12。状态为 normal、utilization limiting、persistent slip、low-confidence fallback。

两遍 runtime 先用 raw command 输出 traction heads，再将 adjusted command 只写入 newest history frame并重新运行固定权重策略；测试确认 preserved baseline actor 实际读取了 governed command。disabled governor 路径严格返回 raw command 和 scale=1。

## 13. Rewards

保留 baseline velocity/yaw tracking、alive、orientation、base height、gait、slide、clearance、undesired contacts、energy、joint limits、action rate 和 joint deviation rewards。新增：

| reward/penalty | weight | truth 使用范围 |
|---|---:|---|
| force estimation supervision | -0.10 | training only |
| contact estimation supervision | -0.08 | training only |
| estimated utilization above warning | -0.04 | deployable estimate |
| estimator temporal residual consistency | -0.01 | deployable residual |
| tangential push | -0.05 | training truth |
| ground-truth slip | -0.40 | training truth |
| high-traction unsupported slowdown | -0.20 | training diagnostics |

Student 决策观测仍只含部署信号；truth 仅形成 reward/label。

## 14. checkpoint warm start

实际 live RSL log：

- Teacher actor first layer `(512,480)→(512,496)`；复制 8 tensors，扩展 1 tensor。
- Teacher critic `(512,495)→(512,511)`；复制 7 tensors，扩展 1 tensor。
- Student actor `(512,480)→(512,499)`；复制 8 tensors，扩展 1 tensor。
- Student critic 495-D；复制 8 tensors，无扩展。
- action std 已加载并由训练脚本的 positivity/finite guard 保护。

新列全部从零开始。初始 Stage-0 Student 独立 baseline action regression：max error 0，mean error 0。

第一次 Student medium PPO 审计发现，只从 baseline partial-load 会保留随机初始化的 traction auxiliary heads，因此该 run 被标为诊断 run，不作为部署候选。随后新增 `build_torque_student_rsl_warmstart.py`：先取得匹配的 499-D RSL template，将 baseline locomotion actor/critic 与离线蒸馏 Student 的 36 个参数显式合并；19 个新增 actor 输入列保持零。实际合并报告为 missing parameters 0，baseline action max/mean absolute error 分别为 `1.1444092e-5` 和 `1.5927764e-6`。PPO 中冻结 Student encoder/auxiliary heads，只更新 locomotion actor/critic，避免 PPO reward 无监督地破坏已蒸馏输出。

DAgger bridge 使用同一 36-parameter 显式合并机制，并把 bounded distilled action residual 加到 RSL locomotion mean；RSL rollout、TorchScript 和 ONNX 三条路径使用同一计算。round-1 合并报告：missing parameters 0；locomotion head 对 baseline 的 max/mean error 为 `1.1444092e-5 / 1.5927764e-6`；已蒸馏 residual 的随机-history max/mean absolute magnitude 为 `0.33703 / 0.24437`，所以 combined action 有意不再声明为 exact baseline。

## 15. 实际 Isaac 数据与 estimator 结果

### Analytical Stage-0

数据：8 environments × 200 steps = 1,600 robot frames / 3,200 foot samples，seed 20260803。

| metric | Fx | Fy | Fz |
|---|---:|---:|---:|
| MAE (N) | 29.910 | 19.065 | 114.029 |
| RMSE (N) | 50.446 | 35.604 | 155.579 |

force-direction error 21.895°；contact precision 0.7874、recall 0.7785、F1 0.7830；swing false-force mean 83.145 N；nonfinite 0；latency mean 2.817 ms、p95 3.007 ms。左右 stance force correlations 较低，尤其 Fz 为负相关，因此 analytical-only 当前不具备足够精度。

### Analytical + temporal correction

数据：12 env × 400 steps；train env 0–8，validation env 9–11，共 1,200 validation samples；300 epochs。

| component | analytical MAE N | corrected MAE N | analytical RMSE N | corrected RMSE N |
|---|---:|---:|---:|---:|
| L Fx | 26.279 | 11.324 | 47.003 | 33.655 |
| L Fy | 14.972 | 7.775 | 27.099 | 17.465 |
| L Fz | 109.080 | 35.449 | 157.070 | 80.863 |
| R Fx | 27.831 | 9.329 | 41.864 | 21.958 |
| R Fy | 16.831 | 6.726 | 28.190 | 16.681 |
| R Fz | 114.826 | 34.640 | 151.737 | 70.934 |

修正网络输入不含 privilege，但训练 target 是仿真 ground-truth force。结果仅适用于该 Stage-0 环境隔离 split，不外推为真机精度。

## 16. Teacher/Student 训练记录

离线 privileged auxiliary Teacher：160 epochs。离线 Student：300 epochs，validation 1,200 samples。slip label 为 `exact simulated contact AND ankle rigid-body planar-speed proxy >0.12 m/s`，不是 contact-point slip truth。

Student validation：slip precision 0.7013、recall 0.4000、F1 0.5094、AUC 0.8343；normalized force component MAE 为 `[0.05069,0.03183,0.16721,0.04389,0.02969,0.17058]`；utilization MAE 0.09313；margin MAE 0.31105。

Teacher medium PPO：128 env、24 steps/env、100 iterations，共 307,200 transitions。iteration 99 的 value loss 0.0084、surrogate loss -0.0166、mean reward 39.31、mean episode length 970.97、velocity XY error 0.1480、yaw error 0.3957、timeout fraction 0.9297。该 run 实际完成且无 NaN/Inf，但 307,200 transitions 仍不等于正式收敛 upper bound。

最终 Student medium PPO：同为 307,200 transitions，从“baseline + distilled auxiliary heads”合并 checkpoint 开始；iteration 99 的 value loss 0.0396、surrogate loss -0.0078、mean reward 39.56、mean episode length 962.38、velocity XY error 0.2030、yaw error 0.5269、timeout fraction 0.9303。该 run 是 **trained candidate**，不是 governor 联合 fine-tuning，也不是鲁棒性通过证明。先前仅从 baseline 启动的 medium run 因 auxiliary heads 随机而被审计淘汰，未用于最终导出或指标。

### DAgger round-1 与候选选择

从 selected Student 在 Stage-5 下实际采集 seeds 20260803/20260804/20260805，每个 8 env × 300 steps；聚合后为 300 steps × 24 env = 7,200 Student-visited samples。每个 sample 同时记录独立的 `[15,125]` Student history 和 `[5,248]` privileged Teacher history；后者仅传入冻结 PPO Teacher生成动作/latent label，元数据明确 `teacher_history_is_policy_input=false`。采样对 slip proxy、contact transition 和 high-utilization/no-slip 状态加权。Student loss实际包括 action、latent、slip、utilization、margin、contact、force、confidence 和 temporal smoothness。

bounded DAgger validation 使用环境隔离的 1,800 samples：slip proxy precision 0.4806、recall 0.6414、F1 0.5495、AUC 0.9075；normalized force MAE `[0.04348,0.02874,0.17477,0.03866,0.03356,0.20368]`；utilization MAE 0.10791；margin MAE 0.25872；PPO Teacher action MAE 0.06797；Teacher latent MSE 0.25746。slip label 仍是 `exact simulated contact AND ankle rigid-body planar-speed proxy >0.12 m/s`，不是 contact-point slip truth。

该 warm start 随后完成 128 env × 24 step × 100 iteration = 307,200-transition Student PPO。iteration 99：value loss 0.1470、surrogate loss -0.0059、mean reward 35.01、mean episode length 936.20、velocity XY error 0.1848、yaw error 0.6007、timeout fraction 0.8672。该 run 是 **trained candidate**，但下述 MuJoCo nominal regression 触发 selection gate，因此被拒绝为最终推荐模型。

checkpoints：

- temporal corrector：`artifacts/traction_torque/temporal_force_corrector_stage0.pt`
- offline Student/Teacher candidate：`artifacts/traction_torque/torque_student_distilled_stage0.pt`
- RSL merged warm start：`artifacts/traction_torque/torque_student_rsl_warmstart_stage0.pt`
- Teacher medium：`logs/rsl_rl/g1_29dof_torque_traction_teacher/2026-08-03_10-44-36_torque_teacher_medium_seed20260803/model_99.pt`
- Student distilled + PPO medium：`logs/rsl_rl/g1_29dof_torque_traction_student/2026-08-03_10-54-10_torque_student_distilled_medium_seed20260803/model_99.pt`
- DAgger aggregate：`artifacts/traction_torque/dagger_round1_aggregate.npz`
- bounded DAgger Student：`artifacts/traction_torque/torque_student_dagger_round1_bounded.pt`
- DAgger RSL warm start：`artifacts/traction_torque/torque_student_rsl_dagger_round1_bounded_warmstart.pt`
- DAgger + PPO medium（实验候选，未选用）：`logs/rsl_rl/g1_29dof_torque_traction_student/2026-08-03_11-27-38_torque_student_dagger_round1_bounded_medium_seed20260803/model_99.pt`

### Isaac fixed-candidate evaluation

最终 Student checkpoint 在 Stage-5 randomization 下运行 seeds 20260803/20260804/20260805；每个 seed 16 env × 300 policy steps，即每环境 6 s。全部非可选数值字段 nonfinite count 为 0。跨种子结果：

| metric | mean | sample std |
|---|---:|---:|
| velocity tracking XY MAE (m/s) | 0.1101 | 0.0521 |
| 有 fall event 的 env 比例 | 0.2292 | 0.2366 |
| fall events / robot-second | 0.0903 | 0.0968 |
| GT slip proxy rate | 0.0968 | 0.0300 |
| action delta L2 | 1.0659 | 0.4422 |

三个 seed 的 contact F1 分别为 0.7593/0.7648/0.7251。部署解析 traction state 在 0.5 threshold 下 slip-proxy F1 都为 0，AUC 为 0.5599/0.6648/0.5786，说明当前 slip confirmation 过于保守。标签仍是 `simulated contact AND ankle rigid-body planar-speed proxy >0.12 m/s`，不是 contact-point slip truth。`slip_event_mu_estimate` 在未确认事件时按 schema 有意为 NaN，评测单独统计 undefined，不将其误报为 numerical failure。

同一 Stage-5 评测也对 DAgger round-1+PPO 候选运行了完全相同的三种子短 horizon：velocity MAE `0.0907±0.0244` m/s、有 fall event 的 env 比例 `0.1042±0.1804`、fall events/robot-second `0.0208±0.0361`、slip proxy rate `0.1282±0.0365`、action delta L2 `0.7519±0.1685`，nonfinite count 为 0。相对 selected candidate，它改善了短程 tracking、falls 和 action smoothness，但 slip proxy rate 从 0.0968 增至 0.1282；该 Isaac 改善不足以抵消 MuJoCo nominal low-μ 回归。

## 17. MuJoCo fixed-policy Sim2Sim

MuJoCo policy 只读取 q、dq、qfrc_actuator-derived tau、simulated IMU、kinematics 和 analytical estimate。`torque_contact_truth.py` 在 policy history/action 完成之后单独调用 contact truth，仅用于指标。MuJoCo 中没有训练或微调。

最终同版本矩阵使用 distilled + PPO-medium 固定 TorchScript，seed 20260803，4 s horizon：

| scenario | survival s | fell | velocity tracking MAE m/s | force MAE N | GT contact-point slip rate | governor active | mean speed scale |
|---|---:|---|---:|---:|---:|---:|---:|
| high μ=0.8 | 4.00 | no | 0.0692 | 59.900 | 0.0300 | 0.125 | 0.758 |
| low μ=0.1 | 4.00 | no | 0.3108 | 65.102 | 0.5500 | 1.000 | 0.435 |
| abrupt 0.8→0.1 at 2 s | 3.00 | yes | 0.1536 | 60.193 | 0.2133 | 0.453 | 0.653 |
| asymmetric 0.1/0.8 | 4.00 | no | 0.2327 | 62.869 | 0.3475 | 0.925 | 0.579 |
| μ=0.35 + Stage-5 estimator randomization | 4.00 | no | 0.0763 | 62.586 | 0.0550 | 0.120 | 0.756 |

所有矩阵 nonfinite count 为 0。这里的 ground-truth slip metric 是 contact-point relative tangential velocity >0.12 m/s，是真值指标，不进入 policy。

同版本 no-governor controls：high μ 4.00 s 未跌倒、velocity tracking MAE 0.0680 m/s；low μ 2.32 s 跌倒、MAE 0.3305 m/s。名义单 seed 下 governor 将 low-μ survival 从 2.32 s 提到完整 4.00 s，但这只是 nominal initial-state evidence。

更严格的 μ=0.1 + Stage-5 torque/IMU/model-randomization 三种子评测给出：

| mode | survival s | velocity MAE m/s | GT slip rate | force MAE N | activation | speed scale |
|---|---:|---:|---:|---:|---:|---:|
| governor | 2.30±0.67 | 0.3358±0.0320 | 0.4612±0.0579 | 68.418±3.383 | 1.000±0.000 | 0.450±0.033 |
| disabled | 2.53±0.62 | 0.3535±0.0526 | 0.4696±0.0331 | 65.872±5.002 | 0 | 1 |

两个 mode 的三个 seed 都跌倒。Governor 略降 tracking error 和 slip rate，但平均 survival 未提升，因此 **Stage-5 robust low-friction gate 未通过**。这些是 sample mean±sample std，不是置信区间。名义矩阵所有 nonfinite count 为 0；MuJoCo slip 指标使用 ground-truth contact-point relative tangential speed，只做评测，不进入 policy。

DAgger round-1+PPO 固定策略也运行了相同 5-scenario nominal matrix：high μ 生存 4.00 s、low μ 2.02 s、abrupt drop 3.70 s、asymmetric 4.00 s、combined Stage-5 μ=0.35 生存 4.00 s。对应 velocity MAE 为 `0.0796/0.3161/0.1601/0.1252/0.0723` m/s；high-μ governor activation 0.61，明显高于 selected candidate 的 0.125，构成 unsupported slowdown/activation 回归。最关键的是 nominal low-μ 从 selected candidate 的完整 4.00 s 退化为 2.02 s 跌倒，因此没有覆盖 selected checkpoint/export。

DAgger candidate 的 μ=0.1 + Stage-5 三种子对照为：governor survival `2.56±0.49` s、velocity MAE `0.2862±0.0207` m/s、GT slip rate `0.5692±0.0049`；disabled survival `2.06±0.31` s、velocity MAE `0.3304±0.0104` m/s、GT slip rate `0.5668±0.0688`。Governor 相对该候选自身使平均 survival 增加 0.50 s 并改善 tracking，但两种 mode 的所有 seed 仍跌倒且 slip rate 未改善，故 robust gate 仍失败。

## 18. 消融注册与实际执行边界

已在 `evaluation.py` 注册 A0–A15，包括 raw tau、analytical force、history、temporal correction、no/full governor、Teacher upper bound、no IMU、no tau history、force-free、history length、GRU/TCN、randomization、no contact classifier 和 no confidence fallback。

实际执行：

- A0：旧 proprio actor warm start regression；MuJoCo high-μ no-governor smoke。
- A2：Analytical Stage-0 estimator evaluation。
- A4：Analytical vs temporal correction held-out comparison。
- A5/A6：最终 MuJoCo high/low μ no-governor vs governor，包含 Stage-5 low-μ 三种子。
- A7：Teacher network/PPO medium candidate，不是收敛 upper-bound 评测。
- A13：MuJoCo Stage-0 与 Stage-5 estimator-randomization 对照。
- DAgger round-1：Student-visited aggregation、PPO Teacher labels、bounded residual、Student PPO、Isaac 3-seed 和 MuJoCo nominal/Stage-5 3-seed；属于训练流程实验，不是额外 A 编号。

未执行：A1、A3 的独立 actor performance、A8–A12、A14–A15 正式 locomotion evaluation，以及任意多-seed confidence interval。未给这些实验填入结果。

## 19. 导出包与 hashes

selected candidate 目录：`artifacts/traction_torque/export_student_distilled_ppo_medium/`。DAgger round-1 的导出保存在 `artifacts/traction_torque/export_student_dagger_round1_bounded_ppo_medium/`，但因上述 selection gate 失败而标记为 rejected experiment；两者均保留且互不覆盖。

包含 `metadata.json`、`observation_schema.json`、`joint_order.json`、`force_frame.json`、`dynamics_estimator.json`、`torque_randomization.json`、`command_governor.json`、TorchScript、ONNX、README 和 `sha256.json`。

输入 float32 `[batch,15,125]`；输出 action29、estimated force6、contact2、slip2、utilization2、margin2、confidence2。导出 network 的 force head 使用 normalized force domain；analytical runtime 在归一化前保留 N。

导出时对 PyTorch reference 的 parity：TorchScript max abs error `3.814697e-6`；ONNX max abs error `1.811981e-5`。独立重载后 batch 1/2/3 的 TorchScript–ONNX max absolute error 为 `7.629e-6 / 4.578e-5 / 2.670e-5`，全部输出 shape 正确且 finite。

主要 SHA-256：

- temporal corrector checkpoint：`c3de46ed0612e054e6d82f0ee3c30d7ec35304b83769e90d33bba0359e1d0217`
- distilled candidate：`19e960ba966503fb40df81c175a789f470ab0eb22e0fbd4176f2ba1bf03b259f`
- Teacher PPO medium model：`1492b4e8688393f060edddf461800fcbcc71eab09d94538bc515aaaa09c745ca`
- Student distilled + PPO medium model：`ed14111f3e7f9e45d592afcb4f3a4018abfec7054a11c040220c4a801e59dd4e`
- ONNX：`6bd1fbef975c3b39f689d7fb020051f4648637f9e122df3ef6882cc6cc31e4bc`
- TorchScript：`04723d3983c32bfb16632040eb83d58523d00daf9f6e77152fc2e8723d2f3074`

rejected DAgger round-1 hashes：PPO model `b0127ab37768469cc738138892192fe110f66eea26d406956d41c08aafc4fd7f`；ONNX `b7bfa3bd759493c3be61940049a70b86584b3f9d346eb3673e16e471713c5f65`；TorchScript `4386fba26c3ad9d91f69bfdcef42936a1723ae7d6c128eb9c67a6677c5223b92`。该导出的独立 TorchScript–ONNX batch 1/2/3 max error 为 `1.526e-5 / 6.867e-5 / 3.433e-5`，shape 正确、nonfinite 0；导出正确不等于策略性能通过。

## 20. 自动测试与静态检查

专项 pytest：`30 passed`。除当前工作区外，还将主仓与 MuJoCo 各自的 `HEAD` 解包为不含任何未跟踪文件的兄弟目录快照，结果同为 `30 passed`，验证提交本身不借用用户脏文件。覆盖：29-D order；125/1875 schema；左右脚/符号；double/single support；inactive decay；singular Jacobian confidence；无 NaN；contact hysteresis/debounce/reset；slip state machine；history reset；zero-gated baseline regression；bounded action residual；governor disabled strict control；runtime command rewrite；privileged API leak；MuJoCo truth隔离；ONNX/TorchScript shape 和数值 parity。

`compileall`、shell syntax 和 `git diff --check` 均通过。科研结果图脚本的静态 preflight 为 14 pass、0 warn、0 fail，并人工检查了最终 PNG。图使用单一 Python/matplotlib backend，183×125 mm，SVG/PDF editable text、600-dpi TIFF；无数据行排除，summary panel 明确为 nominal single-seed，Stage-5 多 seed 结果在表格中单列。

结果图和 source data：`artifacts/traction_torque/figures/`。

## 21. 离线 replay

使用 Stage-0 dataset env0 的 400 frames 通过固定 TorchScript→traction heads→governor→第二遍 policy。结果：nonfinite 0、governor activation ratio 0.025、mean speed scale 0.88861。该流程只做离线计算，不发送 joint command。

## 22. 实际执行命令

Analytical dataset/evaluation：

```bash
/home/mosense/miniconda3/envs/isaaclab-v2/bin/python scripts/traction_torque/collect_torque_force_dataset.py --num_envs 8 --steps 200 --warmup_steps 50 --seed 20260803 --randomization_stage 0 --benchmark_latency --headless --output artifacts/traction_torque/analytical_stage0_seed20260803.npz
/home/mosense/miniconda3/envs/isaaclab-v2/bin/python scripts/traction_torque/evaluate_analytical_force_estimator.py artifacts/traction_torque/analytical_stage0_seed20260803.npz --output artifacts/traction_torque/analytical_stage0_seed20260803_metrics.json
```

Temporal correction/distillation/RSL merge：

```bash
/home/mosense/miniconda3/envs/isaaclab-v2/bin/python scripts/traction_torque/train_temporal_force_corrector.py artifacts/traction_torque/correction_stage0_seed20260803.npz --epochs 300 --seed 20260803 --output artifacts/traction_torque/temporal_force_corrector_stage0.pt
/home/mosense/miniconda3/envs/isaaclab-v2/bin/python scripts/traction_torque/distill_torque_traction_student.py artifacts/traction_torque/correction_stage0_seed20260803.npz --teacher_epochs 160 --student_epochs 300 --seed 20260803 --output artifacts/traction_torque/torque_student_distilled_stage0.pt
/home/mosense/miniconda3/envs/isaaclab-v2/bin/python scripts/traction_torque/build_torque_student_rsl_warmstart.py --template logs/rsl_rl/g1_29dof_torque_traction_student/2026-08-03_10-46-18_torque_student_medium_seed20260803/model_0.pt --baseline model/rl/model_49999.pt --distilled artifacts/traction_torque/torque_student_distilled_stage0.pt --output artifacts/traction_torque/torque_student_rsl_warmstart_stage0.pt --seed 20260803
```

PPO medium 与最终导出：

```bash
/home/mosense/miniconda3/envs/isaaclab-v2/bin/python scripts/traction_torque/train_torque_traction_teacher.py --num_envs 128 --max_iterations 100 --seed 20260803 --partial_checkpoint model/rl/model_49999.pt --run_name torque_teacher_medium_seed20260803 --headless
/home/mosense/miniconda3/envs/isaaclab-v2/bin/python scripts/traction_torque/train_torque_traction_student.py --num_envs 128 --max_iterations 100 --seed 20260803 --partial_checkpoint artifacts/traction_torque/torque_student_rsl_warmstart_stage0.pt --run_name torque_student_distilled_medium_seed20260803 --headless
/home/mosense/miniconda3/envs/isaaclab-v2/bin/python scripts/traction_torque/export_torque_traction_policy.py --rsl_checkpoint logs/rsl_rl/g1_29dof_torque_traction_student/2026-08-03_10-54-10_torque_student_distilled_medium_seed20260803/model_99.pt --seed 20260803 --output_dir artifacts/traction_torque/export_student_distilled_ppo_medium
```

实际 DAgger round-1（该候选最终未选用）：

```bash
for seed in 20260803 20260804 20260805; do
  /home/mosense/miniconda3/envs/isaaclab-v2/bin/python scripts/traction_torque/collect_torque_force_dataset.py --num_envs 8 --steps 300 --warmup_steps 50 --seed "$seed" --randomization_stage 5 --checkpoint logs/rsl_rl/g1_29dof_torque_traction_student/2026-08-03_10-54-10_torque_student_distilled_medium_seed20260803/model_99.pt --headless --output "artifacts/traction_torque/dagger_round1_student_seed${seed}.npz"
done
/home/mosense/miniconda3/envs/isaaclab-v2/bin/python scripts/traction_torque/aggregate_torque_traction_dagger.py artifacts/traction_torque/dagger_round1_student_seed2026080{3,4,5}.npz --output artifacts/traction_torque/dagger_round1_aggregate.npz
/home/mosense/miniconda3/envs/isaaclab-v2/bin/python scripts/traction_torque/distill_torque_traction_student.py artifacts/traction_torque/dagger_round1_aggregate.npz --teacher_checkpoint logs/rsl_rl/g1_29dof_torque_traction_teacher/2026-08-03_10-44-36_torque_teacher_medium_seed20260803/model_99.pt --teacher_epochs 0 --student_epochs 200 --seed 20260803 --output artifacts/traction_torque/torque_student_dagger_round1_bounded.pt
/home/mosense/miniconda3/envs/isaaclab-v2/bin/python scripts/traction_torque/build_torque_student_rsl_warmstart.py --template logs/rsl_rl/g1_29dof_torque_traction_student/2026-08-03_11-09-26/model_0.pt --baseline model/rl/model_49999.pt --distilled artifacts/traction_torque/torque_student_dagger_round1_bounded.pt --output artifacts/traction_torque/torque_student_rsl_dagger_round1_bounded_warmstart.pt --seed 20260803
/home/mosense/miniconda3/envs/isaaclab-v2/bin/python scripts/traction_torque/train_torque_traction_student.py --num_envs 128 --max_iterations 100 --seed 20260803 --partial_checkpoint artifacts/traction_torque/torque_student_rsl_dagger_round1_bounded_warmstart.pt --run_name torque_student_dagger_round1_bounded_medium_seed20260803 --headless
/home/mosense/miniconda3/envs/isaaclab-v2/bin/python scripts/traction_torque/export_torque_traction_policy.py --rsl_checkpoint logs/rsl_rl/g1_29dof_torque_traction_student/2026-08-03_11-27-38_torque_student_dagger_round1_bounded_medium_seed20260803/model_99.pt --seed 20260803 --output_dir artifacts/traction_torque/export_student_dagger_round1_bounded_ppo_medium
```

Isaac 三种子评测：

```bash
for seed in 20260803 20260804 20260805; do
  /home/mosense/miniconda3/envs/isaaclab-v2/bin/python scripts/traction_torque/collect_torque_force_dataset.py --num_envs 16 --steps 300 --warmup_steps 50 --seed "$seed" --randomization_stage 5 --checkpoint logs/rsl_rl/g1_29dof_torque_traction_student/2026-08-03_10-54-10_torque_student_distilled_medium_seed20260803/model_99.pt --headless --output "artifacts/traction_torque/student_distilled_medium_eval_seed${seed}.npz"
done
/home/mosense/miniconda3/envs/isaaclab-v2/bin/python scripts/traction_torque/evaluate_isaac_torque_rollout.py artifacts/traction_torque/student_distilled_medium_eval_seed2026080{3,4,5}.npz --output artifacts/traction_torque/isaac_student_distilled_medium_evaluation.json
```

MuJoCo fixed-policy matrix：

```bash
cd /home/mosense/guo/unitree_mujoco
PYTHONPATH=/home/mosense/guo/unitree_rl_lab/source/unitree_rl_lab /home/mosense/miniconda3/envs/isaaclab-v2/bin/python simulate_python/run_torque_traction_matrix.py --policy /home/mosense/guo/unitree_rl_lab/artifacts/traction_torque/export_student_distilled_ppo_medium/torque_traction_student.ts --duration_s 4 --seed 20260803 --output_dir artifacts/traction_torque/matrix_student_distilled_ppo_medium --scenarios high_friction low_friction abrupt_friction_drop asymmetric_friction combined_randomization
```

## 23. 一键 smoke 复现

脚本在目标输出目录存在时拒绝覆盖：

```bash
cd /home/mosense/guo/unitree_rl_lab
./scripts/traction_torque/reproduce_torque_traction_smoke.sh
```

它依次运行 30 tests、Teacher PPO smoke、Isaac dataset（同时记录隔离的 Student/Teacher history）、analytical evaluation、temporal correction、DAgger aggregation、PPO Teacher action/latent distillation、RSL template 生成、distilled/baseline checkpoint 合并、Student PPO smoke、最终 RSL export 和 MuJoCo matrix。默认输出到独立 `artifacts/traction_torque/reproduction_seed20260803`。

该脚本在 DAgger 集成后已实际从头运行至 exit code 0，最新验证目录为 `artifacts/traction_torque/reproduction_dagger_v4_seed20260803/`，其中 MuJoCo manifest 记录 5/5 scenarios 完成。脚本显式禁用与任务无关的 pytest third-party plugin autoload，并在切换 MuJoCo 工作目录前将输出目录规范化为绝对路径。

最终 dependency audit 还移除了 torque task 对用户未提交 `velocity_raw_foot_env_cfg.py` 和 `mdp/foot_sensor.py` 的运行依赖。Teacher/label 所需的两个 ground-filtered ContactSensor 现在定义在已跟踪的 torque scene 中，并在 docstring 中明确为 truth-only；Student 的 1,875-D policy term 不读取传感器 tensor。重构后用 2 env × 2 steps live Isaac smoke 验证 history shape `(2,2,15,125)`、truth shape `(2,2,6)`、estimated-force nonfinite count 0。

## 24. 已知限制与进入真机前剩余工作

1. analytical-only Fz 和 swing false force 明显偏大；temporal correction 只在 Stage-0 split 上改善，必须做 Stage-1–5、多 terrain、多 seed 训练和独立 test split。
2. 当前 slip auxiliary label 在 Isaac 离线训练中使用 ankle rigid-body planar-speed proxy，不是 contact-point slip 真值；需要更严格的 simulated contact-point label 重新蒸馏。
3. PPO Teacher 已运行 307,200 transitions，但仍是 medium candidate，没有完成多 stage 收敛和 independent upper-bound evaluation。
4. checkpoint bridge 和一轮 PPO-Teacher DAgger 已完成；bounded distilled residual 进入 RSL rollout 后又完成 307,200-transition PPO。stateful governor 仍未放入 Isaac rollout loop，因此 governor joint fine-tuning 尚未执行。
5. selected candidate 的名义 μ=0.1 单 seed 通过 4 s，但 Stage-5 μ=0.1 三种子全部失败、friction drop 也在 3 s 跌倒；DAgger round-1 虽在自身 Stage-5 governor/no-governor 对照中提高平均 survival，却使 nominal low-μ 退化到 2.02 s，已被 selection gate 拒绝。在解决前不可进入真机 locomotion trial。
6. MuJoCo Stage-5 对 torque/IMU/estimator dynamics 进行 observation/model mismatch；尚需独立 physical mass/inertia XML perturbation matrix。
7. Isaac 有 3-seed 6-s/env，MuJoCo Stage-5 low-μ 有 3-seed，仍属于短 horizon；未执行足够样本的 fall-rate confidence interval、recovery distribution、A8–A15 或 GRU/TCN/历史长度完整消融。
8. G1 真机 `tau_est` 的符号、饱和、量化、延迟、gear efficiency、温漂和 IMU linear acceleration 质量均未测量；所有 randomization ranges 仍为 provisional。
9. 真机前必须做无执行器输出的 bag/log recorder、static double support、known horizontal push、single-support harness、timestamp/dropout 和 emergency-stop dry-run，并据此标定 torque bias、delay 和 contact thresholds。
10. 导出 runtime 尚未接入现有 G1 C++ FSM；本任务明确未修改/启动真机控制。

因此当前交付状态是：**motor-torque traction 软件架构、仿真训练/DAgger 入口、medium 候选模型、导出和固定策略 Sim2Sim 管线均已具备；但策略尚未达到“可等待真机联调”的低摩擦性能门槛。** 下一阶段应先在仿真中完成正式多-stage Teacher、更多带 model-selection gate 的 Student DAgger、governor joint fine-tuning 和更长时多-seed MuJoCo gate，随后才进行只读真机信号采集。
