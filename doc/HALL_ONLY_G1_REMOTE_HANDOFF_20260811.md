# Hall-only G1 摩擦自适应行走：远程继续开发交接文档

更新时间：2026-08-11（Asia/Shanghai）  
本地工程：`/home/mosense/guo/unitree_rl_lab`  
当前结论：**短程 H→L→H 已显示有效自适应并优于原始策略的短程安全性；长程高速恢复后的稳定性尚未通过，禁止上真机。**

## 1. 远程 Agent 的任务

继续在现有工程上优化，不要重写项目，也不要改变已经冻结的传感器与部署接口。

最终目标：全部区域保持同一个外部前向速度指令 `0.8 m/s`。G1 在高摩擦保持原有高速能力；进入低摩擦后，仅凭多帧双足 Hall `Bx/By/Bz`、Hall 健康状态和本体感觉历史，自主改变步频、步长、支撑时序，必要时降低实际速度防止滑倒；回到高摩擦后快速且稳定地恢复高速。必须在同条件、多种子 Isaac Sim 中明确超过原始 Unitree `model_49999`，再通过 MuJoCo sim-to-sim。全部通过后才能生成一个**默认不激活**的真机候选。

“Hall-only”的准确含义：

- 摩擦/柔性足底的外感知只使用 Hall 磁场；
- Actor 可以使用关节、IMU、速度命令等正常本体感觉历史；
- Actor 不得读取力、摩擦系数、接触点、滑移真值或地面阶段；
- Hall 数据不转换为力，也不宣称真机能直接测量法向/切向力；
- Isaac 的接触量只允许用于物理仿真、privileged critic、奖励和验收。

## 2. 当前结论与最佳模型

当前最好的短程演示模型：

```text
logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_hall_spatial_cadence_stride/
2026-08-11_17-25-07_cadence_stride_residual_release_canary_seed495_r1/model_55.pt
```

SHA-256：`4ddd2c9bac031e7c6f74bb4874d60e6baf4d36314a67d3c6bab7a54a89a61f92`

它是当前**短程基准和演示模型**，不是长程安全候选，更不是可上真机模型。

当前核心问题已经不是“回到高摩擦后恢复不了速度”，而是：

> 速度能恢复到约 0.8 m/s，但持续高速数秒后会出现航向/横向发散并摔倒。

不能宣称已经达到长程零摔倒，也不能以更大的 checkpoint 编号代表更好。

## 3. 环境与迁移

当前兼容环境：

- Isaac Sim 5.1；
- Isaac Lab 2.3.2；
- RSL-RL 5.0.1；
- Python：`/home/mosense/miniconda3/envs/isaaclab-v2`；
- Isaac 启动：`/home/mosense/IsaacLab/isaaclab.sh`。

当前仓库包含大量未提交修改和未跟踪文件，**不能只重新 clone 干净仓库**。至少同步：

```text
source/unitree_rl_lab/
scripts/rsl_rl/
scripts/traction/
scripts/tests/
doc/
artifacts/hall_speed_demo/
artifacts/hall_cadence_stride/
logs/rsl_rl/*cadence_stride*/
logs/rsl_rl/*cadence_stride_retention*/
model/rl/model_49999.pt
```

关键哈希：

| 文件 | SHA-256 |
|---|---|
| `model/rl/model_49999.pt` | `c508af7910a69e2bc06111caaa677d5bea521bfb52fc654d82d38b499e2ae99b` |
| `artifacts/hall_speed_demo/speedboost112_frozen_teacher.pt` | `f9b62b63b99798a9206f40bc4df6b852d061cd3b6822cc294babf791ba609f86` |
| `model_55.pt` | `4ddd2c9bac031e7c6f74bb4874d60e6baf4d36314a67d3c6bab7a54a89a61f92` |

## 4. 真实足底接口

真实数据规范：

```text
doc/REAL_FOOT_HALL_DATA_FORMAT.md
/home/mosense/guo_1/vola_sensor/真实足底传感器数据格式.md
```

固定约束：左右脚各 15 个 Hall site，编号 `P00..P14`；每个 site 的有效原始信号是 `Bx,By,Bz`，可带相对基线、采样周期、包龄、有效位和温度。左右脚必须独立接收与命名。真机没有法向力/切向力标签。

物理结构：

```text
刚性足底 → PCB 外壳及 PCB/Hall 层 → 嵌有磁片的 TPU 磁化层
```

没有额外中间连接层。每个 Hall 下方对应嵌入 TPU 的四个磁片，默认 2×2。

## 5. 仿真 Hall 实现

主要目录：`source/unitree_rl_lab/unitree_rl_lab/sensors/`

关键文件：

```text
hall_foot_sensor.py
hall_sensor_config.py
magnetic_field_model.py
hall_sensor_noise.py
hall_sensor_visualizer.py
hall_contact_distribution.py
hall_deformable_sole.py
```

当前大规模训练使用 Scheme A：Isaac 计算真实接触，再近似 TPU 的局部压缩/弯曲/剪切，更新每个 Hall 下四个磁片的相对位姿，用磁偶极子叠加得到 `Bx,By,Bz`，最后加入真实传感器噪声、偏置、量化、延迟、掉包、坏通道和整脚掉线。Actor 最终只读取磁场和健康元数据。

磁场接口已解耦：

```text
MagneticFieldModel
├── DipoleMagneticFieldModel
└── CalibratedMagneticFieldModel
```

接触驱动有两种：

- `aggregate`：每脚总力和平均接触点，兼容旧 checkpoint；
- `detailed`：多个 contact patch/friction anchor 分配到 15 个 Hall 区域；CadenceStride 显式使用它。

Scheme B 可变形体适合单足高保真可视化和校准，当前没有用于 512/4096 环境 PPO 主训练，避免 Isaac 5.1 deformable API/吞吐风险。

## 6. Actor 观测 ABI

Actor 输入严格为 `[num_envs,1864]`：

| 索引 | 含义 |
|---|---|
| `0:480` | 5 帧 term-major 本体感觉、命令和动作历史 |
| `480:1830` | 15 帧双足 15-site 三轴 Hall，`[time,left/right,P00..P14,XYZ]` |
| `1830:1860` | 左右脚采样周期历史 |
| `1860:1862` | 左右脚有效状态 |
| `1862:1864` | `body_vy, relative_heading`，不是 sensor age |

命令历史 `30:45` 的正确索引：

```text
vx  = 30,33,36,39,42
vy  = 31,34,37,40,43
yaw = 32,35,38,41,44
```

禁止使用旧 frame-major 索引 `(6,102,198,294,390)`。旧 `future060 sensor_age` 风险模型也禁止使用，因为它误解了 `1862:1864`。Actor 输出 29 维动作，critic 为 570 维 privileged observation。

## 7. 当前策略结构

```text
冻结 speedboost112 高速基座
 + Hall 时空特征 → capture gate → bounded Hall residual
 + 可选 proprio stability residual（Retention 诊断线）
```

- 高速基座永久冻结；
- 任一整脚掉线时 Hall residual 严格归零；
- LOW/HIGH 标签只监督训练 gate，不进入 Actor/ONNX；
- stability residual 只读 `obs[0:480] + obs[1862:1864]`；
- HIGH anchor 已排除 stability branch，避免把稳定修正强制锚成 0；
- gate/capture residual/stability residual 使用互斥参数组和独立梯度裁剪。

核心代码：

```text
source/unitree_rl_lab/unitree_rl_lab/traction/fastbase_capture_residual.py
source/unitree_rl_lab/unitree_rl_lab/traction/anchored_ppo.py
source/unitree_rl_lab/unitree_rl_lab/traction/frozen_speedboost_teacher.py
```

## 8. 新增场景与任务

### 短程 transition-dense 训练

```text
Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SpatialFrictionMediumDenseCadenceStride
```

高/低/高真实静态材质，`mu=.90/.28/.90`；外部命令始终 `.80 m/s`；不规定低摩擦必须降到固定速度，也不规定步频方向；移除了旧低速 cap、固定 gait period 和 touchdown-rate 惩罚。

### 12 m 长可视化

```text
Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SpatialFrictionCadenceStrideLongDemo
```

```text
HighStart [-4,0], mu=.90, 蓝色
Low       [ 0,3], mu=.28, 橙色
HighEnd   [ 3,8], mu=.90, 蓝色
```

reset `[-3.5,-3.1]`，success `x=7.5`，35 s。完整评测至少 1500 policy steps。

### 长尾稳定性训练

```text
Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SpatialFrictionCadenceStrideRetention
```

### 高摩擦扰动课程

```text
Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SpatialFrictionCadenceStrideRecoveryCurriculum
```

只在 HIGH_START/HIGH_END 训练阶段异步注入小速度扰动；LOW 不注入。Play/eval 映射回无扰动 Retention 场景。

代码位置：

```text
source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/__init__.py
source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_foot_env_cfg.py
source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/spatial_friction.py
source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py
source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py
```

## 9. 已完成的评测/语义修正

1. 用 `v_point=v_COM+omega×(p_contact-p_COM)` 计算接触点速度，再投影到接触切平面；
2. 旧 ankle-link origin planar speed 仅保留为 legacy diagnostic；
3. 地面阶段由三块地面的 filtered contact 锁存，腾空时不抖回 HIGH；
4. 到达末端使用 success truncation，避免“跑出地面摔倒”诱导 PPO 慢走；
5. warm-up 跌倒计入；首次跌倒后的 reset 数据不进入主速度统计；
6. 同时报首次跌倒、累计事件、失败环境数、failure-free exposure；
7. 增加 cadence、step/stride length、分区速度和恢复时延；
8. 增加部署可用的累计直行航向误差 reward；
9. 修复 Hall cfg 深复制后不同 observation term 的 DR 配置不同步；
10. 修复 nominal Hall 在 managed reset 后重新随机化的问题。

## 10. 正确性审计状态

已覆盖：1864/570 维度、Hall 历史顺序、左右脚、Motion 尾通道、command term-major 索引、Actor 不含力/μ/contact/slip/stage、frozen teacher 隔离、私有优化器参数组、Hall 失联 fail closed、contact-point slip 公式、reset 和统计污染。

目前没有发现“观测维度搞错”“Hall 切片错位”“左右脚颠倒”或“力进入 Actor”这类低级 bug。stability residual 确实接收到 PPO primary-loss 梯度，并未被 gate/LOW-expert 梯度覆盖。

迁移后仍必须重跑 focused tests，不能假设不同 checkout 自动一致。

## 11. 当前量化结果

### 短程同 seed=497，32 环境，命令恒定 0.8

| 指标 | Hall model55 | 原始 Unitree model49999 |
|---|---:|---:|
| 摔倒 | **0/32** | 4/32 |
| course success | **32/32** | 28/32 |
| HighStart vx | 0.677 | 0.727 |
| Low vx | 0.606 | 0.824 |
| HighEnd vx | 0.734 | 0.840 |
| HighEnd/HighStart vx | **108.4%** | 115.5% |
| Low/HighStart 步频 | **106.1%** | 97.0% |
| Low/HighStart 步长 | **74.7%** | 108.7% |

Hall model55 学出了“低摩擦步频约 +6%、步长约 -25%、实际速度约 -10.6%，回高摩擦后步长和速度恢复”的行为。短程安全优于原始策略，但 HighStart 速度损失约 6.8%，还没达到最终 ≤5% 门槛。

证据：

```text
artifacts/hall_cadence_stride/residual55_nominal_seed497_32env_summary.json
artifacts/hall_cadence_stride/original_unitree_nominal_seed497_32env_summary.json
```

### hardened Hall + HealthEnvelope，seed=496，32 环境

- 0/32 摔倒；
- 27/32 完成 H-L-H；
- 26/32 course success；
- HighEnd `0.731 m/s`；
- 未完成主要来自整脚失联后的健康限速。

文件：`artifacts/hall_cadence_stride/residual55_hardened_health_seed496_32env_summary.json`

### 12 m 长程，seed=498，4 环境

| 指标 | Hall model55 | 原始 Unitree |
|---|---:|---:|
| 摔倒 | 4/4 | 3/4 |
| course success | 0/4 | 1/4 |
| H/L/H vx | .749/.530/.766 | .794/.848/.855 |

长程尚未超过原始策略。

### 最新 Retention r5，seed=500，8 环境

Checkpoint：

```text
logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_hall_spatial_cadence_stride_retention/
2026-08-11_19-06-41_heading_error_recovery_seed510_r5/model_115.pt
```

SHA：`3eb673291d093cfaf13266752aabbcf0853530fde8d684a3e979f77e36d588d2`

| 指标 | 结果 |
|---|---:|
| H/L/H vx | .732/.553/.798 |
| 进入 Low 后 1 s vx | 约 .582 |
| HighEnd 恢复到 ≥.7 | 8/8，平均约 .70 s |
| 最早摔倒 | 7.24 s |
| 最终摔倒 | **7/8** |
| course success | **1/8** |

结论：恢复速度成功，但持续稳定失败。该模型已否决。

## 12. 已否决路线

- Retention r1/r2/r3：提高私有 stability LR 后修正量增大，但仍 7/8 摔倒；
- r4 高摩擦随机扰动课程：model100/110/115 都仍 7/8；
- r5 累计航向误差 reward：仍 7/8；
- `stability_limit .25→1.0`：仍 7/8，证明不是单纯 authority 不足；
- 蒸馏旧 frozen stable branch：离线 loss 下降，但物理仍 7/8；
- 默认/early-heading 外部安全器：只延迟摔倒；
- conservative 安全器：摔倒降至 2/8，但 H/L/H 仅 `.312/.198/.294`，且 0 success，破坏高速目标；
- Stage7 `model6149` 是低摩擦低速捕获专家，不是高速 HighEnd 横向恢复专家，不能直接硬切。

诊断源（不是候选）：

```text
logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_hall_spatial_cadence_stride_retention/
2026-08-11_18-33-25_cadence_retention_stability_lr1e4_seed504_r3/model_90.pt
```

SHA：`8c3a72f34d932ed96c074fe1818b3bcacdb347a289e26cc0840e3c112ac14a02`

## 13. 下一步优先级

### P0：迁移复现，不训练

核对 SHA、任务注册、CPU tests、2-env Isaac smoke；复现 model55 短程和 model90/r5 长程。若偏差明显，先查环境/Hall cfg/seed/材质。

### P1：训练真正成功的高速恢复专家

不要继续用已失败的 stable branch。新恢复任务应从真实 H→L→H 轨迹的 HighEnd 状态初始化，覆盖 `relative_heading`、`body_vy`、roll/pitch rate、非对称足底历史，命令保持 `.8`。训练可用 privileged reward/critic/未来跌倒标签，但 Actor 仍严格 1864。

专家首先必须在物理 rollout 中把固定 seed500 的 7/8 降为 0/8；如果专家本身不成功，禁止蒸馏。

### P2：蒸馏到 bounded stability residual

当且仅当专家物理成功后：冻结 speedboost112；safe HIGH 状态做 action anchor；near-fall 状态做 recovery supervision；显式隔离 PPO 与恢复监督梯度；先固定 seed canary，再 held-out seeds。

### P3：正式 Isaac 门禁

同 task、同 `.8` command、同初态/seed/材质/DR draw，每个 cell 新进程；至少 10 seeds × 32/64 env；短 H-L-H、12 m 长 H-L-H；healthy、dead-channel、delay、single/both-foot offline 分开报告。

### P4：MuJoCo 与真机

Isaac 全过后才做同一 1864 schema 的 MuJoCo sim-to-sim、真实 BLE 时序/归一化/基线/stale watchdog 预检，并只生成未激活候选。真机从吊架和低速开始。

## 14. 最终验收门槛

- nominal 多种子 0 fall、无 NaN；
- 长程 HighEnd 恢复高速后仍持续稳定；
- HighStart 相对原始速度损失 ≤5%；
- HighEnd 速度和步长恢复到自身 HighStart 的 ≥90%；
- 恢复目标 ≤1.5 s；
- corrected contact-point slip、姿态、横漂、冲击、action slew 不劣于基线；
- 行为来自 Hall+本体历史，不偷看 μ/contact/force/stage；
- 以 first-fall survival、failure-free distance/exposure、course success 为主；
- post-reset 样本不得美化主统计。

## 15. 复现命令

远程端替换 Python 和 checkpoint 绝对路径。

### model55 短程

```bash
cd /path/to/unitree_rl_lab
TERM=xterm PYTHONPATH=source/unitree_rl_lab /path/to/isaaclab-v2/bin/python3 \
  scripts/rsl_rl/eval_spatial_friction_course.py --headless \
  --task Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SpatialFrictionMediumDenseCadenceStride \
  --checkpoint /absolute/path/to/model_55.pt \
  --num_envs 32 --steps 400 --seed 497 --command 0.8 \
  --summary_json artifacts/hall_cadence_stride/remote_model55_seed497_summary.json \
  --trace_npz artifacts/hall_cadence_stride/remote_model55_seed497_trace.npz
```

### model55 12 m 长程

```bash
TERM=xterm PYTHONPATH=source/unitree_rl_lab /path/to/isaaclab-v2/bin/python3 \
  scripts/rsl_rl/eval_spatial_friction_course.py --headless \
  --task Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SpatialFrictionCadenceStrideLongDemo \
  --checkpoint /absolute/path/to/model_55.pt \
  --num_envs 4 --steps 1500 --seed 498 --command 0.8 \
  --summary_json artifacts/hall_cadence_stride/remote_model55_long_seed498_summary.json \
  --trace_npz artifacts/hall_cadence_stride/remote_model55_long_seed498_trace.npz
```

### Retention checkpoint 筛选

```bash
TERM=xterm PYTHONPATH=source/unitree_rl_lab /path/to/isaaclab-v2/bin/python3 \
  scripts/rsl_rl/eval_spatial_friction_course.py --headless \
  --task Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SpatialFrictionCadenceStrideRetention \
  --checkpoint /absolute/path/to/retention_model.pt \
  --num_envs 8 --steps 1000 --seed 500 --command 0.8 --skip_label_probe \
  --summary_json artifacts/hall_cadence_stride/remote_retention_seed500.json \
  --trace_npz artifacts/hall_cadence_stride/remote_retention_seed500.npz
```

### GUI/录像

```bash
TERM=xterm PYTHONPATH=source/unitree_rl_lab /path/to/isaaclab-v2/bin/python3 \
  scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SpatialFrictionCadenceStrideLongDemo \
  --checkpoint /absolute/path/to/model_55.pt --num_envs 1 --device cuda:0 \
  --viewer_eye 8.5 -14.0 8.0 --viewer_lookat 2.0 0.0 0.3
```

无 GUI 录像时增加：`--headless --video --video_length 1500`。

## 16. 演示与证据文件

```text
artifacts/hall_cadence_stride/isaac_hall_hlh_cadence_stride_demo.mp4
artifacts/hall_cadence_stride/isaac_demo_model55_seed497_clean/
artifacts/hall_cadence_stride/residual55_nominal_seed497_32env_summary.json
artifacts/hall_cadence_stride/original_unitree_nominal_seed497_32env_summary.json
artifacts/hall_cadence_stride/residual55_long_nominal_seed498_4env_summary.json
artifacts/hall_cadence_stride/original_unitree_long_nominal_seed498_4env_summary.json
artifacts/hall_cadence_stride/retention_model90_lr1e4_seed500.json
artifacts/hall_cadence_stride/heading_reward_r5_model115_seed500.json
artifacts/hall_cadence_stride/recovery_direction_model90_seed500.npz
artifacts/hall_cadence_stride/diagnostic_scaled_stability/
artifacts/hall_cadence_stride/diagnostic_stable_teacher/
```

## 17. 迁移后测试

```bash
cd /path/to/unitree_rl_lab
export PYTHONPATH=source/unitree_rl_lab
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
/path/to/isaaclab-v2/bin/python3 -m pytest -q \
  scripts/tests/test_hall_foot_sensor.py \
  scripts/tests/test_hall_cfg_sync.py \
  scripts/tests/test_hall_contact_distribution.py \
  scripts/tests/test_fastbase_capture_residual.py \
  scripts/tests/test_high_friction_anchored_ppo.py \
  scripts/tests/test_hall_cadence_stride_course.py \
  scripts/tests/test_high_speed_stability_envelope.py \
  scripts/tests/test_hall_spatial_friction_eval.py \
  scripts/tests/test_spatial_friction_state.py
```

再做 2-env Isaac smoke，最后才跑 512-env。不要迁移后立刻启动长训练。

## 18. 禁止事项

1. 不把 Hall 转成法向/切向力作为 Actor 输入；
2. 不把 Isaac contact force 加进 Actor；
3. 不使用旧 `future060 sensor_age` 风险模型；
4. 不改 1864 schema 后继续加载旧 checkpoint；
5. 不让部署 gate 读取 stage、μ 或 contact；
6. 不用平均速度/reset 后样本掩盖跌倒；
7. 不继续 r4/r5、scaled-stability 或 old-stable-teacher 路线；
8. 不把 conservative 低速结果包装成成功；
9. 不自动覆盖或激活真机 `v0` 策略；
10. 不执行破坏性 git reset/clean。

## 19. 一句话交接

> 已证明固定 0.8 m/s 指令下，Hall+本体历史能让 G1 在短 H→L→H 中学出低摩擦步频上升、步长缩短并回高摩擦恢复速度的行为，短程安全优于原始 Unitree；但 12 m 长程会在恢复高速后航向/横向发散。下一阶段必须先训练并物理验证一个真正能处理 HighEnd 高速失稳的恢复专家，再蒸馏到只用 1864 可部署观测的 bounded residual，随后完成多种子 Isaac→MuJoCo→未激活真机候选门禁。
