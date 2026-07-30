# G1 牵引自适应模型选择报告（2026-07-29）

## 最终选择

### Oracle Teacher

- 检查点：
  `<repo>/logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_teacher_motion_balanced_symmetry/2026-07-29_13-44-07_motion_balanced_symmetry_prod_20260729/model_8900.pt`
- SHA-256：
  `9004d60e306798d9377546a6f5f3de58dfd25b1d5ad4a57067d89f0ea112891d`
- 结构：`641 → 512 → 256 → 128 → 29`
- 特权输入：第 641 维真实有效摩擦系数，仅用于 Oracle 训练与诊断。
- 运动反馈：最近五帧 `[body_vy, relative_heading]`。
- 训练方法：从 `model_8750.pt` 热启动，使用左右镜像数据增强、镜像动作损失和 PPO 跟速/直行奖励联合微调。

### 当前最佳无特权闭环

- 1864 维复合策略：
  `<repo>/logs/evaluations/traction_magnetic_motion/20260729_model8900_estguided_motion_mu120/shared_magnetic_policy.pt`
- SHA-256：
  `823c4758c6b499b4df6bb155d547515c91c649e3f2f0fbdfee402ef522284a9f`
- 部署槽：
  `traction_magnetic_motion_8900_mu120`
- 摩擦估计器：
  `<repo>/logs/evaluations/traction_magnetic_speed_lateral/20260729_recovery/friction_estimator_1864/friction_estimator.pt`
- 横向速度估计器：
  `<repo>/logs/evaluations/traction_magnetic_motion/20260729_lateral_velocity_estimator/lateral_velocity_estimator.onnx`
- 输入结构：`480 proprio + 1350 Hall + 30 sample-period + 2 valid + body_vy_hat + relative_heading = 1864`。
- 推理时不使用真实摩擦系数，也不使用真实横向速度。
- 新槽位未写入当前真机 FSM 配置，未激活。

## 核心结果

### Oracle Teacher：MuJoCo 5 摩擦 × 2 速度

完整报告：
`<repo>/logs/evaluations/mujoco_traction_teacher_motion/20260729_model8900_balancedsym_oracle_full/summary.md`

| 指标 | 结果 |
|---|---:|
| 稳定单元格 | 10 / 10 |
| 摔倒 | 0 |
| `mu=0.08, v_cmd=1.0` 实际速度 | 0.157 m/s |
| `mu=1.20, v_cmd=1.0` 实际速度 | 0.907 m/s |
| 高摩擦横向速度 | 0.101 m/s |
| 高摩擦横向漂移（该次完整矩阵） | 0.501 m |

单点重复试验中，`model_8900` 的高摩擦横向漂移为 0.342 m；原
`model_8750` 完整矩阵为 0.600 m。漂移存在运行间波动，但新 Teacher
保持了前向速度并呈现更低的跨仿真侧移。

### 无特权复合策略：MuJoCo 5 摩擦 × 2 速度

完整报告：
`<repo>/logs/evaluations/mujoco_traction_magnetic_motion/20260729_model8900_mu120_estimated_motion_full/summary.md`

| 指标 | 结果 |
|---|---:|
| 稳定单元格 | 10 / 10 |
| 摔倒 | 0 |
| `mu=0.08, v_cmd=1.0` 实际速度 | 0.141 m/s |
| `mu=1.20, v_cmd=1.0` 实际速度 | 0.826 m/s |
| 高摩擦横向速度 | 0.107 m/s |
| 高摩擦横向漂移 | 0.594 m |

该结果满足当前行为门槛：低摩擦主动降速、高摩擦速度不低于
0.8 m/s、MuJoCo 全矩阵零摔倒。

### Isaac 多种子

- `model_8900` 在两个未见种子的默认部署范围（不超过 1.0 m/s）
  全部零摔倒。
- 高摩擦 `1.0 m/s` 的两种子平均速度为 0.934 m/s。
- `1.5 m/s` 仅作为压力测试，合计出现 1 / 64 次摔倒，与
  `model_8750` 基线相同。
- 无特权复合策略在额外 seed 42 的高摩擦矩阵中出现 1 / 64 次摔倒；
  因此尚未满足“进入真机前全随机化零摔倒”的最终门槛。

## 淘汰分支

- `model_8929.pt`：高摩擦 MuJoCo 漂移 0.665 m，且 1.5 m/s
  压力测试摔倒增加，淘汰。
- `model_8989.pt`：对称损失更低，但高摩擦 MuJoCo 漂移回升到
  0.428 m，劣于中期 `model_8900`。
- 多种子摩擦估计器 R1：发生闭环“低摩擦自锁”，高摩擦仅
  0.222 m/s，淘汰。
- DAgger R2：恢复高摩擦速度，但横向漂移 0.865 m；0.9 标定后
  速度 0.804 m/s、漂移 0.799 m，均未优于当前复合策略。

## 当前限制与下一步

1. 当前最佳可部署复合策略仍是“估计器 + 冻结 Teacher”，不是最终单网络 Student。
2. 连续摩擦回归在未见 Isaac 种子上的 MAE 仍高于 0.15；极端摩擦分类已较可靠，但中间摩擦存在命令相关偏差。
3. 下一阶段应以 `model_8900` 为 Teacher，收集多种子、多命令及失效恢复轨迹，蒸馏为共享每足 32 维编码器的 `1864 → 548 → 29` Student。
4. Student 训练目标应同时包含动作蒸馏、Teacher latent 蒸馏、摩擦区间分类、速度单调性和左右镜像一致性。
5. Student 必须重新通过 Isaac 多种子、MuJoCo 5×2、传感器掉线/延迟和跌倒恢复门槛后，才能进入系绳、急停、低速开始的分阶段真机验证。
