# 足底传感摩擦自适应：系统全链路梳理（论文 System Overview 素材）

> 依据仓库当前真实状态整理（2026-08-15）：
> `guo/unitree_rl_lab`（Isaac 训练/评估/部署导出）、`guo_1/vola_sensor`（真实
> 传感与标定）、`guo_1/ble_sensor`（BLE 采集/桥接）。本文只梳理流程，不含
> 真机操作指引。

## 1. 一句话定位

G1 双足采用**柔性磁感知鞋垫**（每脚 15 个 Hall 点，只输出原始磁场 Bx/By/Bz），
通过 BLE 送入 1864 维观测的行走策略；策略不接收任何力/接触/μ 的显式估计，
只从 0.3 s 的 Hall 历史 + 采样健康元数据 + 本体 proprio 推断摩擦变化并切换
步态。分层课程训练出"Hall gate（低摩擦检测）+ capture residual（缩步快频）+
stability residual（回高摩擦的航向/横向收敛）"的组合策略。

## 2. 硬件与感知前端（真机）

- **鞋垫**：左右各 15 个 Hall 点，前掌 P00–P04 / 中足 P05–P09 / 后跟 P10–P14，
  每组十字布局"上/左/中/右/下"；每点输出原始 `Bx,By,Bz` 计数 + 温度。
- **链路**：每脚一帧 125 bytes BLE（帧头 `0x7D/F0/02`，15×`>hhhh` 大端），
  ~100 Hz；左右脚独立 BLE adapter（hci0/hci1），唯一名称 + 地址双重校验。
- **测量边界（重要且固定）**：真实传感器**不输出**法向力/切向力/接触/摩擦。
  预处理只做空载基线差分、温漂补偿、通道归一化、低通滤波，明确禁止把磁场
  预处理命名为"力"。加载/倾斜/剪切/滑移仅是离线实验标签。
- **标定与桥接**：引导式采集（基线、温漂、法向、剪切、倾斜、重心转移、行走、
  可控滑移）→ `calibrate_dual_magnetic.py` 生成左右归一化 → F0M1 实时桥
  （`BLE=1/1` + fresh 健康 JSON 才允许吊架测试）。
- **布局一致性**：`config/sensor_layout_a4_15.json` 与 Isaac 唯一配置表必须跨
  文件一致；PCB 最终装配后还需单点磁场扰动验证通道顺序与 XYZ 符号（当前为
  暂定直通映射）。

## 3. 仿真对偶（Isaac Lab）

- `HallFootSensor`：Scheme A 近似的局部 Kelvin–Voigt 柔度 + 磁偶极前向模型
  （另有 deformable 实现），把每步接触片分布到 15 个磁点，产出与真实同构的
  `Bx/By/Bz` 时间序列。
- `contact_distribution_mode`：`aggregate`（默认，兼容旧任务）vs `detailed`
  （R5 用，按 15 点分布原始法向/摩擦接触）。
- **域随机化**：法向/剪切刚度、阻尼、接触扩散、磁矩、磁体位置抖动、增益、
  跨轴串扰、采样延迟/丢包/健康位——对齐真实 BLE 的时序与噪声。
- 真实侧以 `真实足底传感器数据格式.md` 为字节级规范，仿真观测 schema 与其
  对齐（左右顺序 `[left,right] → P00..P14 → Bx,By,Bz`）。

## 4. 观测 ABI（部署接口 1864-D）

| 块 | 维度 | 说明 |
|---|---:|---|
| proprio 历史 | 480 | base_ang_vel 15 + gravity 15 + cmd 15 + q_rel 145 + dq_rel 145 + last_action 145（5 帧） |
| Hall 历史 | 1350 | 15 帧 × 2 脚 × 15 点 × 3 轴 |
| 采样周期 | 30 | 15 帧 × 2 脚，实测采样周期 |
| 有效位 | 2 | 左右脚 fresh/stale |
| motion feedback | 2 | `[body_vy, relative_heading]`（部署由估计器/C++ IMU 锁存提供） |
| **合计** | **1864** | 不含力/接触/滑移/μ/阶段/风险标签 |

- 特权 570-D critic：接触、法向/切向力、摩擦比、滑移 proxy、负载比、ground_μ、
  健康位等只进 critic/奖励/训练期 gate 监督，永不进 actor。
- 健康保护：gate 输出 × 双脚有效位；任一足 stale>0.25 s 该脚 Hall 置零
  （C++ fail-closed）。

## 5. 训练配方链（分层课程）

1. **基座**：官方 model_49999（480-D，平坦地面），作为后续 partial warm-start。
2. **Teacher**：μ 三档 25/50/25%（(0.05,0.25)/(0.25,0.75)/(0.75,1.20)）+
   特权 μ 教师，产 641-D Teacher 标签流。
3. **Hall student（1864-D）**：FastBase 结构 = 冻结 speedboost112 名义步态
   teacher + Hall gate + capture residual；gate 用 privileged LOW/HIGH stage
   BCE 训练（训练期特权，不进 actor）。
4. **cadence_stride**（固定 0.8 m/s、H→L→H 物理地板、无预置步态）：让 PPO 在
   "速度≈步频×步长"下自由学低摩擦缩步快频——这是 R5 低摩擦步态的真正来源。
5. **retention → transition-retention**：加宽地板、LOW/HIGH_END 速度型 heading
   注入扰动，训练零初始化的 stability residual（proprio + motion feedback），
   学"进 LOW 的航向保持 + 回 HIGH 的 vy/heading 收敛"。
6. **R5**：低 μ 再平衡课程（35% 精确 0.28 + 45% U(0.14,0.28) + 20% U(0.10,0.14)），
   冻结 capture 分支，只训 stability；得到当前 1864-D 候选策略。
7. **对照 ablation（本实验）**：同样 R5 课程/reward/特权 critic，仅把 actor 换
   成纯 480-D proprio（无足底通道）→ 证明 Hall 的价值在"接触级预判 + 提前换
   步态"，proprio 只能"打滑后反应"。

## 6. 评估协议与门禁

- 常数 μ 矩阵（0.8/0.28/0.20/0.10 @ 0.3 m/s，3 seeds × 16 envs × 40 s）；
- 时间型 H→L→H（0.8→0.10→0.8）与位置型 LongDemo 课程（0.8 m/s，70 s）；
- 指标：动力学摔倒率、存活/完成、vx、低摩擦段速度、接触点滑移、步频/步长、
  heading/vy RMS、stance 占比、逐关节动作差；
- 门禁 G1–G5：低摩擦缩步保持、速度恢复、heading、横向漂移、动力学摔倒；
  μ=0.10 极端工况单独列为残余风险（当前规格 μ_min=0.2 不触发）。

## 7. 部署（C++，仿真已验收、真机按主 agent 节奏进行）

- `policy.onnx`（1864→29），`joint_target = default_pos + 0.25·action`；
- F0M1 桥健康 fail-closed、lateral velocity estimator ONNX、
  relative heading（C++ 对 IMU yaw 锁存）、PI heading-hold（仿真已验证，
  部署侧待接入）、可一键回退官方策略；
- 本梳理不含任何真机操作，仅描述已固化接口。

## 8. 论文可提取的贡献点

1. 柔性磁足底（纯磁场原始量）的 sim2real 观测/健康 ABI 与域随机化对齐；
2. "Hall gate + capture residual + stability residual"的组合结构：低摩擦
   缩步快频、机动保持、回高摩擦收敛三项能力分而治之；
3. 分层课程（Teacher → cadence/stride → transition-retention → R5）；
4. 干净 ablation：1864-D Hall vs 480-D proprio 在**同一课程同一协议**下的
   "预判 vs 反应"机理证据（减速提前量、滑移峰值、摔倒率、逐关节动作差）。
