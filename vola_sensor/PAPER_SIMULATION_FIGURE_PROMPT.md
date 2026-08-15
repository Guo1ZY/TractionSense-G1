# 论文机制图 Prompt：柔性磁感知足底在 Isaac Sim 中的仿真与策略流程

## 图的核心结论

地面接触只在仿真内部驱动 TPU 磁化层的局部压缩、弯曲和剪切；嵌入 TPU 的四磁片相对 Hall 元件发生位姿变化，经磁偶极子叠加得到每个 Hall 点的 `Bx/By/Bz`。策略只读取多帧磁场、采样健康状态和允许的本体感觉历史，不把磁场转换成法向力或切向力，并通过步频、步长和必要时速度变化实现高–低–高摩擦路面的自适应行走。

## 推荐版式

- 类型：论文方法机制图，schematic-led composite；
- 画幅：横向 16:9，双栏宽度，白色背景，2K 或更高；
- 布局：四个面板 `a–d`，左侧结构、中央物理机制、下方计算链、右侧行走策略；
- 风格：Nature/Science Robotics 风格、扁平矢量感、工程剖面、线条清楚、颜色克制；
- 本图是概念/方法示意图，不显示虚构实验曲线、准确率或数值结果。

## 可直接用于绘图模型的完整 Prompt

```text
Create a clean publication-quality scientific mechanism schematic for a robotics paper, titled conceptually “Hall-only flexible magnetic sole simulation and friction-adaptive humanoid locomotion”. Use a wide 16:9 double-column composition, white background, flat vector-like technical illustration, restrained Nature/Science Robotics color palette, crisp thin outlines, consistent perspective, minimal short English labels, no decorative effects, no logos, no fabricated quantitative results.

Arrange four clearly separated but visually connected panels labeled a, b, c, d.

Panel a — Physical sole architecture and 15-site layout:
Show an exploded and cross-sectional view of one humanoid foot sole. The physical stack from top to bottom must be exactly: rigid robot sole; PCB housing containing the PCB and Hall chips; bottom flexible magnetized TPU layer. There is NO intermediate connector layer. Use opaque colors: rigid sole dark blue-gray, PCB housing teal-gray, Hall chips red, TPU layer warm orange. Inside the TPU layer, directly beneath every Hall sensor, embed four small circular disk magnets in a symmetric 2-by-2 square arrangement. Magnets are embedded inside the TPU, not attached below it and not floating. Add a top-view footprint inset with exactly 15 Hall sites in the measured nonuniform arrangement: three longitudinal sensing regions at forefoot, midfoot, and heel; each region contains five sites arranged as a cross, one center plus top, bottom, left, and right. Number consistently from toe to heel, and within each region in a fixed order. Indicate that the right foot mirrors the left foot while maintaining local-axis conventions.

Panel b — Local deformation-to-magnetic-field mechanism, the main hero panel:
Show a zoomed paired comparison: unloaded state on the left and ground-loaded state on the right. A Hall chip is fixed in the rigid PCB housing above four disk magnets embedded in orange TPU. In the unloaded state, the four magnets form a regular 2-by-2 pattern beneath the Hall sampling point. In the loaded state, a gray ground patch contacts the TPU and causes local compression, bending, and shear; visibly change the magnets’ relative positions and small orientations while keeping them embedded in the TPU. Draw thin green magnetic-field vector arrows from the four magnets toward the Hall point and a small local xyz coordinate triad at the Hall chip. Show vector superposition producing only Bx, By, Bz and delta-B relative to the unloaded baseline. Use one dashed gray arrow labeled “simulation-only contact mechanics” pointing to TPU deformation. Do not draw a force sensor and do not draw force as a Hall output. Add the short label “dipole-field superposition” without long equations; leave clean space for the exact dipole equation to be added later as editable vector text.

Panel c — Isaac Sim computation pipeline:
Draw a concise left-to-right computational flow. Start with the G1 rigid foot pose and a coplanar ground-contact patch. Next show a detailed local contact distribution feeding a stable Scheme-A local-compliance model for massively parallel Isaac Lab training: local normal compression, bending, and tangential shear states are estimated at 15 sensing regions without creating 60 expensive independent magnet rigid bodies. Then show four virtual magnet poses updated for each Hall site. Next show a replaceable magnetic-field block labeled “MagneticFieldModel”, with a highlighted “Dipole model” and a secondary future branch “Calibrated / lookup model”. Then rotate and sum the four magnet contributions in the Hall local frame. End with a tensor icon labeled [num_envs, 2 feet, 15 sites, 3 axes] and symbols Bx, By, Bz, dBx, dBy, dBz. Use solid blue arrows for deployable Hall data and dashed gray arrows for simulation-only contact/deformation variables. Explicitly place a red crossed-out symbol over “force conversion”: no B-to-force conversion.

Panel d — Hall-conditioned friction-adaptive locomotion:
Show a Unitree G1 humanoid walking from left to right across a long, wide, opaque three-section flat course: blue high-friction ground, yellow low-friction ground, then blue high-friction ground again. Keep all patch tops coplanar and make each colored section visibly long and wide. Above the path show the same requested forward command, vx command = 0.8 m/s, unchanged across all sections. Below the robot show a temporal observation strip with 15 frames of both feet’s 15-site Bx, By, Bz signals, plus compact proprioceptive history and Hall health/validity. Feed these into a frozen fast locomotion base plus Hall risk/capture gate plus bounded residual policy, then to 29 joint actions. On the first high-friction section show fast confident steps; on low friction show adaptive shorter step length and higher or adjusted cadence, with optional moderate speed reduction only when necessary; on the final high-friction section show rapid recovery of the original fast gait. Emphasize that adaptation is not defined as mandatory slowing down. Add a sensor-fault branch: if one foot loses Hall data, confidence becomes zero, Hall residual closes, and a conservative safety envelope is activated.

Scientific signal-boundary rule across the whole figure:
The policy input path contains only multi-frame Hall Bx/By/Bz, Hall timing/validity, velocity command, and deployable proprioception. Ground friction coefficient, contact force, pressure, true slip, terrain labels, and deformation states may appear only as dashed gray simulation-internal or critic/reward-side information; none may enter the actor observation. Do not imply that Hall measurements directly equal normal force, tangential force, pressure, or coefficient of friction.

Visual encoding:
Hall sensors red; embedded magnets dark purple or black; TPU orange; rigid sole blue-gray; PCB housing teal-gray; magnetic vectors green; high-friction terrain blue; low-friction terrain yellow; deployable data arrows solid blue; privileged simulation-only arrows dashed gray. Use opaque surfaces, slim vector arrows, no transparency that obscures layers, no photorealism, no dramatic shadows, no cartoon style. Make panel b visually dominant, with panels a, c, d supporting the causal chain.
```

## Negative Prompt / 必须排除的错误

```text
No Hall-to-force conversion, no normal-force output, no tangential-force output, no pressure-map output, no friction-coefficient sensor, no ordinary Isaac magnetometer, no force sensor icon in the actor path. No intermediate connector layer between PCB housing and TPU. No magnets floating outside the TPU. No single magnet per Hall site. No magnets above the Hall chip. No transparent TPU that makes the layer structure ambiguous. No uniformly spaced 15-point grid; use three nonuniform cross-shaped groups of five. No extra sensors, no missing sensors, no incorrect left/right mirroring. No ground-force or friction truth entering the policy actor. No fake plots, fake accuracy values, fake reward curves, p-values, logos, journal marks, or institution names. No excessive text, no misspelled labels, no dark background, no glossy 3D advertising render, no cartoon humanoid, no exploding parts, no non-coplanar ground steps.
```

## 建议后期用矢量软件添加的短标签

AI 绘图模型容易拼错文字，建议生成底图后在 Illustrator/Inkscape/PowerPoint 中重新添加以下标签：

```text
a  Magnetic sole architecture
Rigid sole
PCB housing + Hall sensors
Magnetized TPU layer
15 Hall sites per foot
4 embedded magnets / Hall site

b  Deformation–field coupling
Compression
Bending
Shear
Dipole-field superposition
Hall local frame
Bx, By, Bz
ΔB from baseline

c  Isaac Sim + custom field computation
Detailed contact distribution
Scheme-A local compliance
Virtual magnet poses
MagneticFieldModel
Dipole model
Calibrated / lookup model
[Nenv, 2, 15, 3]
No B-to-force conversion

d  Hall-conditioned locomotion
High friction
Low friction
High friction
Same requested command: vx = 0.8 m/s
15-frame Hall history
Hall health
Frozen fast base
Risk / capture gate
Bounded residual
Cadence and stride adaptation
Fast-gait recovery
```

## 推荐添加的磁偶极子公式

公式不建议让图片模型直接生成，应由排版软件添加为可编辑矢量文字：

```text
B_ij = μ0 / (4πr³) [3(m_j · r_hat)r_hat − m_j]
B_i = Σ(j=1...4) B_ij
```

其中 `r` 为磁片中心指向 Hall 采样点的向量，最终磁场旋转到 Hall 局部坐标系。对过小的 `r` 使用数值下限保护。

## 论文图注草稿

**Fig. X | Physics-informed simulation and Hall-only friction-adaptive locomotion with a flexible magnetic sole.**  
**a,** The sole comprises a rigid robot sole, a PCB housing containing 15 Hall sensing sites, and a bottom magnetized TPU layer. Four disk magnets are embedded in the TPU beneath each Hall sensor. The 15 sensing sites follow the measured forefoot–midfoot–heel arrangement and are mirrored between the two feet.  
**b,** Ground contact produces local compression, bending and shear of the TPU, changing the relative position and orientation of the four embedded magnets. Their magnetic contributions are superimposed using a dipole approximation and expressed in the local Hall frame to obtain `Bx`, `By`, `Bz` and baseline-relative changes. Hall outputs are not converted into normal or tangential force.  
**c,** For massively parallel Isaac Lab training, detailed contact distributions drive a stable local-compliance approximation that updates virtual magnet poses. A replaceable `MagneticFieldModel` maps the magnet–Hall relative poses to a batched two-foot magnetic tensor. Contact mechanics and ground-truth friction remain simulation-internal and are excluded from the actor observation.  
**d,** Multi-frame Hall signals, sampling health and deployable proprioceptive history condition a bounded residual on top of a frozen fast locomotion base. Under a constant requested velocity, the policy adapts cadence, stride length and, when required, actual speed on low-friction ground, before recovering the fast gait on high-friction ground.

## 科学真实性说明

当前大规模强化学习训练主要采用稳定的 Scheme-A 局部顺应性/详细接触分布近似，而不是宣称所有训练环境都运行完整 PhysX 可变形 TPU 网格。Scheme-B 体积可变形 TPU 应作为后续高保真验证场景单独展示，不能在本图中误写成当前并行训练的默认实现。

## 算法部分详细 Prompt（可单独生成算法面板）

```text
Create the algorithm panel of a publication-quality robotics paper figure titled “Hall-only friction-adaptive locomotion policy”. Use a clean left-to-right flowchart with five stages and a small inset for the actor/critic information boundary. The visual language must be technical, causal and auditable, not a generic deep-learning diagram.

Stage 1 — Deployable observation construction:
Show synchronized left and right foot Hall streams. Each foot has 15 Hall sites and each site provides only Bx, By, Bz. Stack a causal temporal window of 15 frames, including baseline-relative magnetic changes when available, packet validity, sample age and channel-health indicators. Concatenate only deployable proprioception and the commanded velocity history. Label the resulting actor input as “1864-D policy observation” only if the implementation uses that exact schema; otherwise write “Hall + proprioception observation”. Make clear that the temporal window is causal and contains no future information.

Stage 2 — Hall feature encoder and risk/state representation:
Draw a compact temporal encoder or shared MLP that extracts magnetic temporal patterns rather than a single absolute magnetic value. The encoder should be visually associated with multi-frame Bx, By, Bz changes, left–right asymmetry, cadence-related periodicity, missing packets and valid-mask patterns. Add a separate health path that detects one-foot dropout, stale packets, non-finite values or severely degraded channels. Do not label this block as a force estimator. Its output is “Hall confidence / adaptive-state evidence”, not force, pressure or friction coefficient.

Stage 3 — Frozen fast locomotion base and adaptive branches:
Show a frozen high-friction fast locomotion base receiving the same deployable observation and velocity command. Its output is a 29-DOF nominal joint action a_base. In parallel, show a Hall gate/risk head producing a bounded capture probability or residual authority p_capture. Show a small bounded residual policy producing Δa. The final action is
    a = a_base + valid_left × valid_right × p_capture × Δa,
with an explicit per-joint/action-rate safety bound. If either foot is invalid, force the residual contribution to zero and fall back to the conservative nominal controller or safety envelope.

Stage 4 — Gait adaptation objective:
Show that the policy is not required to reduce speed whenever friction decreases. Under a fixed requested command, the adaptive branch may modify step frequency, step length, foot placement timing and only when necessary the commanded forward speed. Draw three alternatives in the low-friction branch: “cadence increase”, “shorter stride / larger support margin”, and “bounded speed reduction”. Mark the first two as preferred stabilizing actions and the third as conditional. The objective is to keep the body in a stable dynamical region while preserving forward progress.

Stage 5 — High–low–high state machine and recovery:
Draw an explicit hysteretic state machine with HIGH_START, LOW_TRANSITION, LOW_STABLE and HIGH_END/RECOVERY states. State transitions are driven by causal Hall temporal evidence and health checks, not by privileged ground-friction labels. HIGH_START uses the fast gait. LOW_TRANSITION applies bounded adaptive authority and avoids abrupt action switching. LOW_STABLE allows cadence/stride adaptation and conservative speed limiting. HIGH_END requires evidence that the magnetic pattern and body motion have returned to the high-traction regime, then smoothly releases the residual and restores the fast gait. Use 0.15–0.30 s action blending and a 5-frame command-history rewrite in the schematic if those mechanisms are part of the implementation.

Auxiliary training branch — privileged critic only:
Add a dashed gray branch from the simulator to a critic/reward module. It may use ground friction, contact distribution, contact-point tangential slip, local TPU deformation, terrain phase and other simulator-only variables to shape reward, curriculum and safety termination. Explicitly label this branch “training-time privileged information — not provided to actor”. The actor path must not receive friction coefficient, contact force, pressure, true slip, terrain label or deformation state.

Training and deployment loop:
Show Isaac Lab parallel environments generating H→L→H transitions with randomized TPU stiffness/damping, magnet parameters, Hall bias/noise/delay/dropout and terrain friction. Use the same causal observation and action interface for training and deployment. Show PPO updating the trainable gate/residual/critic components while the fast base is frozen or anchored. Mark model selection by held-out seeds and fault randomization, not by a single rollout. The final output is a deployable policy that consumes Hall Bx/By/Bz streams and proprioception and outputs 29 joint targets; no simulation-only signal is exported.

Use solid blue arrows for actor/deployment data, orange arrows for adaptive gait decisions, green arrows for Hall confidence, and dashed gray arrows for privileged critic-only variables. Include a small equation box for the bounded residual action and a small warning symbol reading “Hall field ≠ force”. Avoid generic labels such as “friction estimator” unless the block is explicitly described as a risk/state-evidence estimator. Do not show a direct B-to-force regression.
```

### 算法面板中每个模块应该表达什么

1. **输入不是单帧磁场，而是时序磁场**：15 帧、左右脚、15 个 Hall 点、每点 `Bx/By/Bz`，同时带有效位、延迟和健康状态。
2. **策略不是直接预测摩擦系数**：Hall 分支提取的是摩擦变化相关的时序证据和风险置信度，而不是把磁场回归成力或摩擦系数。
3. **基础控制器负责正常走路**：高摩擦时保持原有高速 gait；自适应分支只产生有限幅度的残差，避免破坏基础步态。
4. **低摩擦时不强制降速**：优先通过提高步频、缩短步长、调整落脚时序和扩大支撑裕度稳定身体，速度下降只是安全余量不足时的最后手段。
5. **高摩擦恢复必须有滞回和渐变**：不能一帧检测到高摩擦就突然切换，否则会产生动作跳变；应经过连续证据确认，再用短时间 action blending 恢复高速 gait。
6. **仿真真值只给 critic/reward**：摩擦系数、接触力、真实滑移和 TPU 形变可用于训练奖励或验收，但不能进入 actor observation，否则会形成真值泄漏。

### 算法图中建议添加的公式

```text
o_t = [H_{t-14:t}, health_{t-14:t}, proprio_t, command_{t-14:t}]

a_t = a_base(o_t) + M_t · p_capture(o_t) · clip(Δa(o_t), -δ, δ)

M_t = valid_left_t · valid_right_t

J = J_velocity + J_progress
    - λψ e_ψ² - λy v_y² - λω ||ω||²
    - λa ||a_t-a_{t-1}||² - λsat C_sat
    + λrecovery R_retention
```

其中 `M_t` 是传感器健康门控，不是摩擦真值；`p_capture` 只控制残差权限；`a_base` 保持高速基础步态；`Δa` 必须经过幅值和动作变化率限制。若论文当前版本没有显式使用某个公式，应删去对应公式，不要让示意图超出实际代码。

### 算法图的关键负面约束

```text
No direct Hall-to-force regression, no pressure estimator, no contact-force input to actor, no true friction coefficient input, no terrain-stage input to actor, no future observation, no non-causal smoothing, no unrestricted residual action, no abrupt policy switching, no claim that low friction always means slower speed, no claim that a single Hall frame identifies friction, no generic “AI brain” icon, no unexplained privileged arrow into the actor.
```

## 建议补充绘制的论文图组

仅有结构图和算法流程图还不够。建议论文至少形成以下图组，使“传感器物理机制—仿真实现—策略学习—可靠性验证—真机接口”闭环完整。

### Figure 1：足底结构与 15 点传感器布局

用途：证明仿真的足底结构和真实足底一致。

建议内容：

- 足底外形的俯视图，标出脚尖、前掌、中足、后跟方向；
- 左右脚各 15 个 Hall 点，编号顺序从脚尖到脚跟；
- 每个 Hall 点下方的 2×2 四磁片布局；
- 剖面结构：刚性足底、PCB 外壳、嵌入 TPU 的磁化层；
- 足底长度、宽度、TPU 厚度、Hall 高度和磁片间距；
- 左右脚镜像坐标系和局部 xyz 方向。

这张图主要回答：仿真中的传感器位置、结构层级和真实硬件是否对应。

### Figure 2：TPU 形变到磁场变化的物理机制

用途：解释为什么 Hall 只输出磁场，但磁场会携带接触和形变信息。

建议画成三列：

```text
无载状态 → 受压/弯曲/剪切状态 → Bx/By/Bz 变化
```

需要显示：

- TPU 局部压缩量 `dz`；
- 切向位移 `dx、dy`；
- 四个磁片的位置和姿态变化；
- Hall 局部坐标系；
- 四个磁片磁场矢量叠加；
- `B`、`B0` 和 `ΔB = B - B0`。

不要在图中画成“Hall 输出力”。正确标签应是：

```text
deformation changes magnet–Hall relative pose
relative pose changes magnetic field
```

### Figure 3：磁场计算与坐标变换验证

用途：让审稿人确认磁场计算没有坐标方向错误或数量级错误。

建议包含四个小图：

- 单个磁片的偶极子磁场方向和距离衰减；
- 四个磁片叠加后的 `Bx、By、Bz`；
- 世界坐标系到 Hall 局部坐标系的旋转；
- 左脚镜像到右脚后的磁场符号变化。

可画的定量曲线包括：

- `|B|` 随 Hall–磁片垂直距离变化；
- `Bx、By、Bz` 随 `dz` 变化；
- `ΔB` 随 `dx、dy` 变化；
- 四磁片叠加与单磁片贡献的对比。

这张图不是策略结果，而是磁场模型本身的单元验证。

### Figure 4：Isaac Sim 场景和大规模训练环境

用途：说明仿真如何产生 H→L→H 摩擦变化以及如何并行训练。

建议内容：

- 加长、加宽的蓝色高摩擦—黄色低摩擦—蓝色高摩擦路面；
- 三段地面必须共面且足够长，避免机器人因为跑出路面而被误判摔倒；
- G1 机器人在三段路面上的运动方向；
- 足底 Hall 点、TPU 层、磁片和磁场箭头可视化；
- 多环境网格排列；
- 每个环境的摩擦系数、TPU 刚度/阻尼和 Hall 故障随机化标注在仿真内部。

建议加入一条时间轴：

```text
HIGH_START → LOW_TRANSITION → LOW_STABLE → HIGH_END
```

### Figure 5：训练课程和随机化设计

用途：说明策略为什么具有不准确传感器下的鲁棒性。

建议画成课程阶梯图：

```text
阶段 1：高摩擦稳定高速
阶段 2：固定低摩擦步态适应
阶段 3：高速进入低摩擦的捕获步
阶段 4：低摩擦返回高摩擦的速度恢复
阶段 5：多故障、多种子、长时程验证
```

右侧用随机化矩阵表示：

- 磁体强度和间距；
- Hall 安装偏差；
- 交叉轴误差；
- 偏置、噪声、温漂和采样延迟；
- 坏通道、单脚掉线、双脚掉线；
- TPU 杨氏模量、阻尼和局部顺应性；
- 地面摩擦系数和切换位置。

要明确区分：随机化是训练增强，不是 actor 的额外输入。

### Figure 6：高—低—高摩擦下的步态和磁场时序

用途：直接证明“摩擦变化导致磁场时序变化，并引起步态适应”。

建议采用共享时间轴的多行图：

1. 地面摩擦阶段：High / Low / High；
2. 左右脚磁场模长或选定 Hall 点的 `Bx、By、Bz`；
3. 磁场基线变化 `dBx、dBy、dBz`；
4. Hall 有效性和延迟状态；
5. 步频、步长和左右脚相位；
6. `vx、vy、relative heading`；
7. residual 权限和最终动作修正量；
8. 最终速度与恢复时间。

重点不是只展示速度下降，而是展示：

```text
低摩擦：步频/步长/落脚时序先变化，速度只在必要时下降
高摩擦恢复：残差逐渐释放，步态恢复到高速模式
```

### Figure 7：与原始 Unitree 策略的公平对照

用途：回答“新方法到底比原始策略好在哪里”。

建议采用同一坐标、同一种子、同一摩擦切换和同一指令的 paired comparison：

- 原始 Unitree locomotion；
- Hall-only 基础策略；
- Hall + bounded residual；
- Hall + recovery / stability branch。

建议展示的指标：

- 首次摔倒时间；
- 零摔倒通过率；
- 高摩擦速度保持率；
- 低摩擦步频变化；
- 低摩擦步长变化；
- 高摩擦恢复时间；
- 横向速度 RMS；
- 航向误差 RMS；
- 动作饱和比例；
- Hall 失联时的安全响应时间。

所有曲线要使用相同 seed、相同地面长度、相同 episode 时长，并将首次摔倒后的 reset 样本从主速度统计中剔除。

### Figure 8：消融实验

用途：证明每个模块确实有贡献。

建议至少包括：

- 无 Hall 时序，只用单帧 B；
- 无左右脚差异；
- 无健康状态；
- 无 gate；
- 无 bounded residual；
- 不改变步频/步长，只允许降速；
- 不使用 detailed contact distribution；
- 无长期 HighEnd 稳定性训练；
- 旧的 force/contact proxy 与 corrected contact-point slip 标签对照。

首选 heatmap 或表格，而不是堆很多重复柱状图。每项消融应同时报告安全性和性能，不能只报告平均速度。

### Figure 9：故障随机化和失联安全机制

用途：证明“不准确柔性磁足底”不是只在理想传感器下有效。

建议画四种故障分支：

```text
Healthy
Single-foot dropout
Dead / delayed channels
Both-foot invalid
```

每个分支显示：

- 有效位变化；
- residual 是否关闭；
- command envelope 是否收紧；
- 步频/步长/速度如何变化；
- 恢复需要多长健康保持时间。

核心原则是：传感器失联时 confidence=0，不能继续使用不可信的 Hall residual。

### Figure 10：Isaac Sim 到 MuJoCo 再到真机的部署闭环

用途：说明算法不是只在单一仿真器里有效。

建议画成三列：

```text
Isaac Sim training → MuJoCo sim-to-sim → real robot candidate
```

三列之间只保留相同的部署接口：

```text
left/right Hall Bx, By, Bz
Hall health and timing
deployable proprioception
velocity command
→ 29-DOF joint action
```

在 Isaac Sim 一侧可以显示 privileged critic；在 MuJoCo 和真机侧必须把这些虚线变量删掉，以证明没有仿真真值依赖。

### 论文主文与补充材料的取舍

主文建议保留：

1. 结构与 15 点布局；
2. TPU 形变—磁场机制；
3. 算法与 actor/critic 信息边界；
4. H→L→H 步态和磁场时序；
5. 原始策略公平对照。

补充材料建议放：

1. 偶极子方向、数量级和坐标变换验证；
2. 15 个 Hall 点逐点响应；
3. 所有随机化范围；
4. 故障矩阵；
5. 多种子完整结果；
6. MuJoCo 和真机接口；
7. 失败轨迹和恢复前兆分析。
