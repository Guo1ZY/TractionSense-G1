# G1 足底力—摩擦—速度自适应训练

## 目标

新任务 `Unitree-G1-29dof-Velocity-Foot-TractionAdaptive` 不覆盖已经训练的
`StraightMu`。正常速度指令为 `vx ∈ [-0.3, 1.0] m/s`，横向速度和转向均为零；
15% 的运动 episode 使用 `1.0–1.5 m/s` 前向压力指令，使手柄过大输入仍处于训练分布内。

策略期望行为：

- 高摩擦：跟随给定速度，压力样本中允许快走或轻跑到 1.5 m/s。
- 低摩擦：即使收到 1.5 m/s，也跟随平滑安全速度上限，优先直线、低滑移和不摔倒。
- 足底传感器丢失或过期：依靠 IMU/关节历史退化运行，而不是把零力误认为悬空。

## 方法依据

- RMA 用域随机化训练基础策略，再从短历史隐式推断环境参数，支持对湿滑地面等未知动力学快速适应：
  <https://www.roboticsproceedings.org/rss17/p011.html>
- DreamWaQ 表明纯本体感觉可以学习隐式地形表征，不要求部署时直接给真实地形参数：
  <https://arxiv.org/abs/2301.10602>
- Real-World Humanoid Locomotion 使用观测—动作历史做因果适应，支持未见环境上的在线调整：
  <https://arxiv.org/abs/2303.03381>
- SlipSense 直接使用传感足的力信号和 LSTM 做早期滑移识别，说明足底力时间序列比单帧阈值更有价值：
  <https://arxiv.org/abs/2606.24350>
- 在线摩擦辨识工作通过接触动力学估计摩擦，说明 `Ft/Fn` 应被理解为接触利用率线索，不能直接当作真实 μ：
  <https://arxiv.org/abs/2502.16843>

因此 actor 不接收仿真真实 μ，只接收 0.3 秒的左右脚接触、法向力、切向力、
`Ft/(Fn+eps)`、负载比例和传感器健康历史。critic 和奖励可读取精确随机化 μ，作为训练教师。

## 奖励与随机化

随机化为每个环境分配一个一致的材料桶，静摩擦 `0.05–1.20`，动摩擦
`0.04–1.10`；写入 critic 的 μ 与实际赋给 PhysX 的值完全一致。

安全速度上限为平滑函数：

`v_cap(μ) = 0.20 + 1.30 * sigmoid((μ - 0.55) / 0.14)`

奖励跟踪 `sign(cmd) * min(abs(cmd), v_cap)`，同时惩罚超过安全上限、摩擦锥利用率过高、
足端滑动、侧向漂移和偏航。步态周期从低速约 0.85 秒平滑缩短到高速约 0.50 秒，
它只是弱引导，不强制固定步态。

## 训练

先执行 20 轮冒烟训练：

```bash
/home/mosense/guo/scripts/finetune_g1_foot.sh --traction-adaptive --smoke
```

正式训练从 `model/rl/model_49999.pt` 部分加载，actor 的前 480 维和 critic 的前
495 维与基础策略完全对齐：

```bash
/home/mosense/guo/scripts/finetune_g1_foot.sh \
  --traction-adaptive \
  --num-envs 4096 \
  --max-iterations 16000 \
  --run-name foot_traction_adaptive
```

## 验收矩阵

固定前向指令分别测试 `0.5、1.0、1.5 m/s`，每种指令测试 μ
`0.08、0.20、0.40、0.80、1.20`，每格至少 5 个随机种子。记录：存活率、身体 `vx/vy`、
偏航率、足端滑移、摩擦锥利用率和横向位移。

进入 MuJoCo/实机的最低门槛：

- 同一 1.5 m/s 指令下，μ≥0.8 的中位 `vx` 明显高于 μ≤0.2。
- μ≤0.2 时不追逐 1.5 m/s，且存活率优先于速度误差。
- `|vy|` 和累计横移显著低于 `StraightMu model_11999`。
- 足底数据断开 0.25 秒时不立即发散；恢复后能继续稳定行走。

只有通过 Isaac Lab 固定 μ 矩阵后才导出 ONNX；先在独立策略目录测试，不覆盖当前可用策略。
