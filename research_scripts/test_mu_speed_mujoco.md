# MuJoCo：大指令 + 切 μ 测摩擦适应

思路：固定「速度指令尽量大」，只改地面 μ，看行为差。

| 期望 | 高 μ（键 3 GRIP） | 低 μ（键 1 ICE） |
|------|-------------------|------------------|
| 满杆前进 | 能跟上、少摔，\|v\| 偏高 | 不应硬冲到摔；更慢/更收 或 略滑但稳 |
| 旧 49999/窄包络 + 强行放大 clamp | 大指令易摔 | 更容易摔 |

注意：`cmd = clamp(gain * stick, deploy.ranges)`，**上限由 deploy.yaml 决定**，不是 gain 无限放大。

---

## 1. 准备（foot_full 训完后）

`deploy.yaml` 与训练一致：

```yaml
lin_vel_x: [-0.5, 1.2]
lin_vel_y: [-0.3, 0.3]
ang_vel_z: [-0.6, 0.6]
```

ONNX：`config/policy/velocity/foot/exported/policy.onnx`

---

## 2. 启动

终端 1 — MuJoCo：

```bash
export G1_MUJOCO_FOOT_BRIDGE=1
./research_scripts/run_mujoco_friction.sh normal
```

终端 2 — 策略（大指令：gain≥1，满杆尽快顶到 1.2）：

```bash
cd TractionSense-G1/deploy/robots/g1_29dof
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# 满杆更容易打满上限；增益过大仍被 clamp 到 1.2
export G1_CMD_GAIN_LIN=1.0
# 若摇杆行程不足、推不满，可试 1.2～1.5（仍 clamp 在 1.2）
# export G1_CMD_GAIN_LIN=1.3
./run_g1_ctrl.sh --network lo
```

终端 3 — 可选看速度/足底：

```bash
# MuJoCo 窗口按 V 已有 |v_xy|
./research_scripts/watch_foot_bridge.sh 5
```

---

## 3. 操作协议（A/B）

1. **A** 站稳 → **X** 进策略
2. 焦点 **MuJoCo 窗口**，按 **3**（GRIP μ≈1.8）
3. 左摇杆 **前后推满**，保持 5–10 s
   - 记：是否摔、\|v_xy\| 大概多少
4. **不松杆**，按 **1**（ICE μ≈0.08）
   - 记：是否摔、\|v\| 是否下降、是否乱滑
5. 再按 **3** 对比恢复
6. 可选 **4**（ULTRA-ICE）做更狠一档

也可用 **F** 在 1↔2↔3 循环。

---

## 4. 怎么判「μ 适应有没有」

| 结果 | 含义 |
|------|------|
| 3 能稳走约 0.8–1.2，1 明显更慢/更收、少摔 | **好**（大指令下按 μ 调节） |
| 3 和 1 都约 1.0 且都稳 | 适应弱，但至少比「一加速就倒」好 |
| 3 稳、1 猛冲后摔 | 低 μ 仍硬跟指令，需再训 |
| 两边大指令都摔 | 包络/模型不匹配，或仍是旧窄模型 |

对照「原来版本」：把 ONNX 换回 49999/窄 clamp 的 v0 或旧 foot，**同样** `G1_CMD_GAIN_LIN` + 键 1/3，应更容易在大指令下倒。

---

## 5. 不要做的

- 把 `lin_vel_x` 上限改到 **超过训练**（如 2.0）只靠 yaml → OOD 必摔
- 只加 `G1_CMD_GAIN_LIN=3` 以为能超过 clamp
- 焦点在 g1_ctrl 终端按 1/3（无效，必须 MuJoCo 窗口）
