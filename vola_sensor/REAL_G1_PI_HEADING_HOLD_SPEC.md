# G1 真机 PI 航向保持模块：velocity 指令层实现规格

版本 v1（2026-08-14）。本模块**不修改策略网络**，只在操作者速度指令与宇树
官方走跑/走路模式之间做有界航向修正。真机首发必须 `observe_only`。

## 1. 位置与数据流

```text
操作者速度指令 (vx, vy, yaw_rate)
        │
        ▼
PI heading hold（本模块，50 Hz，可旁路）
        │  修正后的 yaw_rate
        ▼
官方模式接口：802 走跑 / 501 控腰走路（限速策略由 supervisor 另管）
        │
        ▼
机载 IMU / 状态 → 相对航向 ψ 反馈回本模块
```

只读：直行段起始时的前进参考方向、IMU 偏航、操作者 yaw 指令、vx 指令。
不读：摩擦系数、接触力、课程阶段、Hall 数据。

## 2. 变量与单位

- `ψ`：相对航向误差（rad），= 当前前进方向 - 直行参考方向，左偏为正；
- `yaw_op`：操作者 yaw_rate 指令（rad/s）；
- `yaw_corr`：本模块输出修正（rad/s）；
- 控制周期 `dt = 0.02 s`（50 Hz）。

直行参考方向在"直行段开始"时锁存：操作者 vx>0 且 yaw 指令近零持续 N 步后，
以当前前进方向为参考；操作者给出明显转向指令时释放参考、清零积分。

## 3. 算法（有界 PI，带泄漏）

激活条件（全部满足）：

```text
enable == True
|yaw_op| <= 0.05 rad/s
vx_op >= 0.65 m/s
非 E-stop / 非故障回退
```

每步更新：

```text
I = decay * I + ψ            # 泄漏积分
I = clamp(I, -I_cap/Ki, I_cap/Ki)
corr = Kp * ψ + Ki * I
corr = clamp(corr, -yaw_cap, yaw_cap)
yaw_out = yaw_op - corr
yaw_out 再过一次变化率限幅（见 §4）
```

参数（仿真已验证的默认值）：

| 参数 | 值 |
|---|---:|
| Kp | 0.50 |
| Ki | 0.05 |
| decay | 0.995 |
| I_cap（积分对输出的贡献上限） | 0.20 rad/s |
| yaw_cap（总修正上限） | 0.40 rad/s |
| 激活 yaw 指令阈值 | 0.05 rad/s |
| 激活 vx 阈值 | 0.65 m/s |

## 4. 安全与复位

- **observe_only**：默认只记录 `suggested_yaw_correction`，不写入指令；
- **旁路开关**：任一故障/人工干预可立即 `enable=False`，输出 = 操作者原始指令；
- **变化率限幅**：`yaw_out` 相邻步变化不超过 0.25 rad/s²，防止指令阶跃；
- **积分清零**条件：模式切换（802↔501）、操作者转向意图、Hall 健康回退、
  supervisor 状态变化、E-stop、直行参考重置；
- **输出饱和**：`yaw_out` 永远在 ±0.40 rad/s 内。

## 5. 上机验收（observe 阶段）

1. 记录逐帧 `ψ、I、corr、yaw_out、是否激活`；
2. 检查修正方向与 ψ 符号一致、无振荡、无积分饱和持续；
3. 高摩擦直行段 `corr≈0`；低摩擦段 `|corr|` 有界且随 ψ 收敛；
4. 人工复核后，才讨论把 `suggested → applied` 打开。

## 6. 与策略导出的关系

策略 ONNX 只管 29 关节动作（`joint_target = default + 0.25 * onnx(obs)`）；
本模块只改 velocity 指令，两者独立、可分别开关。仿真中二者组合已通过
μ∈[0.20,0.28] 全部门禁（0 动力学摔倒、48/48 HLH、|Δy|<0.5 m）。
