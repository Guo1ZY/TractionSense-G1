# 普通 G1 两地面自动变速实施说明

## 最终实现

不估计连续物理摩擦系数，只识别实验中的两类已知地面：

```text
普通 G1 五帧本体历史（480维）
  gyro / gravity / command / q / dq / last action
                    │
                    ▼
       二分类器 ONNX：p_low
                    │
          滞回 + 时间一致性判断
                    │
        LOW 0.20 / HIGH 0.35 m/s
                    │
        原有 v0 行走策略 → 29维动作
```

足底与 Hall 数据不在输入中；已有不准确数据等价于被丢弃，而不是让网络
猜测其含义。两块地面在采集时由实验人员提供类别标签，部署时由本体历史
自动分类。这个方法针对“固定机器人 + 固定鞋底 + 固定两种测试地面”，
不是通用摩擦系数测量仪。

## 已完成的仿真验证

同一原始命令均为 `0.8 m/s`，控制器自动选择 `0.20/0.35 m/s` 上限：

| 测试 | 自动状态 | 实际 vx | |vy| | 摔倒 |
|---|---|---:|---:|---:|
| 独立启动，μ=0.15 | LOW | 0.149 | 0.033 | 0 |
| 独立启动，μ=1.20 | HIGH | 0.326 | 0.059 | 0 |

连续摩擦切换也完成了 `LOW↔HIGH` 自动转换且无摔倒，反向序列响应约
`2.4–2.9 s`。长时间不复位的连续序列存在横向漂移和高摩擦速度下降，
因此首轮真机必须在两块地面上分别起步测试，不能直接测试跨接缝奔跑。

当前仿真分类器：

```text
<repo>/logs/evaluations/traction_classifier_480/
20260730_v0_binary_dagger1/traction_classifier.onnx
```

它只证明软件闭环可行，**不得直接作为真机分类器**。真机启动脚本已加入
保护，不会默认选择该模型。

## 真机采集

全程使用额定保护架、独立急停操作员，关闭横向和转向输入。两块地面都采用
相同的 `v0` 策略、`0.20 m/s` 上限和相同前推摇杆位置，每次持续至少
20 秒。低摩擦板必须固定，先分别测试，不跨地面边界。

进入目录并确认真机网卡：

```bash
cd <repo>/deploy/robots/g1_29dof
ip -br link
export G1_REAL_TEST_ACK=YES
```

低摩擦地面采集三次：

```bash
./collect_two_surface_proprio.sh low --network <interface> --log
```

高摩擦地面采集三次：

```bash
./collect_two_surface_proprio.sh high --network <interface> --log
```

每次按 `A` 站立、`X` 行走，保持同一个前推量；完成后按 `B`，再
`Ctrl-C`。脚本会输出各自的 `low.npz` 或 `high.npz`。采集器只保存
480 维本体前缀，`mu=0.15/1.20` 只是 LOW/HIGH 类别编码，不代表实测值。

## 训练与显式安装

把三次 LOW 和三次 HIGH 的 NPZ 都传入：

```bash
./train_real_two_surface_classifier.sh --install \
  /path/to/low_run1/low.npz \
  /path/to/low_run2/low.npz \
  /path/to/low_run3/low.npz \
  /path/to/high_run1/high.npz \
  /path/to/high_run2/high.npz \
  /path/to/high_run3/high.npz
```

`--install` 才会生成：

```text
config/traction/real_classifier.onnx
```

训练输出的准确率只是数据完整性检查；由于数据可按用户需求混合使用，
真正的验证必须是下一轮未参与训练的闭环行走。

## AUTO 真机测试

先分别在两块地面重新起步，不跨接缝：

```bash
export G1_REAL_TEST_ACK=YES
./run_two_surface_governor.sh auto --network <interface> --log
```

状态切换时终端会打印 `state=LOW/HIGH source=auto`，日志保存原始命令、
限速命令和 `p_low`。手柄保留人工安全覆盖：

- `RB + ↓`：强制 LOW；
- `RB + ↑`：强制 HIGH，仅限高摩擦地面；
- `RB + ←`：回到 AUTO/重新判断；
- `B`：Passive。

首轮通过条件：

- 低摩擦地面从起步到结束不出现一次错误 `HIGH`；
- 高摩擦地面在约 3 秒内进入 `HIGH`；
- 同一摇杆输入下，外部相机测得低摩擦约不高于 `0.20 m/s`，高摩擦
  至少达到 `0.28 m/s`；
- 每块地面三次、每次至少 10 秒，零摔倒；
- 横向漂移没有持续增加。

若某次误判，保留该次 `policy_obs.bin` 并按真实地面重新转换、加入训练集，
再训练一次。这就是最小化的真机 DAgger；它比继续增加仿真随机化更直接。

## 论文表述边界

当前有效方法应写成“proprioceptive history–based binary traction-state
classification and hysteretic speed governance”。可以如实说明机器人装有
实验性 Hall 足底、其通道在本实验中施加 dropout/噪声并做消融，但不能把
自动变速归因于 Hall 足底感知。若要把足底作为论文核心贡献，必须重新标定
并证明加入足底后相对 480 维本体基线有统计显著提升。
