# G1 双足 Hall 摩擦数据采集格式（v1）

## 1. 测量边界

真实足底直接记录的测量值仅包括：

- 左、右脚各 15 个霍尔点 `P00...P14` 的 `Bx, By, Bz` 原始计数；
- 每个点的温度；
- BLE 帧时间戳、序号、采样周期、包龄和有效标志。

数据集不得把 Hall 数据预先转换成法向力、切向力、压力或摩擦系数。`dBx, dBy, dBz` 只能作为相对静态基线的派生磁场特征，并保留原始值。

## 2. 机器人端目录

```text
/home/unitree/hall_friction_mode_bridge/
├── app/capture_robot_hall.py
├── app/loco_state_probe.py
├── app/robot_deploy/collect_dual_friction_trial.py
├── app/config.magnetic.json
└── logs/
    ├── robot_capture_sessions/
    ├── friction_trials/
    └── loco_trials/
```

左右脚按蓝牙适配器 MAC 地址绑定，不能依赖重启后可能变化的 `hciN` 编号：

- 左脚 BLE：`98:A3:16:A1:BF:CA`；当前适配器 `F4:4E:FC:44:B6:10`；
- 右脚 BLE：`98:A3:16:A1:C1:2E`；当前适配器 `F4:4E:FC:CE:51:3B`。

## 3. 原始 BLE 文件

`raw_frames.csv` 每收到一只脚的一帧就写一行，典型速率约为每脚 100 Hz。关键字段为：

```text
wall_time_ns, monotonic_ns, side, device_name, address, adapter,
source_sequence, sample_period_s, valid,
temp_0_x10 ... temp_14_x10,
mag_0_x, mag_0_y, mag_0_z ... mag_14_x, mag_14_y, mag_14_z
```

`side` 必须为 `left` 或 `right`，点号顺序固定为 `P00...P14`。

## 4. 双足同步文件

`paired_50hz.csv` 以 50 Hz 发布，不对 Hall 数据插值，每次使用左右脚各自最近一帧。必须保留：

```text
publish_sequence, publish_wall_ns, publish_monotonic_ns,
left_right_frame_skew_ns,
left_valid, left_age_s, left_sample_period_s, left_frame_monotonic_ns,
left_P00_bx ... left_P14_bz,
right_valid, right_age_s, right_sample_period_s, right_frame_monotonic_ns,
right_P00_bx ... right_P14_bz
```

正式训练使用的 NPZ 中：

| 键 | 形状 | 类型 | 含义 |
|---|---:|---|---|
| `sequence` | `[N]` | `uint64` | 50 Hz 发布序号，必须连续 |
| `publish_monotonic_ns` | `[N]` | `int64` | 同机同步主时钟 |
| `frame_monotonic_ns` | `[N,2]` | `int64` | 左右脚真实帧时刻 |
| `valid` | `[N,2]` | `bool` | 左右脚数据是否有效 |
| `age_s` | `[N,2]` | `float32` | 左右脚包龄 |
| `period_s` | `[N,2]` | `float32` | 实测采样周期 |
| `hall_xyz` | `[N,2,15,3]` | 整数 | `[left/right, P00...P14, Bx/By/Bz]` |
| `temperature_x10` | `[N,2,15]` | 整数 | 摄氏度乘 10 |
| `metadata_json` | 标量 | UTF-8 JSON | 地面、模式、命令和试验信息 |

## 5. 标签与官方控制模式

固定地面试验必须记录：

- `surface_label`: `high` 或 `low`；
- `surface_name`: 实际材料名称；
- `controller_mode`: `walkrun` 或 `waist_walk`；
- `fsm_id`: 当前固件实测映射，走跑为 `802`，走路（控腰）为 `501`；
- `requested_vx_mps`, `requested_vy_mps`, `requested_wz_radps`；
- `trial_id`, `operator_note`, 起止 UTC 时间；
- 机器人只读状态日志的文件名和 SHA256。

采集器只监听机器人状态，严禁调用 `SetFsmId`、`SetVelocity` 或其他运动命令。运动由操作者通过官方 App/手柄完成。

## 6. 防混淆实验矩阵

必须采齐四种组合，否则分类器可能把步态模式变化误认为摩擦变化：

| 地面 | 走跑 `802` | 稳腰走路 `501` |
|---|---|---|
| 高摩擦 | 必采 | 必采 |
| 低摩擦 | 必采，先低速并防护 | 必采 |

用于在线模式切换的关键过渡数据为：

1. 走跑模式下 `high -> low`；
2. 稳腰模式下 `low -> high`。

同一对高/低摩擦分类对照必须保持控制模式、速度档位、方向和试验时长一致。先分别在 `vx=0.2, 0.4, 0.6 m/s` 采集；确认安全后再扩展到更高速度、转向和加减速。

## 7. 首轮纯高摩擦走跑命名

首轮建议：

```text
surface_label=high
surface_name=<实际高摩擦材料>
controller_mode=walkrun
fsm_id=802
requested_vx_mps=0.4
requested_vy_mps=0.0
requested_wz_radps=0.0
duration_s=20
```

机器人必须处于防护环境，操作者保持直行和固定滑杆。若 FSM 不为 802、任一脚无效、双脚健康占比低于 98%、50 Hz 序号不连续，整次试验作废并重采。

