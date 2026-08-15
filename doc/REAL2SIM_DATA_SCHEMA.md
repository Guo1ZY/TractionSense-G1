# Real2Sim 同步数据 Schema（草案）

zorn 当前为 Isaac ContactView **仿真**足底，collector ~20 Hz，ROS 50 Hz 重复帧，`Float32MultiArray` 无 Header 时间戳。  
**不得**把无时间戳 multiarray 当作严格系统辨识数据集。

## 推荐采集路径

```
真实传感器驱动 / 仿真 ContactView
  → 源时间戳 + 接收时间戳 + seq
  → rosbag2 (raw)
  → 离线 Parquet/HDF5 (同步表)
```

## 统一记录字段（最低集合）

| 字段 | 类型 | 说明 |
|------|------|------|
| trial_id | str | 试验 ID |
| sequence_id | u64 | 递增 |
| source_timestamp_ns | u64 | 传感器/仿真源 |
| receive_timestamp_ns | u64 | 主机接收 |
| left_taxel_raw[N] | f32 | 原始 ADC/taxel（真实协议未定时先占位） |
| right_taxel_raw[N] | f32 | 同上 |
| left_fn, right_fn | f32 | 校准法向力 N |
| left_cop_xy, right_cop_xy | f32[2] | CoP 局部 |
| left_valid, right_valid | bool | 饱和/掉线/超时 |
| sensor_temp | f32 | 若有 |
| joint_pos[29], joint_vel[29] | f32 | |
| joint_tau_est[29] | f32 | 估计力矩/电流 |
| imu_quat, imu_gyro, imu_acc | f32 | |
| base_vel_est | f32[6] | |
| foot_pose, foot_vel | f32 | FK |
| joystick_cmd | f32[3] | vx,vy,wz |
| policy_obs | f32[D] | 完整 actor 向量 |
| policy_action | f32[29] | |
| motor_target | f32[29] | 实际下发 |
| control_hz, latency_ms | f32 | |
| ground_label | str | tile/wood/ice/... |
| mu_gt | f32? | 有标定才填，禁止伪造 |
| fall_event, e_stop | bool | |
| calib_version | str | 传感器标定版本 |
| robot_config | str | 质量/脚掌安装 |

## 仍需用户提供（未提供则只定义接口）

1. 真实足底硬件协议 / ADC 位宽 / taxel 数量与布局  
2. 法向-only 还是含剪切  
3. 标定板 / 已知 μ 地面参考  
4. 与 G1 低层控制的时间同步方式（DDS 时钟 vs ROS）  

## 标定流程（概要）

zero offset → ADC→力 → 左右增益 → 温度 → 迟滞 → 串扰 → 饱和/死区 → CoP → 安装坐标系  

## Real2Sim 辨识环

真实轨迹 → 时间对齐清洗 → 仿真 replay 同 cmd/action → 优化 μ、接触刚度/阻尼、电机 delay、传感器模型 → 更新 DR → 独立轨迹验证 → 再训/微调  
