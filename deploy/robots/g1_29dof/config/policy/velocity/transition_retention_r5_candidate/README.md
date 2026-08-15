# R5 transition-retention candidate（1864-D Hall/proprio actor）

部署公式：`joint_target = default_joint_position + 0.25 * policy.onnx(obs)`。
观测 ABI 与 hall_traction_r26_r25_candidate 完全一致（15+15+15+145+145+145
proprio + 15 帧 Hall 1350 + 采样周期 30 + 有效位 2 + motion feedback 2）。
最后两项运行时是 `[body_vy, relative_heading]`，不是传感器 age。

安全约定（首个真机 bring-up）：

1. F0M1 桥必须已运行且 `BLE=1/1`；任一脚 stale >0.25 s，C++ 端将该脚置零，
   不得当作有效测量；
2. 必须先吊挂 + 独立硬件急停 + `G1_HALL_HARNESS_ACK=HALL_B_ONLY_HARNESS`；
3. 第一轮只允许低速指令（deploy.yaml lin_vel_x 上限 0.6）；
4. 末尾两通道 `[body_vy, relative_heading]` 必须与训练 ABI 一致（不得回退成
   `foot_sensor_age_lr`）。`relative_heading` 由 C++ 对 IMU yaw 锁存计算；
   `body_vy` 在真机无世界速度 sidecar，已接打包的
   `exported/lateral_velocity_estimator.onnx`（1862→1，EMA alpha=0.35，
   C++ 侧 preflight 缺少该来源会 fail-closed，见 run_transition_retention_r5.sh）；
5. PI 航向保持暂未接入部署 C++（仿真已验证，接口规格见
   vola_sensor/REAL_G1_PI_HEADING_HOLD_SPEC.md），先上基础策略再单独接入；
6. 全程保留回退：config.yaml 改回 `policy_dir: config/policy/velocity/v0`
  即为官方策略。

gate / 低摩擦残差 / stability 残差全部在 policy.onnx 内部，部署侧不另设
外部风险模型。
