# G1 双足 15 点磁场采集与控制桥

本项目保留原始参考程序 `ble_viz_superres_hot.py`，并将同一 125 字节
`FootSensor15` BLE 协议扩展为两只脚独立接收、原始数据记录、温漂校准、
归一化和运动控制 IPC。实时输出为 400 字节 `F0M1`：

```text
双足 BLE → 每脚 15×XYZ + 温度 → 标定/归一化 → F0M1
         → g1_ctrl 15 帧历史 → 共享左右脚编码器 → 29 关节动作
```

控制器默认读取 `/tmp/g1_foot_rl_obs.bin`。任一脚超过 0.20 秒没有新数据
时，桥停止刷新；控制器在 0.25 秒内判定数据失效并退出到 Passive。

## 1. 安装和发现两个设备

```bash
cd /home/mosense/guo_1/ble_sensor
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python discover_ble.py
```

两块传感器固件的 BLE 广播名分别设置为 `left` 和 `right`，再运行：

```bash
.venv/bin/python discover_ble.py --left-name left --right-name right
```

`config.magnetic.json` 同时保存唯一名称和唯一地址；连接以地址为最终身份，名称
用于扫描和防止左右脚误绑定。唯一名称不会改变 BLE 射频连接间隔，双脚
Dashboard 还会同步开启 Notify、独立缓冲并统一重采样，避免一侧刷新率拖慢
另一侧。左右脚标签、15 个通道顺序和 XYZ 正负方向必须在装入鞋底后固定，不能
靠模型猜测。

## 2. 采集真实标定数据

推荐用引导式采集工具完成空载、温漂、前掌、中足、后跟、剪切、倾斜、站立、
行走和可控滑移阶段：

```bash
.venv/bin/python capture_magnetic_dataset.py --config config.magnetic.json
```

先用 `--dry-run` 检查计划，或用 `--quick` 做 8–12 秒接线测试。每个阶段保存
独立 CSV、左右设备身份、SHA256 和 `manifest.json`。原始数据始终只有
15×(Bx/By/Bz) 与温度，不创建法向力或切向力列。

也可直接用 `--raw-only` 记录双脚悬空/无载荷数据（至少 2 分钟，期间覆盖实际
温度变化），再记录穿戴后的踩踏、前后剪切和缓慢行走数据：

```bash
.venv/bin/python run_magnetic_bridge.py \
  --config config.magnetic.json --raw-only \
  --record calibration/baseline.csv

.venv/bin/python run_magnetic_bridge.py \
  --config config.magnetic.json --raw-only \
  --record calibration/motion.csv
```

对引导采集的完整 session 一次生成左右脚归一化文件：

```bash
.venv/bin/python calibrate_dual_magnetic.py \
  --session calibration/sessions/<session_name> \
  --output normalization
```

也可分别生成左右脚归一化文件：

```bash
.venv/bin/python calibrate_magnetic.py \
  --baseline calibration/baseline.csv --motion calibration/motion.csv \
  --side left --output normalization/left.json

.venv/bin/python calibrate_magnetic.py \
  --baseline calibration/baseline.csv --motion calibration/motion.csv \
  --side right --output normalization/right.json
```

### 真机实验数据采集接口

真机实验使用独立的原始采集入口。启动前它会检查 `hci0/hci1` 是否同时存在，
并拒绝与 BLE 可视化、标定或另一个采集进程抢占设备：

```bash
./start_robot_hall_capture.sh --preflight-only
./start_robot_hall_capture.sh --note "顶置吊架，低速行走"
```

启动后终端应持续显示 `F0R1=ON BLE=1/1`。按 `Ctrl-C` 请求正常结束；看到
`[CAPTURE] status=complete` 后本轮文件才封存完成。每轮创建独立目录：

```text
logs/robot_capture_sessions/robot_hall_<timestamp>/
  raw_frames.csv    # 两条 BLE 链路各自到达的原始帧
  paired_50hz.csv   # 同一主机单调时钟上的左右脚 50 Hz 配对快照
  health.json       # 左右脚连接、valid、age、周期和坏帧统计
  manifest.json     # 设备/适配器身份、文件 SHA256 和采集结果
```

`paired_50hz.csv` 在每个统一发布时刻保留两侧最新的真实帧（零阶保持），不对
Hall 数值插值；应结合每侧 `frame_monotonic_ns`、`age_s` 和
`left_right_frame_skew_ns` 判断配对质量。需要分析每个真实 Notify 时，使用
`raw_frames.csv`。

采集期间还会原子发布 `/tmp/g1_foot_hall_capture.bin`（`F0R1`，固定大小），
同机机器人记录器可用
`dual_foot_bridge.capture_ipc.read_packet()` 读取。包内保留左右脚各自的
15×3 Hall、15 路温度、帧时间戳、源序号、valid、age 和采样周期；若某脚失联，
该脚 `valid=false`，旧数组不得作为新测量使用。停止采集后 F0R1 文件会删除，
避免误读陈旧数据。

### 真机采集可视化控制台

远端图形桌面使用以下入口：

```bash
cd /home/mosense/guo_1/ble_sensor
DISPLAY=:1 ./start_robot_hall_capture_ui.sh
```

控制台本身不连接 BLE，因此可以先打开检查状态。它固定要求左脚使用 `hci0`、
右脚使用 `hci1`，并在缺少适配器、旧可视化/标定/采集进程仍占用 BLE、或当前
阶段安全确认未完成时禁用“开始采集”。当前第一轮真机采集只保留三个阶段：
悬空无载（180 秒）、吊架双足站立（90 秒）和吊架低速直走（120 秒）。它们用于
先验证无载稳定性、静态双脚链路和动态时间对齐；低摩擦、侧移、转向等工况待
三项稳定后再单独加入。工况名称只作为实验注释，绝不由 Hall 数据生成力、压力、
接触力或摩擦真值。

每个阶段的时长从左右脚均连接且 fresh 后开始计算。界面分别显示左右脚的
adapter、BLE/fresh、采样率、age、帧数、坏帧和温度，并在封存后按固定门槛
显示 `PASS / REVIEW / FAIL`。只有看到“本阶段已封存，现在可以卸力/调整”后，
操作员才应改变载荷或进入下一阶段。

同一窗口还从 `/tmp/g1_foot_hall_capture.bin` 读取 F0R1，按 A4 实测鞋底轮廓和
P00–P14 布局显示左右脚 Hall 三轴变化。可视化不建立额外 BLE 连接；左右脚使用
完全相同的固定无载基线、死区、滤波和共享 ΔB counts 色标，不设置左右权重。
持续形变时基线不会自动追踪，因此不会把保持按压错误清零；信号回到固定无载
死区后显示会快速退色。“无载重置显示基线”按钮只有在当前阶段明确要求并已
确认完全卸力时才能使用。该显示基线只服务 UI，不是 normalization 文件。

显示基线还包含稳定性门控：至少观察连续 15 秒窗口，并检查 45 个 Hall 轴的
线性漂移速率。只有漂移 P95 不高于 `1.2 counts/s` 且最大值不高于
`2.5 counts/s` 才锁定；否则界面持续显示左右脚 P95/max 漂移并保持未校准。
不得通过放大死区隐藏硬件温漂、TPU/磁片回弹或装配应力释放。

每个批次的 `ui_operator_events.jsonl` 记录 UI 开始、人工安全停止、窗口关闭停止、
显示基线重置和子进程结束事件；`manifest.json` 同时记录 `stop_reason`，固定时长
阶段若由信号提前结束会保持为 `incomplete`，不会伪装成完成。

原始双脚采集质量门槛为：任一脚低于 `80 Hz`、坏帧非零、双脚有效率低于 95%，
或左右帧时间戳偏差 P95 超过 `50 ms`，均为 `FAIL`；低于 `95 Hz` 或 P95 高于
`20 ms` 但未触发 FAIL 时为 `REVIEW`。这些门槛针对数据可用性，不把 Hall 解释为
机械力学真值。

`manifest.json.raw_frame_timing` 给出每只脚真实 Notify 到达的全程平均频率、
间隔 P50/P95/最大值及 ≥40/100 ms 长间隔计数。它与 50 Hz 配对快照、最后一次
health 滚动频率共同用于诊断，不能由平均值掩盖间歇性右脚掉频。

## 3. 启动实时桥

```bash
.venv/bin/python run_magnetic_bridge.py \
  --config config.magnetic.json \
  --record logs/dual_foot_$(date +%Y%m%d_%H%M%S).csv
```

终端必须持续显示 `F0M1=ON BLE=1/1`。健康详情在
`/tmp/g1_foot_magnetic_health.json`，其中包含左右脚独立的有效位、数据
年龄和实际采样周期。控制器启动前应检查：

```bash
stat -c '%s bytes %y' /tmp/g1_foot_rl_obs.bin
```

正常包必须严格为 `400 bytes`。实机吊架测试前不要绕过失效保护，也不要
直接把未标定的原始 int16 数据送入策略。

## 4. 启动双脚可视化 Dashboard

双脚界面按 `left`、`right` 唯一名称发现，再以两个唯一 BLE 地址校验身份；
配置文件为 `vis/dual_foot_dashboard.json`，15 点 A4 实测布局位于
`config/sensor_layout_a4_15.json`。启动命令：

```bash
cd /home/mosense/guo_1/ble_sensor/vis
../.venv/bin/python ble_viz_dashboard_demo.py
```

如果画面中的物理左右脚相反，按 `X` 可临时交换并重新连接；确认后再交换
配置文件中的 `left_address` 和 `right_address` 以永久保存。离线检查界面：

```bash
../.venv/bin/python ble_viz_dashboard_demo.py --mode demo
```

单个蓝牙适配器同时连接两只高频鞋垫时，BlueZ 协商出的两路原始 Notify
频率可能不同。Dashboard 会先同步启用两路 Notify，每只脚独立保留短帧缓冲，
再以同一个时间戳线性插值到统一的 60 Hz 数据时钟。顶部和设备卡片显示的
`sync Hz` 应始终相同；`Raw links` 仅保留蓝牙控制器实际协商出的物理 Notify
频率用于诊断，不再决定任何一侧的动画或计算更新速度。每次单侧重连也会清空
该侧旧的频率统计，避免断线时间把 Hz 长期拉低。

程序启动时会在主 Dashboard 外额外打开独立的 `Left / Right Load Weights`
窗口。窗口内的两个水平滑块可以分别调整左右脚占比权重，范围为
`0.10–3.00`，并实时显示权重比例。权重只参与左右及分区域占比计算，不修改
原始传感器数据；停止拖动后会自动保存到 `vis/dual_foot_dashboard.json`。

按 `F9` 开始或停止 MP4 录制。程序优先使用系统 `ffmpeg`，未安装时自动使用
Python 依赖 `imageio-ffmpeg` 自带的 FFmpeg，无需额外安装系统软件。

可直接生成指定步频的双足步态演示视频，例如：

```bash
../.venv/bin/python ble_viz_dashboard_demo.py --mode demo \
  --demo-cadence 80 --left-weight 1 --right-weight 1 \
  --record-demo demo_videos/walking_demo_80spm.mp4 --record-seconds 8
```

`--demo-cadence` 使用双脚合计的 steps/min。仿真包含足跟着地、中足承重、
前掌推进、左右轻微差异、传感器噪声和缓慢基线漂移。
