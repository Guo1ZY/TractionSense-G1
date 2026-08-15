# G1 双足 15 点磁场采集与控制桥

本项目保留原始参考程序 `ble_viz_superres_hot.py`，并将同一 125 字节
`FootSensor15` BLE 协议扩展为两只脚独立接收、原始数据记录、温漂校准、
归一化和运动控制 IPC。实时输出为 400 字节 `F0M1`：

```text
双足 BLE → 每脚 15×XYZ + 温度 → 标定/归一化 → F0M1
         → g1_ctrl 15 帧历史 → 共享左右脚编码器 → 29 关节动作
```

这里的 XYZ 是 Hall 原始计数，F0M1 是基线/温度补偿后的无量纲磁响应。
这条链路没有、也不允许输出法向力、切向力、压力或摩擦系数。参考 Dashboard
中的热图、中心、区域占比和 load/force 字样均是磁响应幅值的可视化代理，不能
作为 N、Pa 或真实 COP 使用。只有独立力传感器/仿真接触真值才能进入旧 F0T1
力策略；Hall-only 策略只读取 F0M1。

控制器默认读取 `/tmp/g1_foot_rl_obs.bin`。任一脚超过 0.20 秒没有新数据
时，桥停止刷新；控制器在 0.25 秒内判定数据失效并退出到 Passive。

## 1. 安装和发现两个设备

```bash
cd /home/mosense/guo/unitree_rl_lab/hardware/remote_ble_sensor
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

复制 `config.magnetic.example.json` 为 `config.magnetic.json`，填写左右脚
不同的 BLE 地址。左右脚标签、15 个通道顺序和 XYZ 正负方向必须在装入
鞋底后固定，不能靠模型猜测。可先用下文双脚 Dashboard 查看两个设备地址，
也可给原始记录器显式传入 `--address`。

Linux 主机有两个控制器时，可在左右脚配置中同时填写不同的 `adapter`
（例如 `left.adapter=hci0`、`right.adapter=hci1`），桥会把每只脚固定到对应
控制器；只配置一侧或两侧填写同一控制器会直接拒绝启动。`hciX` 编号可能在
重启或重新插拔后变化，运行前应使用 `bluetoothctl list` 对照控制器地址。

两只足底没有共享的硬件采样时钟，125 字节 BLE 帧也不携带硬件时间戳。
桥因此分别用同一主机的单调接收时间缓存左右帧，再将最新且时差不超过
`ble.max_pair_skew_s`（默认 10 ms）的一对送入同一个 50 Hz F0M1 包。可用
`ble.sync_holdback_s` 设置额外等待窗口，默认 0 ms 以免无收益地增加控制延迟。
超过时差上限时不会插值伪造数据，也不会把未配对的单脚新帧写入 F0M1；短暂
没有新配对时，控制器按原包时间戳和 age 自然保持上一对，连续两个控制周期
仍无有效配对才报告 `SYNC=0`。同步方法、等待窗口、最后/最大配对偏差和近期
配对率记录在健康文件的 `synchronization` 字段中。

## 2. 采集真实标定数据

先用 `--raw-only` 记录双脚悬空/无载荷数据（至少 2 分钟，期间覆盖实际
温度变化），再记录穿戴后的踩踏、前后剪切和缓慢行走数据：

```bash
.venv/bin/python run_magnetic_bridge.py \
  --config config.magnetic.json --raw-only \
  --record calibration/baseline.csv

.venv/bin/python run_magnetic_bridge.py \
  --config config.magnetic.json --raw-only \
  --record calibration/motion.csv
```

分别生成左右脚归一化文件：

```bash
.venv/bin/python calibrate_magnetic.py \
  --baseline calibration/baseline.csv --motion calibration/motion.csv \
  --side left --output normalization/left.json

.venv/bin/python calibrate_magnetic.py \
  --baseline calibration/baseline.csv --motion calibration/motion.csv \
  --side right --output normalization/right.json
```

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

双脚界面按两个唯一 BLE 地址分别连接同名 `FootSensor15`，配置文件为
`vis/dual_foot_dashboard.json`。启动命令：

```bash
cd /home/mosense/guo_1/vola_sensor/vis
/home/mosense/guo/unitree_rl_lab/hardware/remote_ble_sensor/.venv/bin/python \
  ble_viz_dashboard_demo.py
```

如果画面中的物理左右脚相反，按 `X` 可临时交换并重新连接；确认后再交换
配置文件中的 `left_address` 和 `right_address` 以永久保存。离线检查界面：

```bash
/home/mosense/guo/unitree_rl_lab/hardware/remote_ble_sensor/.venv/bin/python \
  ble_viz_dashboard_demo.py --mode demo
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
/home/mosense/guo/unitree_rl_lab/hardware/remote_ble_sensor/.venv/bin/python \
  ble_viz_dashboard_demo.py --mode demo \
  --demo-cadence 80 --left-weight 1 --right-weight 1 \
  --record-demo demo_videos/walking_demo_80spm.mp4 --record-seconds 8
```

`--demo-cadence` 使用双脚合计的 steps/min。仿真包含足跟着地、中足承重、
前掌推进、左右轻微差异、传感器噪声和缓慢基线漂移。
