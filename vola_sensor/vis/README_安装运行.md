# FootSensor15 足底 Hall 磁场可视化

## 双脚实时可视化（新增）

双脚版本默认分别扫描广播名为 `left` 和 `right` 的 BLE 设备，两路数据使用
独立的帧解析、基线校准和滤波状态，并在两路 GATT 都连接后同步开启 Notify：

```bash
chmod +x run_dual_dashboard.sh
./run_dual_dashboard.sh
```

推荐在两块传感器固件中把广播名永久设置为 `left` 和 `right`。唯一名称解决
自动发现的身份歧义；独立缓冲、同步开启 Notify 和统一显示时钟负责避免一侧
采样频率拖慢另一侧。名称本身不会改变 BLE 射频连接间隔。也可以同时固定地址：

```bash
./run_dual_dashboard.sh \
  --left-address AA:BB:CC:DD:EE:01 \
  --right-address AA:BB:CC:DD:EE:02
```

使用其他广播名时显式指定：

```bash
./run_dual_dashboard.sh --left-name left --right-name right
```

足底外轮廓和 15 点位置都从 `../config/sensor_layout_a4_15.json` 加载。该文件以
`2.png` 的 A4（210×297 mm）1:1 描线为尺度基准；P00–P14 分为前掌、中足、
后跟三组，每个十字依次为“上、左、中、右、下”。修改实测毫米坐标或归一化
坐标不需要改可视化源码，左右脚仅镜像显示，不改变两路数据身份或 Hall 轴符号。

两路 BLE 原始通知率、10 ms 主机时间配对率和 50 Hz 显示率是三个不同指标。
界面先消化两次渲染之间的全部原始帧，再将最新真实配对按 50 Hz 节拍显示，
不会把重复显示帧计作新采样。每只脚独立使用稳健空载基线、迟滞死区和断流
衰减。空载校准完成后基线保持锁定，运行时不根据峰值下降或残余响应自动移动；
因此持续静态加载不会被错误清零。所有显示量始终是 Hall `Bx/By/Bz` 的 counts
变化，不是力或压力。

左右脚热图默认使用“每脚独立即时色标”，保证响应较弱的一脚也能看清分布，
但这种颜色不可用于跨脚比较；原始 counts 峰值卡不做增益。按 `G` 可切换为
共享 counts 色标，用于严格比较两脚绝对 Hall 响应。两种模式均不保留历史峰值
量程，因此超量程不会长期压暗后续颜色。双脚在同一机械加载下仍可能因磁片、
TPU、装配间隙和传感器偏置产生不同 Hall 响应，只有采集真实双脚标定数据后才
能建立独立归一化，不能用显示权重冒充力标定。int16 仅在相邻线传值跨越
60000 counts 时按 ±65536 展开，重连会重新锚定，普通大跳变不再误计圈数。

Hall 数据本身无法区分“持续静态加载”和“卸载后的 TPU/磁片回程残余”，因此
可视化不再自动追踪基线。真实残余会如实显示；确认两只脚已经完全卸载后，按
`B` 执行“空载归零”。不要在踩压状态按 `B`，否则该加载状态会被当成新基线。

离线查看双脚界面：

```bash
./run_dual_dashboard.sh --mode demo
```

双脚版快捷键：`D` 重连、`X` 交换左右、`B` 双脚空载归零、空格暂停、`H` 热图、
`M` 磁矢量、`G` 独立/共享色标、`C` Hall 响应中心、`I` 编号、`F9` 录制、
`S` 截图、`Esc` 退出。

打包双脚版：

```bash
chmod +x build_dual_linux.sh
./build_dual_linux.sh
./dist/FootSensorDualDashboard
```

## 运行环境

- Ubuntu 24.04/Linux x86_64（当前单文件程序的构建平台）
- Python 3.12 或更高版本
- 支持 BLE 的蓝牙适配器与 BlueZ
- FootSensor15 设备名称和程序中的名称一致
- 视频录制功能需要 FFmpeg

Python 依赖只有：

- numpy
- pygame
- bleak

## 已打包程序：直接运行

当前目录已经生成 Linux x86_64 单文件程序：

```bash
chmod +x dist/FootSensorDashboard
./dist/FootSensorDashboard
```

这种方式不需要安装 Python 或执行 `pip install`，但电脑仍需启用 Linux 蓝牙
服务。使用视频录制时仍需安装 FFmpeg。

如果目标电脑的 Linux 发行版明显旧于 Ubuntu 24.04，建议使用下方源码方式，
由目标电脑本机创建 Python 环境，避免 glibc 版本不兼容。

## 源码方式：一键安装与运行

```bash
cd <本项目目录>
chmod +x install_env.sh run_dashboard.sh
./install_env.sh
./run_dashboard.sh
```

也可以手动安装：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python ble_viz_dashboard_demo.py
```

程序默认直接扫描并连接真实 `FootSensor15`，不需要添加 `--mode ble`。

## 系统依赖

Ubuntu/Debian：

```bash
sudo apt update
sudo apt install bluez ffmpeg fonts-noto-cjk
sudo systemctl enable --now bluetooth
```

如果普通用户无法扫描 BLE，请先确认：

```bash
bluetoothctl show
bluetoothctl scan on
```

设备必须已上电且处于广播状态。程序顶部状态栏会显示扫描、连接和 Notify
数据诊断信息。

## 打包为 Linux 单文件程序

```bash
chmod +x build_linux.sh
./build_linux.sh
./dist/FootSensorDashboard
```

单文件程序已包含 Python、NumPy、Pygame、Bleak 和 Logo，但不会内置系统
蓝牙服务与 FFmpeg。换到另一台 Linux 电脑时，仍需安装 `bluez`；要使用录屏
功能还需安装 `ffmpeg`。

## 输出文件

- 视频：`recordings/footsense_*.mp4`
- 截图：`screenshots/footsense_*.png`
