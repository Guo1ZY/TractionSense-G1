# FootSensor15 足底多物理场可视化

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
