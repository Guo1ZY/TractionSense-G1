# 双足 Hall 真实接口、15 点布局与标定

> 真机采集、IPC、1864 维策略观测的完整字段和字节规范见
> [`真实足底传感器数据格式.md`](./真实足底传感器数据格式.md)。本文主要保留运行和标定操作流程。

## 已固定的测量边界

真实鞋垫每只脚输出 15 个通道，每通道只有原始 `Bx/By/Bz` 计数和温度。
所有采集、归一化和部署文件均禁止生成法向力或切向力。加载位置、倾斜、剪切
和滑移只是离线实验阶段标签。

## 左右脚身份

两块传感器固件的 BLE 广播名应分别设置为：

- 左脚：`left`
- 右脚：`right`

远端当前保存的地址为：

- 左脚：`98:A3:16:A1:BF:CA`
- 右脚：`98:A3:16:A1:C1:2E`

唯一名称用于自动发现，地址用于最终身份校验。名称本身不会修改 BLE 射频连接
间隔；可视化通过两路独立解析/缓冲、同步开启 Notify 和统一显示时钟避免一侧
刷新率拖累另一侧。

## A4 1:1 布局

布局文件为 `config/sensor_layout_a4_15.json`。根据 `2.png` 的 A4 1:1 描线得到
足底外轮廓和三组十字形，输出编号顺序为：前掌 P00–P04、中足 P05–P09、
后跟 P10–P14；每组均为“上、左、中、右、下”。当前统一到 Isaac 前向模型的
手绘测量初值为长 215.02 mm、宽 80.04 mm，15 个中心还保存了原图像素坐标，
避免可视化和训练各自二次手工测量。制造或装配完成后的正式尺量坐标必须同时
更新该 JSON 和 Isaac 的唯一配置表，并通过跨文件一致性测试后才能重新训练。

## 远端运行

```bash
ssh mosense@192.168.3.22
cd /home/mosense/guo_1/ble_sensor

# 两块鞋垫上电后确认唯一名称和地址
.venv/bin/python discover_ble.py --left-name left --right-name right --all

# 启动双脚可视化
cd vis
../.venv/bin/python ble_viz_dashboard_demo.py
```

本机简化可视化：

```bash
cd /home/mosense/guo_1/vola_sensor/vis
./run_dual_dashboard.sh \
  --left-address 98:A3:16:A1:BF:CA --left-adapter hci0 \
  --right-address 98:A3:16:A1:C1:2E --right-adapter hci1
```

该命令明确保证一只脚使用一个适配器。可视化、原始采集和实时桥不能同时抢占
同一 BLE 设备；切换程序前先确认上一进程已经退出。启动/重连后会采集固定空载
基线，期间必须让两只脚完全卸载。运行中不会自动追基线；真实回程残余需要在
确认空载后按 `B` 执行“空载归零”。

仅做诊断时可分别固定地址和适配器保存原始 NPZ：

```bash
# 终端 1
cd /home/mosense/guo_1/vola_sensor
.venv/bin/python record_raw_hall.py --foot_id left \
  --address 98:A3:16:A1:BF:CA --adapter hci0 \
  --duration_s 30 --output diagnostics/left_raw.npz

# 同时在终端 2
cd /home/mosense/guo_1/vola_sensor
.venv/bin/python record_raw_hall.py --foot_id right \
  --address 98:A3:16:A1:C1:2E --adapter hci1 \
  --duration_s 30 --output diagnostics/right_raw.npz
```

两条命令需要并发启动才可使用同主机 monotonic 到达时间进行离线配对。该工具
不做死区、基线、左右增益或力转换，并以临时文件原子落盘；诊断数据不能替代
下方引导式双脚标定会话。

## 引导式数据采集

先做不连接硬件的计划检查：

```bash
cd /home/mosense/guo_1/ble_sensor
.venv/bin/python capture_magnetic_dataset.py \
  --config config.magnetic.json --quick --dry-run
```

接线快速测试：

```bash
.venv/bin/python capture_magnetic_dataset.py \
  --config config.magnetic.json --quick \
  --phase baseline_unloaded --phase forefoot_normal --phase shear_x
```

完整采集：

```bash
.venv/bin/python capture_magnetic_dataset.py \
  --config config.magnetic.json \
  --note "双足最终装配，吊架保护"
```

程序会依次引导空载基线、温漂、前掌、中足、后跟、两个剪切方向、两个倾斜
方向、站立重心转移、低速行走和可控滑移。每阶段保存独立 CSV、SHA256、设备
身份和 `manifest.json`。

## 双脚归一化

```bash
.venv/bin/python calibrate_dual_magnetic.py \
  --session calibration/sessions/<session_name> \
  --output normalization
```

生成 `normalization/left.json`、`normalization/right.json` 和质量摘要。归一化
只包含空载基线、温度系数、每通道尺度和裁剪范围，仍不包含力转换。

完成后启动实时 F0M1 桥：

```bash
.venv/bin/python run_magnetic_bridge.py \
  --config config.magnetic.json \
  --record logs/dual_foot_$(date +%Y%m%d_%H%M%S).csv
```

只有终端持续显示 `F0M1=ON BLE=1/1`，且
`/tmp/g1_foot_magnetic_health.json` 中左右脚均 fresh，才可以进入吊架测试。
