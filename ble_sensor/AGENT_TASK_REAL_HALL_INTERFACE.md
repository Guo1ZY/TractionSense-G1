# 远程 Agent 任务：对齐 G1 双足 Hall 真机数据接口

你当前工作的真机代码根目录是：

```text
/home/mosense/guo_1/ble_sensor
```

请先完整阅读同目录下的真机数据规范：

```text
/home/mosense/guo_1/ble_sensor/doc/REAL_FOOT_HALL_DATA_FORMAT.md
```

这份规范已经对照本机 Isaac/部署工程和远程 BLE 代码整理。但你仍必须先读取当前工作树中的实际代码、配置和最新采集 manifest，不要盲目覆盖现有改动。若代码与规范冲突，先记录精确文件、行号和运行证据，再做最小修复。

## 不可违反的测量边界

1. 真实足底每脚只有 15 个 Hall 点，每点直接量是 `Bx/By/Bz` 原始计数和温度。
2. 不得将 Hall 数据命名或伪装为法向力、切向力、压力、摩擦系数或真实接触点。
3. 基线差分、温漂补偿、滤波和归一化仍是磁场信号预处理，不是 Hall-to-force 标定。
4. 左右数组顺序必须固定为 `[left, right, P00..P14, Bx/By/Bz]`。
5. 原始单位是 device counts，仿真内部是 Tesla，F0M1 是无量纲归一化响应；日志中必须明确单位，不得混用。

## 已确认的线路和 ABI

- 左脚：`left`，`98:A3:16:A1:BF:CA`，`hci0`；
- 右脚：`right`，`98:A3:16:A1:C1:2E`，`hci1`；
- BLE Notify UUID：`0000ab01-0000-1000-8000-00805f9b34fb`；
- 单脚 BLE 帧：125 bytes，15 组大端 `>hhhh = temperature_x10,x,y,z`；
- `F0R1`：936 bytes，用于原始采集/机器人时间对齐；
- `F0M1`：400 bytes，含 `[2,15,3] float32` 归一化 Hall；
- 当前 Motion actor 严格为 1864 维；Hall 历史在 `480:1830`，最后两维 `1862:1864` 是 `body_vy, relative_heading`，不是传感器 age。

任何对 BLE/F0R1/F0M1 字节格式的修改，必须同时更新 Python writer/reader、C++ reader、文档和字节级回归测试；不允许单边修改。

## 当前必须保持为“未标定”的项

1. 数据包不能证明 Hall 芯片型号，不得声称已确认为 MLX90393。
2. 没有已验证 counts-to-Tesla 系数。
3. `sensor_permutation` 和全局 `axis_sign` 当前仍是临时 identity。
4. 可视化中的逐芯片 XY 旋转不等于已验证真机轴变换；当前 bridge 只应用 permutation + 全局 axis sign。
5. A4 布局的横向正号、右脚镜像规则和足底 link 坐标对齐尚需已知方向的三轴磁体位移实验。
6. 截至 2026-08-11，没有已验收的 `normalization/left.json` 和 `normalization/right.json`，当前 `normalized=false`。不得把 raw counts 直接冒充正式 F0M1 策略输入。

## 请在远程主机上完成的工作

1. 扫描并汇报当前代码、配置、运行脚本、日志和 git 工作树；保留用户现有改动。
2. 用自测和字节级测试核验 125/936/400-byte ABI、左右脚顺序、15 点顺序和大小端。
3. 核对 `config.magnetic.json`、启动脚本和实际运行时中的 left/right 名称、地址、hci0/hci1，避免两个程序抢占同一 BLE Notify。
4. 先解决左脚链路掉频/重连。2026-08-09 步行 session 只有左脚 42.966 Hz、9 次重连、双脚同时有效率 43.58%，不能用于正式归一化或训练。
5. 采集左右脚分开的空载基线、温漂、逐点/逐轴已知方向磁体位移和可重复加载/卸载数据。
6. 标定完成后生成左右独立 normalization JSON，并报告每点每轴 baseline、scale、温漂、噪声、回程残余、饱和和坏通道。
7. 为每次采集保存 raw counts、temperature、host monotonic timestamp、valid/age/period、左右帧时差、重连次数、配置和 SHA256。
8. 只有数据质量门禁通过后才允许发布正式 F0M1；否则保持 raw-only/诊断状态。

## 最低验收证据

- 左右脚各至少 30 s 单脚原始采集，帧率均 `>=95 Hz`、坏帧 0；
- 至少 120 s 双脚采集，双脚同时有效率 `>=95%`，帧时差 P95 `<=20 ms`；
- 链路中断后 valid/age 及时失效，不重放旧帧；
- 标定 JSON 左右脚身份、shape、finite、scale>0 和 SHA 检查通过；
- 空载、单点、已知轴向、加载/卸载图全部保留 raw 与 normalized 曲线；
- 整个过程不产生任何伪“法向力/切向力”字段。

完成后请给出：改动文件 diff、实际运行命令、原始数据路径、质量报告、已通过/未通过项，以及尚未解决的真机风险。
