# G1 机器人端 Hall 摩擦模式桥接（安全部署区）

本目录用于机器人 PC4 上的真实双足 Hall 数据采集、摩擦分类和官方运动模式监督。
当前部署原则如下：

- 输入永远是左右脚 `15 × (Bx, By, Bz)` 多帧磁场、时间戳和健康状态，不转换成力。
- 左脚固定使用 `hci0`，右脚固定使用 `hci1`，两个 BLE 连接互不抢占。
- 未得到真实高/低摩擦标定模型前，只采集和记录，不输出摩擦结论。
- 未确认本机固件的 App 模式映射前，不猜测 FSM ID，不发送运动命令。
- 最终 supervisor 默认 `observe_only=true`，不注册 systemd 自启动；失联或模型异常时只输出保守请求。

## 当前只读模式探针

`loco_state_probe.py` 只注册官方 GET API，不包含任何 Set/Move 调用。机器人端执行：

```bash
cd /home/unitree/hall_friction_mode_bridge
PYTHONPATH=pydeps:vendor/unitree_sdk2_python \
  python3 app/loco_state_probe.py --interface eth0 --duration 30 \
  --output logs/loco_mode_probe.jsonl
```

运行期间可在 App 中依次选择“走跑”和“走路（控腰）”，输出状态变化用于建立该固件的精确映射。

## 数据边界

真实足底可直接获得的是 Hall 三轴原始计数/标定后的磁场变化以及温度、采样周期、有效性与包龄。
摩擦类别是基于多帧时空模式训练得到的分类结果，不是传感器直接测得的摩擦系数，也不是力。

## 最小真机闭环

第一版不重新部署 Isaac 策略，也不尝试直接控制指定步频：

```text
left hci0 ─┐
           ├─ 2×15×3 原始 Hall 多帧 ─ 摩擦分类 ─ 滞回状态机 ─┬─ HIGH: 走跑 / 高速度上限
right hci1 ┘                                                  └─ LOW/异常: 控腰 / 0.25 m/s 上限
```

官方高层接口没有直接设定步频的公开 API。这里由官方“走跑/控腰”策略自行改变步频和步长，
监督器只选择模式并限制速度。脚底尚未接触新路面前无法提前知道摩擦；第一次接触后的多帧响应
用于判断，因此状态机必须有保守启动与掉线回退。

## 标定数据采集

先单独运行 `capture_robot_hall.py`，使 `/tmp/g1_foot_hall_capture.bin` 持续存在。随后每个
独立路面试次运行一次：

```bash
PYTHONPATH=pydeps:app/ble \
  python3 app/collect_dual_friction_trial.py \
  --surface high --surface-name rubber \
  --controller-mode waist_walk --requested-vx 0.25 \
  --trial-id high_01 --duration 20
```

低摩擦试次只改 `--surface low --surface-name ... --trial-id ...`。至少各采 3 个独立试次，
推荐各 6 个。所有高/低试次必须使用完全相同的 `controller-mode` 和 `requested-vx`，否则
训练器会直接拒绝，防止把官方步态模式识别成地面摩擦。

训练并执行整试次留一验证：

```bash
PYTHONPATH=app:app/ble python3 app/train_friction_classifier.py \
  logs/friction_trials/*.npz --output models/friction_hall_v1.json
```

只有 `trial AUC >= 0.85`、`trial balanced accuracy >= 0.80` 且 `window AUC >= 0.75`
的模型才能被运行时加载。随机打散帧得到的高分不算通过。

## 观察模式运行

```bash
PYTHONPATH=app:app/ble python3 app/friction_mode_supervisor.py \
  --model models/friction_hall_v1.json
```

输出 `/tmp/g1_hall_friction_status.json`，但不会向机器人发送任何命令。当前版本若传
`--apply-mode-requests` 会故意报错；只有完成固件模式映射和控制权验证后才增加执行器。
