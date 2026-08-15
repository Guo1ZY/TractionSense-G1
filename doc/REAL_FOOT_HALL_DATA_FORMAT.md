# 真实双足 Hall 数据格式

真机传感器接口的完整中文规范位于：

[`/home/mosense/guo_1/vola_sensor/真实足底传感器数据格式.md`](/home/mosense/guo_1/vola_sensor/真实足底传感器数据格式.md)

该文档定义了：

- 单脚 125-byte BLE 原始帧；
- 单脚 NPZ、双脚 `raw_frames.csv` 和 `paired_50hz.csv`；
- 936-byte 原始采集包 `F0R1`；
- 400-byte 策略磁信号包 `F0M1`；
- `[left/right, P00..P14, Bx/By/Bz]` 顺序；
- F0M1 到 1864 维 Hall-only actor 观测的精确索引；
- 归一化、采集阶段、质量门槛和当前实测链路问题。

不可更改的边界是：真实 Hall 输出仅为三轴磁响应和温度，不是法向力或切向力。
