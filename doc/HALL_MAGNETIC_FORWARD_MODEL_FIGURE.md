# 柔性磁感知足底正向测量示意图

图的核心结论是：地面接触引起 TPU 磁化层的局部压缩、剪切和弯曲，改变每个霍尔元件下方四个磁片的相对位置与磁化方向；磁偶极子正向模型将这些位姿映射为霍尔局部坐标系中的 `Bx/By/Bz`。霍尔测量边界止于磁场及其零载基线变化，不包含法向力或切向力反演。

图中曲线使用项目方案 A 相同的 SI 磁偶极子公式和工程默认磁矩，沿一条人为指定的复合形变路径生成，仅用于说明方向和数量级，不是载荷标定曲线，也不能解释为力—磁场关系。

生成命令：

```bash
cd /home/mosense/guo/unitree_rl_lab
/home/mosense/miniconda3/envs/isaaclab-v2/bin/python \
  scripts/figures/plot_hall_magnetic_forward_model.py
```

输出包括可编辑文字的 SVG、矢量 PDF、600 dpi PNG/TIFF，以及生成曲线所用的 CSV 数据。
