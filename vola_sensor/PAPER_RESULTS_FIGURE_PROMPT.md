# 论文结果部分图 Prompt

> 结果图是**数据图**（有坐标轴和真实数字），AI 生图不能直接画数据（会编数字）。
> 正确用法：AI 只出**版式底图/示意图**，真实曲线用 matplotlib 从验收 JSON 绘制，
> 数字一律人工核对后补字。本文件给出：图规划 → 真实数据（供制图/补字）→ AI 版式 Prompt。

数据来源：`analysis/acceptance_20260812/*.json`（2026-08-12 多种子验收，配对 A/B，同任务同 seed）。

---

## 1. 结果图规划（4 张）

| 图 | 类型 | 内容 | 用什么画 |
|---|---|---|---|
| R1 主结果 | 路线图 + 速度剖面 | H→L→H 三段路面（μ 0.9/0.28/0.9 阴影）+ vx-位置曲线（model_52 vs 原始基线）+ gate 激活叠图 | matplotlib（trace npz）或 AI 底图+补曲线 |
| R2 完成率/摔倒 | 分组条形图 | 配对 A/B：完成 H→L→H 96/96 vs 94/96；摔倒 0 vs 8；强化故障 96/96, 1 fall | matplotlib |
| R3 步态自适应 | 分组条形图 | 低摩擦区步长 0.30→0.16 m（减半）vs 基线 0.34→0.34（不变）；步频 2.64→2.58 Hz | matplotlib |
| R4 鲁棒性 | 条形 + 散点 | gate AUC 标称 0.88 vs 强化 0.72；恢复时延中位数 0.64–0.94 s；长程 12m 边沿退出分类 | matplotlib |

## 2. 真实数据（全部来自验收 JSON，供制图与补字）

### R1 主结果（Medium 短程课程，seed 450，16 env，0.8 m/s 指令）
- 课程几何：高摩擦起点 x∈[−2,0)，低摩擦 μ=0.28 x∈[0,1)，高摩擦终点 x∈[1,2.6]
- model_52：vx_high_start 0.668 · vx_low 0.571 · vx_high_end 0.662 m/s
- 原始基线：vx_high_start 0.727 · vx_low **0.824**（低摩擦反而加速）· vx_high_end 0.840
- model_52 低区接触后 0.5 s 减速 0.228 m/s²、1.0 s 减速 0.313 m/s²（基线为负）
- gate：LOW/HIGH AUC 0.877，低摩擦段激活 100%，回到高摩擦释放后恢复 0.7 m/s 用时中位数 0.71 s
- 多种子（451–455）：HLH 全部 16/16、0 摔倒；恢复中位数 0.64–0.94 s

### R2 配对 A/B（6 seeds × 16 env，同任务同 seed）
- model_52 标称：**96/96 完成，0 摔倒**
- 原始基线：**94/96 完成，8 摔倒**（3/1/2/2/0/0）
- model_52 强化故障（掉点10%/死通道8%/延迟5帧）：96/96 完成，1 摔倒（seed 453）

### R3 步态自适应（24 m 长程，seed 450，12 m 宽）
- model_52 低摩擦区：步长 **0.301 → 0.162 m（−46%）**，步频 2.64 → 2.58 Hz，vx 0.783 → 0.484
- 原始基线低摩擦区：步长 0.335 → 0.343 m（+2%），步频 2.33 → 2.24 Hz，vx 0.795 → 0.834（不减速）

### R4 鲁棒性
- gate AUC：标称 0.877–0.901（激活 100%）；强化故障 0.676–0.767（激活 69–81%）
- 恢复时延（0.7 m/s 阈值）：中位数 0.64–0.94 s，15–16/16 达标
- 24 m 长程 12 m 宽（3 seeds）：model_52 首回合完成 12/16, 12/16, 14/16；
  全部 35 次"摔倒"均为走出 ±6 m 边缘（0 次动力学摔倒）；基线 3 次动力学摔倒
- 均匀高摩擦 30 s（model_49999）：3.2 m 宽 15/16 摔倒、平均存活 13.9 s；
  12 m 宽 11–13/16、平均存活 26.4–27.1 s，首摔横向位置全部 ≈ ±6.2–6.6 m（边沿退出）

## 3. AI 版式 Prompt（R1 主结果示意图；数据曲线留空，后期补）

```text
Create a Modern-Minimal style results overview figure for an IEEE RAL paper.
White background, two stacked panels (a) and (b), publication quality.

VISUAL STYLE:
- Ultra-clean geometric shapes, crisp edges, no 3D, no gradients
- Rounded corners (8-12px radius), no visible borders
- Thin arrows (1.5px), dark gray (#6B7280)
- Sans-serif font (Inter/Helvetica), titles bold, body regular
- No icons, no clip art, no photographs, no logos

COLOR PALETTE (exact hex):
- high-friction ground: #2F6FBF (blue)
- low-friction ground: #E6B325 (yellow)
- our method (model_52): #009E73 (green)
- original baseline: #8C8C8C (gray)
- gate activation: #D55E00 (vermillion)
- fault/edge annotations: #CC3311

PANEL (a) — "H-to-L-to-H friction course" top-down schematic:
A wide horizontal three-section floor strip, long and wide, all coplanar:
blue | yellow | blue, labeled
"high friction mu = 0.9" / "low friction mu = 0.28" / "high friction mu = 0.9".
A small G1 humanoid stick figure walks left to right across it.
Below the strip, one thin horizontal time axis labeled "course position x".
Above the strip one small label: "command vx = 0.8 m/s, unchanged across all sections".
Leave a large EMPTY plot area below the floor strip (for the real speed curves
to be added later): draw only axes and a small note "insert speed profile here".

PANEL (b) — "Speed and gait response" empty chart frame:
One empty plot frame with x-axis "position along course (m)" and y-axis
"forward speed (m/s)", plus one inset empty frame labeled "gate g"
with y-axis "0 to 1". Draw only the axes, tick marks and axis titles;
do NOT draw any curves, do NOT invent any data points.
Below the frame, two placeholder legend swatches with text
"our method (model_52)" in green and "original baseline" in gray.

CONSTRAINTS:
- Do not draw any curves, data points, bars, or numbers inside the plot areas
- All numbers and curves will be added later from real experimental data
- Keep all text spelled exactly as given
- Total aspect ratio about 7:5
- White background, all text dark gray or black
```

## 4. 重要提醒

1. **曲线必须用真实数据画**：R1 的速度剖面需要 `--trace_npz` 重跑 1–2 条 rollout（每条约 30 s）拿到逐帧 vx/位置/gate，然后用 matplotlib 绘制；AI 只能出空版式底图；
2. **R2–R4 直接用 matplotlib**（数据已在本文件 §2，绘制脚本可一键生成 PDF+PNG）；
3. AI 生图版只适合做汇报/海报的示意图，投稿终稿数据图一律 matplotlib 矢量输出；
4. 涉及统计声明的数字（96/96、0 vs 8、0.301→0.162）投稿前再与 `analysis/acceptance_20260812/` 原始 JSON 核对一遍。
