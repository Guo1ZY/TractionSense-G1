# 论文算法图 Prompt（完整版·最终）

> 用途：Gemini 生图（示意图/幻灯片素材）；投稿终稿请用 matplotlib 矢量版
> （`vis/paper_figs/fig_policy_arch_style_test.pdf`，数字精确）。
> 本 Prompt 中所有数字已于 2026-08-13 与代码逐项核对，25/25 通过。
> AI 生图后必须用 Illustrator/Inkscape/PowerPoint 补字（AI 会拼错密集数字标签）。

---

## 1. 图的三面板规划

| 面板 | 标题 | 内容 | 对应论文位置 |
|---|---|---|---|
| (a) | Observation o_t ∈ R^1864 | 1864 维度堆叠条 + 两个展开（本体 5×96、Hall 张量）+ 特权信号边界 | 方法 §X.1 |
| (b) | Student policy (deployed) | Hall 编码器 → 冻结教师(safe/fast/stable) → a_base；gate 支路 → g；residual 支路 → δ；合成 a = a_base + g·δ → 29 关节动作；失联 fail-closed 分支 | 方法 §X.3（核心） |
| (c) | Training only (teacher-student) | 高速教师(anchor) / 低摩擦专家(蒸馏) / 特权标签(stage-BCE) 三条虚线箭头指向学生 | 方法 §X.4 |

颜色语义：**灰=冻结**，**绿=可训练 gate**，**橙红=可训练 residual**，**蓝=动作输出**，
**红虚线=fail-closed**，**灰虚线=训练侧/特权信号**。

---

## 2. 与代码核对的数字清单（25/25 ✓）

| 项 | 值 |
|---|---|
| 观察总维度 | 1864 = 480 + 1350 + 30 + 2 + 2 |
| 本体每帧 | 96 = ω(3) + g(3) + cmd(3) + q(29) + q̇(29) + a(29)，共 5 因果帧 |
| Hall 历史 | 1350 = 15帧 × 2脚 × 15点 × 3轴 |
| 采样周期 | 30 = 15帧 × 2脚 |
| Hall 编码器 | 每点 MLP 3→16→16；每帧 MLP 241(=15×16+1)→64→32；Conv1d k=3 ×3；每脚 latent 32（冻结复用） |
| 融合维度 | 548 = 480 + 32×2 + 4 |
| 教师子策略 | safe / fast(带 μ̂ 头) / stable，各 548→512→256→128→29 |
| μ̂ 牵引门 | σ((μ̂−0.65)·10)，μ̂∈(0, 1.30] |
| 教师置信度 | calib × evidence × valid × fresh |
| boost 混合 | 1.12 × conf × traction × σ((vx−0.70)·15) → lerp-mix |
| gate 头 | 548→128→32→1 → sigmoid → 校准 σ(2.75·logit−3.2) → ×min(valid_L, valid_R) = g |
| residual 头 | 548→256→128→29 → δ = 0.55·tanh(·) |
| 合成 | a = a_base + g·δ，clamp ±3，29 关节位置动作 |
| 训练 anchor | 复合输出锚定 speedboost112 教师（HIGH 段），cap 0.3 |
| 训练蒸馏 | Stage7 低摩擦专家（反事实指令 0.16 m/s）监督 residual，仅 LOW 段，supervised_only |
| 训练 gate | 仅 stage-BCE 监督（梯度隔离），high-end 权重 4.0 |
| 信号边界 | 力/接触/摩擦/滑移不进 Actor 观察（ABI 审计禁词） |

---

## 3. 完整 Gemini Prompt

```text
Create a Modern-Minimal style technical architecture diagram for an IEEE RAL
paper. Precise, clean, publication-ready, like a method figure in a robotics
journal. White background, three panels stacked vertically: (a) observation,
(b) deployable student policy, (c) training-only teacher-student structure.

VISUAL STYLE:
- Ultra-clean geometric shapes with crisp edges, no 3D, no gradients
- Rounded corners (8-12px radius), no visible borders; boxes float on faint
  section backgrounds
- One accent color per logical group, used sparingly
- Thin arrows (1.5px), dark gray (#6B7280), small filled circle at source,
  clean arrowhead at target
- Sans-serif font (Inter/Helvetica), titles bold, body regular
- Labels INSIDE boxes, generous whitespace, no icons, no decorative elements

COLOR PALETTE (exact hex):
- proprioception: #56B4E9 (sky blue)
- Hall magnetic: #E69F00 (orange), light fill #FDF0D8
- meta channels (period/valid/feedback): #B8B8B8 and #9A9A9A
- frozen teacher: #E3E3E3 fill, #8A8A8A border
- trainable gate path: #009E73 (green), light fill #D8F3EA
- trainable residual path: #D55E00 (vermillion), light fill #FBE3D5
- action output: #0072B2 (blue), light fill #DCEBF7
- training-side zone: dashed gray #777777
- fault path: #CC3311 dashed red

PANEL (a) — title "Observation o_t in R^1864", a wide horizontal band:
Draw ONE stacked bar of four contiguous segments, widths proportional to
480 : 1350 : 30 : 4, colored blue / orange / gray / darker gray.
Four leader lines below point to labels:
  "proprioception history, 480 = 5 frames x 96"
  "Hall magnetic history, 1350 = 15 frames x 2 feet x 15 sites x 3 axes"
  "sample period, 30 = 15 x 2"
  "valid 2 + feedback 2"
Under the proprio label, a small exploded strip of six mini-blocks:
omega(3), g(3), cmd(3), q(29), q-dot(29), a-1(29), with note "x 5 causal frames".
Under the Hall label, a small isometric tensor sketch labeled
"[T=15, feet=2, sites=15, axes=3 (Bx, By, Bz)]".
At the right end of panel (a), one dashed gray rounded box:
"privileged (sim only): contact force / ground mu / slip — critic & gate
labels, never in o_t (no Hall-to-force conversion)".

PANEL (b) — title "Student policy (deployed): frozen fast base +
Hall-gated bounded residual". Five horizontal layers:

Layer 1 (Hall encoder, frozen): chain of three rounded boxes with arrows:
  "Hall 1350 + 30 / [T,2,15,3]" (orange)
  "shared Hall encoder: per-point MLP 3->16 -> per-frame MLP 241->64->32 ->
   temporal Conv1d (T=15, k=3)" (light orange)
  "per-foot latent 32 x 2" (light orange)
with small note above: "frozen, reused from speedboost112 teacher".

Layer 2 (frozen teacher): one LARGE light-gray rounded container titled
"FROZEN speedboost112 teacher -> a_base". Inside it three white boxes side
by side:
  "safe policy 548->512->256->128->29"
  "fast policy 548->...->29 (+ mu-head)"
  "stable policy 548->...->29"
and two annotation lines at the container bottom:
  "mu -> traction gate sigma((mu-0.65)*10) · confidence =
   calib*evidence*valid*fresh"
  "boost = 1.12 x conf x traction x sigma((vx-0.70)*15) -> lerp-mix"
Plus one italic line under the container:
  "frozen: low-friction training never erases high-speed gait".

Layer 3 (trainable paths): one light gray box on the left
"features 548 = 480 + 32x2 + 4". Two colored paths flow right from it:
  GREEN path (three green boxes in a row): "gate head 548->128->32->1",
  "sigmoid * calib sigma(2.75*logit - 3.2)", "x min(valid_L, valid_R)"
  -> output g; annotation "g = gate(o_t) x min(foot validity)".
  VERMILLION path (two boxes): "residual head 548->256->128->29",
  "bound: delta = 0.55 * tanh(.)" -> output delta; annotation
  "bounded correction cannot override the base gait".

Layer 4 (composition): a white circle-plus node containing
"a = a_base + g * delta" and small note "clamp +/-3", receiving one gray
arrow labeled "a_base" from the teacher container and two colored arrows
labeled "g" and "delta" from the green/vermillion paths.

Layer 5 (output): one blue box "29 joint position actions" and a small box
"G1 robot" next to it, one blue arrow from the node.

Fault branch: one short dashed red arrow from the green "x min(valid)" box
down to a dashed red rounded box:
"foot dropout => g = 0 => pure base gait + external speed envelope".

PANEL (c) — title "Training only (teacher-student)", one large dashed gray
rounded container. Inside it three small dashed boxes with three dashed
arrows pointing UP into panel (b):
  1. "High-speed teacher (speedboost112)" ->
     arrow to the frozen-teacher container in panel (b), labeled
     "anchor loss on HIGH: student composite stays close to teacher"
  2. "Low-grip recovery expert (Stage7, 0.16 m/s)" ->
     arrow to the vermillion residual path, labeled
     "distillation target for residual delta (LOW stages only, supervised)"
  3. "Privileged friction labels (never in o_t)" ->
     arrow to the green gate path, labeled
     "stage-BCE supervises gate (gradient-isolated)"
Add one final line of small dashed gray text at the very bottom of panel (c):
"contact force / friction / slip: training side only (drive deformation,
supervise gate), never in o_t — no Hall-to-force conversion".

CONSTRAINTS:
- Spell every label exactly as given; do not paraphrase, do not add stages,
  do not invent numbers; all numbers above are verified values
- Do NOT draw loss curves, training hyperparameters, or extra modules
- No Hall-to-force conversion anywhere; no "force sensor" or "pressure map"
- Do not draw the robot itself except a small labeled rectangle "G1 robot"
- No icons, no clip art, no photographs, no logos, no watermark
- Total aspect ratio about 7:5 (three stacked panels)
- White background, all text dark gray or black
- All three panels (a)(b)(c) and the fault branch and both signal-boundary
  lines must be present

After generation, add or fix any garbled text in Illustrator/Inkscape/PowerPoint:
AI image models routinely misspell dense dimension labels.
```

---

## 4. 使用提醒

1. **生图后必须补字**：1864 / 480 / 1350 / 548 / 29 / 0.55 / 2.75 / −3.2 等数字，AI 大概率拼错；
2. **投稿终稿用 matplotlib 版**：`vis/paper_figs/fig_policy_arch_style_test.pdf`（矢量、数字精确、可复现）；
3. **AI 排版失败时的降级方案**：退回精简版（部署侧 4 块 + 训练侧 3 条虚线箭头），核心语义（冻结动机 / 有界修正 / fail-closed / 不转力）不丢；
4. 与本图配套的整体机制图 Prompt 见同目录 `PAPER_SIMULATION_FIGURE_PROMPT.md`（仿真与行走场景的 4 面板 a–d 图）。
