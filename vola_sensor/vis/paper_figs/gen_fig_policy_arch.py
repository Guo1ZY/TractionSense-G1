#!/usr/bin/env python3
"""RAL-style method figure (style test): 1864-D observation tensor + policy architecture.

Double-column IEEE/RAL width 7.16 in.  Vector PDF + high-DPI PNG.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

# ---------- palette (Okabe-Ito, muted for IEEE/RAL) ----------
C_PROPRIO = "#56B4E9"   # sky blue   - proprio history
C_HALL = "#E69F00"      # orange     - Hall magnetic history
C_META = "#B8B8B8"      # gray       - period/valid/feedback
C_FROZEN = "#E3E3E3"    # light gray - frozen teacher
C_FROZEN_E = "#8A8A8A"  # frozen border
C_GATE = "#009E73"      # green      - trainable gate
C_RESID = "#D55E00"     # vermillion - trainable residual
C_ACT = "#0072B2"       # blue       - actions
C_PRIV = "#777777"      # dashed privileged
C_FAULT = "#CC3311"     # fault path
C_TEXT = "#222222"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "svg.fonttype": "none",
})

FIG_W, FIG_H = 7.16, 4.30
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

def box(x, y, w, h, fc, ec, text=None, fs=7.0, weight="normal",
        rounded=0.004, ls="-", lw=1.0, tc=C_TEXT, ha="center", va="center",
        zorder=3, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rounded}",
                       transform=ax.transAxes, fc=fc, ec=ec, ls=ls, lw=lw,
                       zorder=zorder, alpha=alpha)
    ax.add_patch(p)
    if text is not None:
        ax.text(x + w / 2, y + h / 2, text, ha=ha, va=va, fontsize=fs,
                color=tc, weight=weight, transform=ax.transAxes, zorder=zorder + 1)

def arrow(x0, y0, x1, y1, color=C_TEXT, lw=1.2, ls="-", zorder=4, style="-|>",
          ms=7, shrinkA=0, shrinkB=0):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                        mutation_scale=ms, color=color, lw=lw, ls=ls,
                        transform=ax.transAxes, zorder=zorder,
                        shrinkA=shrinkA, shrinkB=shrinkB)
    ax.add_patch(a)

def note(x, y, text, fs=6.3, color=C_TEXT, ha="center", style="normal", zorder=5,
         weight="normal"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color,
            style=style, weight=weight, transform=ax.transAxes, zorder=zorder)

# ================================================================
# LEFT PANEL: observation tensor  (x in [0.012, 0.375])
# ================================================================
LX0, LX1 = 0.012, 0.375
ax.text((LX0 + LX1) / 2, 0.965, "Observation  $\\mathbf{o}_t \\in \\mathbb{R}^{1864}$",
        ha="center", va="center", fontsize=8.5, weight="bold", color=C_TEXT,
        transform=ax.transAxes)

# ---- main stacked bar ----
bar_y, bar_h = 0.885, 0.050
bar_w = LX1 - LX0
fracs = [480 / 1864, 1350 / 1864, 30 / 1864, 4 / 1864]
colors = [C_PROPRIO, C_HALL, C_META, "#9A9A9A"]
labels = ["proprio\n480", "Hall magnetic history\n1350", "period 30", "valid+fb 4"]
x = LX0
seg_bounds = []
for frac, col in zip(fracs, colors):
    seg_w = max(frac * bar_w, 0.004)
    seg_bounds.append((x, x + seg_w))
    ax.add_patch(Rectangle((x, bar_y), seg_w, bar_h, fc=col, ec="#FFFFFF",
                           lw=0.8, transform=ax.transAxes, zorder=3))
    x += seg_w

# segment labels with leader lines (staggered heights, no mutual overlap)
labs = [
    (seg_bounds[0], "proprio 480\n= 5 frames × 96", 0.845, C_PROPRIO, "left", seg_bounds[0][0]),
    (seg_bounds[1], "Hall magnetic history\n1350 = 15 × 2 feet × 15 sites × 3 axes", 0.800, C_HALL, "center", (seg_bounds[1][0] + seg_bounds[1][1]) / 2),
    (seg_bounds[2], "period\n30 = 15 × 2", 0.715, C_TEXT, "right", LX1),
    (seg_bounds[3], "valid 2\n+ fb 2", 0.630, C_TEXT, "right", LX1),
]
for (x0, x1), lab, yl, cc, ha, xa in labs:
    xm = (x0 + x1) / 2
    ax.plot([xm, xa], [bar_y - 0.004, yl + 0.008], color=cc, lw=0.7,
            transform=ax.transAxes, zorder=2)
    note(xa, yl, lab, fs=6.0, color=cc, ha=ha)

# ---- Hall tensor cube expansion ----
cube_x0, cube_y0, cube_w, cube_h = 0.02, 0.55, 0.135, 0.115
note(cube_x0 + cube_w / 2, cube_y0 + cube_h + 0.014,
     "Hall block expanded", fs=6.0, color=C_HALL)
ax.add_patch(Rectangle((cube_x0, cube_y0), cube_w, cube_h, fc="#FDF0D8",
                       ec=C_HALL, lw=1.0, transform=ax.transAxes, zorder=3))
# stacked frames (left mini-lines)
nf = 7
for i in range(nf):
    yy = cube_y0 + 0.012 + i * (cube_h - 0.024) / nf
    ax.plot([cube_x0 + 0.012, cube_x0 + cube_w * 0.30], [yy, yy],
            color=C_HALL, lw=0.8, transform=ax.transAxes, zorder=4)
note(cube_x0 + cube_w * 0.35, cube_y0 + cube_h - 0.028, "T=15 frames", fs=6.2,
     color=C_HALL, ha="left")
# two feet squares side by side
for k, dx in enumerate((0.0, 0.023)):
    yy = cube_y0 + 0.015
    ax.add_patch(Rectangle((cube_x0 + cube_w * 0.42 + dx, yy), 0.018, 0.024,
                           fc="#F5C242", ec="#B8860B", lw=0.6,
                           transform=ax.transAxes, zorder=4))
note(cube_x0 + cube_w * 0.42 + 0.050, cube_y0 + 0.048,
     "left / right\nfoot", fs=6.0, ha="left", color=C_TEXT)
note(cube_x0 + cube_w * 0.42, cube_y0 + 0.002,
     "15 sites × 3 axes", fs=6.0, ha="left", color=C_TEXT)

# ---- proprio expansion ----
px0, py0 = 0.02, 0.395
note(px0 + 0.10, py0 + 0.075, "proprio block expanded\n96 per frame:", fs=6.0,
     ha="left", color=C_PROPRIO)
sub = [("ω", 3), ("g", 3), ("cmd", 3), ("q", 29), ("q̇", 29), ("a₋₁", 29)]
sw = 0.0165
for i, (lab, d) in enumerate(sub):
    x0 = px0 + i * (sw + 0.003)
    w = sw * (d / 29)
    ax.add_patch(Rectangle((x0, py0), w, 0.035, fc=C_PROPRIO, ec="white",
                           lw=0.5, transform=ax.transAxes, zorder=3))
    note(x0 + w / 2, py0 - 0.017, lab, fs=5.4, color=C_TEXT)
note(px0 + 6 * (sw + 0.003), py0 + 0.0175, "× 5 causal frames", fs=6.2,
     color=C_PROPRIO, ha="left")

# ---- privileged boundary ----
priv_y = 0.30
ax.add_patch(Rectangle((0.02, priv_y), 0.30, 0.055, fc="#F7F7F7", ec=C_PRIV,
                       ls=(0, (3, 2)), lw=0.9, transform=ax.transAxes, zorder=3))
note(0.17, priv_y + 0.0275,
     "privileged (sim only): contact force,\nground μ, slip — critic & gate labels,\nnever in $\\mathbf{o}_t$",
     fs=5.8, color=C_PRIV)

# ================================================================
# RIGHT PANEL: policy architecture  (x in [0.405, 0.995])
# ================================================================
RX0, RX1 = 0.405, 0.995
ax.text(RX0 + 0.055, 0.960,
        "Fast base + Hall-gated bounded residual",
        ha="left", va="center", fontsize=8.2, weight="bold",
        transform=ax.transAxes)

# ---- row 1: frozen shared Hall encoder (3 boxes) ----
enc_y, enc_h = 0.855, 0.060
enc_boxes = [
    (0.000, 0.115, "Hall 1350 + 30\n[T,2,15,3]", C_HALL, "#B8860B"),
    (0.125, 0.190, "frozen shared Hall encoder\npoint MLP → frame MLP → Conv1d", "#FDF0D8", "#B8860B"),
    (0.325, 0.115, "per-foot latent\n32 × 2", "#FDF0D8", "#B8860B"),
]
for dx, w, lab, fc, ec in enc_boxes:
    box(RX0 + dx, enc_y, w, enc_h, fc, ec, lab, fs=5.4)
for i in range(len(enc_boxes) - 1):
    x0 = RX0 + enc_boxes[i][0] + enc_boxes[i][1]
    x1 = RX0 + enc_boxes[i + 1][0]
    arrow(x0 + 0.003, enc_y + enc_h / 2, x1 - 0.003, enc_y + enc_h / 2, lw=0.9)
note(RX0 + 0.010, 0.935, "frozen (reused from speedboost112 teacher)",
     fs=5.8, color=C_FROZEN_E, ha="left")

# ---- row 2: frozen teacher (single box) ----
tc_x0, tc_y0, tc_w, tc_h = RX0 + 0.005, 0.735, RX1 - RX0 - 0.010, 0.085
box(tc_x0, tc_y0, tc_w, tc_h, C_FROZEN, C_FROZEN_E,
    "FROZEN speedboost112 teacher  →  a_base\n(safe / fast / stable mixture, μ̂ traction gate)",
    fs=5.8, weight="bold")
arrow(RX0 + 0.3825, enc_y, RX0 + 0.3825, tc_y0 + tc_h, lw=0.9, color="#555555")

# ---- row 3: features -> gate / residual paths ----
f_x0, f_y0, f_w, f_h = RX0 + 0.005, 0.600, 0.115, 0.085
box(f_x0, f_y0, f_w, f_h, "#F3F3F3", "#AAAAAA", "features 548 =\n480 + 64 + 4", fs=5.2)
arrow(RX0 + 0.0625, tc_y0, RX0 + 0.0625, f_y0 + f_h, lw=0.9, color="#555555")

g_y, g_h = 0.615, 0.075
r_y, r_h = 0.525, 0.075
box(RX0 + 0.150, g_y, 0.240, g_h, "#D8F3EA", C_GATE,
    "Hall gate head: 548→128→32→1 → σ · calib\ng = σ(2.75·logit−3.2) × min(valid_L, valid_R)",
    fs=5.2)
box(RX0 + 0.150, r_y, 0.240, r_h, "#FBE3D5", C_RESID,
    "bounded residual head: 548→256→128→29\nδ = 0.55 · tanh(·)",
    fs=5.2)
arrow(f_x0 + f_w, f_y0 + 0.060, RX0 + 0.148, g_y + g_h / 2, lw=0.9, color=C_GATE)
arrow(f_x0 + f_w, f_y0 + 0.020, RX0 + 0.148, r_y + r_h / 2, lw=0.9, color=C_RESID)

# ---- composition node ----
comp_x, comp_y = RX0 + 0.405, 0.545
box(comp_x, comp_y, 0.100, 0.135, "#FFFFFF", "#333333", None, lw=1.2)
note(comp_x + 0.050, comp_y + 0.096, "$a = a_{base} + g\\delta$", fs=5.8)
note(comp_x + 0.050, comp_y + 0.040, "clamp ±3", fs=4.8, color="#666666")
arrow(RX0 + 0.390, g_y + g_h / 2, comp_x + 0.010, comp_y + 0.100, lw=1.0, color=C_GATE)
arrow(RX0 + 0.390, r_y + r_h / 2, comp_x + 0.010, comp_y + 0.055, lw=1.0, color=C_RESID)
note(RX0 + 0.430, g_y + g_h / 2 + 0.024, "g", fs=6.0, color=C_GATE)
note(RX0 + 0.428, r_y + r_h / 2 - 0.024, "δ", fs=6.0, color=C_RESID)
arrow(RX0 + 0.290, tc_y0, comp_x + 0.085, comp_y + 0.135, lw=1.1, color="#555555")
note(RX0 + 0.262, tc_y0 - 0.017, "a_base", fs=5.6, ha="right", color="#555555")

# ---- output ----
out_x, out_y, out_w, out_h = RX0 + 0.507, 0.565, 0.070, 0.095
box(out_x, out_y, out_w, out_h, "#DCEBF7", C_ACT, "29 joint\nactions\n→ G1", fs=5.2)
arrow(comp_x + 0.100, comp_y + 0.070, out_x, out_y + out_h / 2, lw=1.0, color=C_ACT)

# ---- fault branch ----
box(RX0 + 0.150, 0.452, 0.240, 0.045, "#FDE8E8", C_FAULT, None, lw=0.9, ls=(0, (3, 2)))
note(RX0 + 0.270, 0.4745,
     "foot dropout ⇒ g = 0 ⇒ pure base + speed envelope", fs=5.0, color=C_FAULT)
arrow(RX0 + 0.270, g_y, RX0 + 0.270, 0.499, lw=0.8, color=C_FAULT, ls=":")

# ---- training annotation ----
box(RX0 + 0.005, 0.395, RX1 - RX0 - 0.01, 0.042, "#FBFBFB", "#999999", None,
    ls=(0, (2, 2)), lw=0.8)
note(RX0 + 0.295, 0.416,
     "training only: stage-BCE supervises gate · PPO: gate + residual · anchor ≈ teacher on HIGH",
     fs=5.0, color="#666666")

# ================================================================
# cross-panel connection
# ================================================================
arrow(LX1 + 0.008, bar_y + bar_h / 2, RX0 - 0.008, enc_y + enc_h / 2, lw=1.3,
      color="#444444")
arrow(LX1 + 0.005, bar_y + bar_h / 2 - 0.055, RX0 - 0.008, tc_y0 + tc_h - 0.05,
      lw=1.1, color=C_PROPRIO, ls="-")
note((LX1 + RX0) / 2, bar_y + bar_h / 2 + 0.030, "1864", fs=7.0, weight="bold")

# panel labels
note(LX0 + 0.010, 0.984, "(a)", fs=9, weight="bold", ha="left")
note(RX0 + 0.010, 0.984, "(b)", fs=9, weight="bold", ha="left")

out_pdf = "/home/mosense/guo_1/vola_sensor/vis/paper_figs/fig_policy_arch_style_test.pdf"
out_png = "/home/mosense/guo_1/vola_sensor/vis/paper_figs/fig_policy_arch_style_test.png"
import os
os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.02)
print("saved", out_pdf, out_png)
