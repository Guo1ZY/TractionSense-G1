#!/usr/bin/env python3
"""Visualize Hall/TPU structured randomization and fail-safe fallback.

Figure contract
---------------
Core conclusion:
    Structured physical and acquisition randomization makes the Hall policy
    use stable spatiotemporal relations instead of an ideal absolute field;
    confidence-gated residual control falls back exactly to the proprioceptive
    baseline and a conservative command governor when a foot sensor is lost.
Archetype: schematic-led composite with a fault-injection time series.
Outputs: editable SVG, vector PDF, 600-dpi PNG/TIFF and source-data CSV.

The traces are deterministic explanatory signals, not experimental results.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches
import numpy as np


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
plt.rcParams["axes.linewidth"] = 0.7

CHINESE_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if CHINESE_FONT_PATH.exists():
    font_manager.fontManager.addfont(str(CHINESE_FONT_PATH))
CHINESE_FONT_FAMILY = (
    font_manager.FontProperties(fname=str(CHINESE_FONT_PATH)).get_name()
    if CHINESE_FONT_PATH.exists()
    else "Droid Sans Fallback"
)

COLORS = {
    "magnetic": "#7C6CCF",
    "mechanical": "#238F86",
    "electronics": "#D6923B",
    "fault": "#C54F4F",
    "hall": "#0F4D92",
    "baseline": "#5B6472",
    "residual": "#2E9E68",
    "governor": "#D06A4B",
    "ink": "#252525",
    "muted": "#747474",
    "panel": "#F7F8FA",
    "grid": "#D8DCE2",
    "safe": "#DDEFE8",
}


def rounded_box(ax, xy, width, height, facecolor, edgecolor, radius=0.04, **kwargs):
    item = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0,
        **kwargs,
    )
    ax.add_patch(item)
    return item


def arrow(ax, start, end, color=None, width=1.25, mutation=10, **kwargs):
    item = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=width,
        color=color or COLORS["ink"],
        shrinkA=1.5,
        shrinkB=1.5,
        **kwargs,
    )
    ax.add_patch(item)
    return item


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.045,
        1.035,
        label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="top",
        ha="left",
        color=COLORS["ink"],
    )


def explanatory_signals(samples: int = 481) -> dict[str, np.ndarray]:
    """Construct a deterministic Hall packet-loss demonstration."""

    time = np.linspace(0.0, 8.0, samples)
    gait = np.sin(2.0 * np.pi * 1.45 * time)
    slow = np.sin(2.0 * np.pi * 0.23 * time + 0.7)
    bz = 0.38 * gait + 0.10 * slow
    valid = np.ones_like(time)
    valid[(time >= 3.55) & (time <= 5.45)] = 0.0

    # The raw Hall residual is intentionally non-zero during the lost packet
    # interval.  Multiplication by confidence proves exact fail-safe gating.
    raw_residual = 0.24 * np.tanh(1.7 * bz + 0.18 * np.sin(2.0 * np.pi * 0.4 * time))
    confidence = valid.copy()
    recovery = (time > 5.45) & (time < 6.15)
    confidence[recovery] = (time[recovery] - 5.45) / (6.15 - 5.45)
    gated_residual = confidence * raw_residual
    speed_limit = 0.90 * np.ones_like(time)
    speed_limit[valid == 0.0] = 0.10
    speed_limit[recovery] = 0.10 + 0.80 * confidence[recovery]
    return {
        "time_s": time,
        "hall_bz_normalized": bz,
        "packet_valid": valid,
        "confidence": confidence,
        "raw_hall_residual": raw_residual,
        "gated_hall_residual": gated_residual,
        "forward_speed_limit_mps": speed_limit,
    }


def draw_randomization_panel(ax) -> None:
    panel_label(ax, "a")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(
        0.05,
        9.55,
        "结构化域随机化：覆盖真实柔性磁足底的不确定性",
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
    )

    groups = [
        (
            7.25,
            COLORS["magnetic"],
            "磁体与安装",
            "磁矩强度/方向\n2×2间距与厚度\n位置抖动、交叉轴、固定偏置",
        ),
        (
            4.30,
            COLORS["mechanical"],
            "TPU与局部形变",
            "法向/剪切刚度\n阻尼、形变扩散\n压缩、弯曲与剪切耦合",
        ),
        (
            1.35,
            COLORS["electronics"],
            "采样与故障",
            "噪声、量化、温漂、延迟\n丢包、坏通道、采样周期抖动\n整脚掉线与左右不一致",
        ),
    ]
    for y, color, title, body in groups:
        rounded_box(ax, (0.15, y), 5.85, 2.30, "white", color, radius=0.12)
        ax.add_patch(patches.Rectangle((0.15, y), 0.15, 2.30, color=color, lw=0))
        ax.text(0.52, y + 1.82, title, color=color, fontsize=8.3, fontweight="bold", va="top")
        ax.text(0.52, y + 1.40, body, color=COLORS["ink"], fontsize=6.8, va="top", linespacing=1.42)

    # Compact distribution cloud feeding a common sensor packet.
    for row, color in enumerate((COLORS["magnetic"], COLORS["mechanical"], COLORS["electronics"])):
        phase = np.linspace(-np.pi, np.pi, 70)
        values = (0.62 - 0.08 * row) * np.sin(phase) * (0.72 + 0.18 * np.cos(3.0 * phase))
        yy = 8.20 - 2.95 * row + 0.10 * np.sin(5.0 * phase + 0.6 * row)
        ax.scatter(7.00 + values, yy, s=6, color=color, alpha=0.30, edgecolors="none")
        ax.plot([6.25, 7.75], [yy.mean(), yy.mean()], color=color, lw=1.4)
        arrow(ax, (7.95, yy.mean()), (8.55, 5.03), color=color, width=1.0, mutation=8)
    rounded_box(ax, (8.25, 3.55), 1.55, 2.60, COLORS["panel"], COLORS["hall"], radius=0.15)
    ax.text(9.02, 5.78, "随机化样本", ha="center", va="top", fontsize=7.4, color=COLORS["hall"], fontweight="bold")
    ax.text(9.02, 4.78, "Bx, By, Bz\n时间戳/有效位\n左脚 + 右脚", ha="center", va="center", fontsize=6.8, color=COLORS["ink"], linespacing=1.55)
    ax.text(6.30, 0.50, "不把 Hall 反演为 Fn/Ft", color=COLORS["fault"], fontsize=7.3, fontweight="bold")


def draw_spatiotemporal_panel(ax) -> None:
    panel_label(ax, "b")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.05, 5.70, "学习稳定的空间—时间关系，而非理想磁场绝对值", fontsize=9.2, fontweight="bold", va="top")

    # Five by three footprint with three history layers.
    base = np.array(
        [
            [0.90, 0.75], [1.65, 0.62], [2.38, 0.78],
            [0.73, 1.52], [1.55, 1.43], [2.45, 1.55],
            [0.63, 2.32], [1.53, 2.23], [2.55, 2.36],
            [0.78, 3.16], [1.60, 3.04], [2.45, 3.17],
            [1.00, 3.94], [1.68, 3.84], [2.28, 3.98],
        ]
    )
    for history_index, alpha in enumerate((0.22, 0.42, 0.95)):
        shift = 0.22 * (2 - history_index)
        ax.add_patch(
            patches.FancyBboxPatch(
                (0.30 + shift, 0.70 + shift), 2.75, 3.72,
                boxstyle="round,pad=0.03,rounding_size=0.36",
                facecolor="white", edgecolor=COLORS["hall"], lw=0.8, alpha=alpha,
            )
        )
        ax.scatter(base[:, 0] + shift, base[:, 1] + shift, s=26, color=COLORS["hall"], alpha=alpha, zorder=3)
    ax.text(1.76, 0.28, "双足 15×3 Hall 历史  B(t−H:t)", fontsize=7.4, ha="center", color=COLORS["hall"], fontweight="bold")

    rounded_box(ax, (4.10, 1.25), 2.05, 2.70, "#EAF1FA", COLORS["hall"], radius=0.16)
    ax.text(5.12, 3.52, "共享空间编码器", ha="center", fontsize=7.7, fontweight="bold", color=COLORS["hall"])
    ax.text(5.12, 2.63, "站点邻域\n左右镜像一致性\n跨轴相关", ha="center", va="center", fontsize=7.0, linespacing=1.55)
    arrow(ax, (3.28, 2.58), (4.08, 2.58), COLORS["hall"])

    rounded_box(ax, (7.00, 1.25), 2.10, 2.70, "#EDF7F4", COLORS["mechanical"], radius=0.16)
    ax.text(8.05, 3.52, "时间融合", ha="center", fontsize=7.7, fontweight="bold", color=COLORS["mechanical"])
    ax.text(8.05, 2.63, "多帧接触演化\n压缩/剪切模式\n采样健康度", ha="center", va="center", fontsize=7.0, linespacing=1.55)
    arrow(ax, (6.17, 2.58), (6.98, 2.58), COLORS["mechanical"])

    rounded_box(ax, (9.90, 1.25), 1.82, 2.70, "#F8F1E8", COLORS["electronics"], radius=0.16)
    ax.text(10.81, 3.52, "策略输出", ha="center", fontsize=7.7, fontweight="bold", color=COLORS["electronics"])
    ax.text(10.81, 2.64, "Δa_Hall\np(low traction)\nconfidence c", ha="center", va="center", fontsize=7.0, linespacing=1.55)
    arrow(ax, (9.12, 2.58), (9.88, 2.58), COLORS["electronics"])
    ax.text(7.92, 0.42, "部署侧只使用 Hall + 本体感觉历史", fontsize=7.5, ha="center", color=COLORS["ink"], fontweight="bold")


def draw_gate_panel(ax) -> None:
    panel_label(ax, "c")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.05, 5.70, "置信度门控的残差控制与保守调速", fontsize=9.2, fontweight="bold", va="top")

    rounded_box(ax, (0.20, 3.55), 2.25, 1.22, "#EEF0F3", COLORS["baseline"], radius=0.14)
    ax.text(1.32, 4.16, "本体感知基线\na_base", ha="center", va="center", fontsize=7.8, color=COLORS["baseline"], fontweight="bold")
    rounded_box(ax, (0.20, 1.10), 2.25, 1.22, "#EDF7F4", COLORS["residual"], radius=0.14)
    ax.text(1.32, 1.71, "Hall 残差策略\nΔa_Hall", ha="center", va="center", fontsize=7.8, color=COLORS["residual"], fontweight="bold")
    ax.text(2.94, 1.71, "× c", fontsize=10.5, ha="center", va="center", color=COLORS["hall"], fontweight="bold")
    arrow(ax, (2.47, 1.71), (2.69, 1.71), COLORS["residual"])
    rounded_box(ax, (3.35, 2.05), 2.90, 1.85, "white", COLORS["ink"], radius=0.12)
    ax.text(4.80, 3.28, "a = clip(a_base + c·Δa_Hall)", ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax.text(4.80, 2.55, "c∈[0,1]\n由有效位、延迟与网络置信度决定", ha="center", va="center", fontsize=6.4, color=COLORS["muted"], linespacing=1.4)
    arrow(ax, (2.47, 4.16), (3.32, 3.50), COLORS["baseline"])
    arrow(ax, (3.18, 1.71), (3.32, 2.40), COLORS["residual"])

    rounded_box(ax, (7.15, 3.55), 2.60, 1.22, COLORS["safe"], COLORS["mechanical"], radius=0.14)
    ax.text(8.45, 4.16, "Hall 正常：允许自适应\n速度 / 加速度 / 转向", ha="center", va="center", fontsize=7.1, color=COLORS["mechanical"], fontweight="bold")
    rounded_box(ax, (7.15, 1.10), 2.60, 1.22, "#FBECEC", COLORS["fault"], radius=0.14)
    ax.text(8.45, 1.71, "整脚失联：c=0\na=a_base；governor 保守", ha="center", va="center", fontsize=7.1, color=COLORS["fault"], fontweight="bold")
    arrow(ax, (6.28, 3.38), (7.12, 4.16), COLORS["mechanical"])
    arrow(ax, (6.28, 2.50), (7.12, 1.71), COLORS["fault"])
    ax.text(5.05, 0.38, "故障降级不需要、也不执行 Hall→力反演", ha="center", fontsize=7.2, color=COLORS["fault"], fontweight="bold")


def draw_fault_timeseries(ax, data: dict[str, np.ndarray]) -> None:
    panel_label(ax, "d")
    time = data["time_s"]
    ax.axvspan(3.55, 5.45, color=COLORS["fault"], alpha=0.10, lw=0)
    ax.plot(time, data["hall_bz_normalized"], color=COLORS["hall"], lw=1.05, label="Hall Bz（归一化）")
    ax.plot(time, data["confidence"], color=COLORS["magnetic"], lw=1.25, label="置信度 c")
    ax.plot(time, data["raw_hall_residual"], color=COLORS["residual"], lw=0.9, alpha=0.45, ls="--", label="原始 Δa_Hall")
    ax.plot(time, data["gated_hall_residual"], color=COLORS["residual"], lw=1.35, label="门控 c·Δa_Hall")
    ax.plot(time, data["forward_speed_limit_mps"], color=COLORS["governor"], lw=1.35, label="前进限速 (m/s)")
    ax.text(4.50, 1.09, "整脚掉线", ha="center", va="top", fontsize=7.1, color=COLORS["fault"], fontweight="bold")
    ax.annotate(
        "残差精确归零\n回退到 a_base",
        xy=(4.45, 0.0),
        xytext=(5.00, -0.58),
        fontsize=6.8,
        color=COLORS["fault"],
        arrowprops=dict(arrowstyle="->", lw=0.8, color=COLORS["fault"]),
    )
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(-0.75, 1.18)
    ax.set_xlabel("时间 (s)", fontsize=7.4)
    ax.set_ylabel("归一化信号 / 限速", fontsize=7.4)
    ax.set_title("故障注入示例：信号丢失时策略残差与调速器同步降级", loc="left", fontsize=9.2, fontweight="bold", pad=7)
    ax.grid(True, color=COLORS["grid"], lw=0.45, alpha=0.75)
    ax.tick_params(labelsize=6.8, width=0.6, length=2.5)
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.90, edgecolor="none", fontsize=5.8, handlelength=1.6, columnspacing=0.8)


def save_source_data(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(data)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(keys)
        writer.writerows(zip(*(data[key] for key in keys), strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("doc/figures/hall_structured_randomization_fallback"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = explanatory_signals()
    fig = plt.figure(figsize=(7.20, 6.45), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(0.93, 1.17),
        height_ratios=(1.02, 0.98),
        left=0.055,
        right=0.985,
        top=0.945,
        bottom=0.075,
        wspace=0.18,
        hspace=0.24,
    )
    left_top = grid[0, 0].subgridspec(1, 1)
    ax_a = fig.add_subplot(left_top[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    left_bottom = grid[1, 0].subgridspec(1, 1)
    ax_c = fig.add_subplot(left_bottom[0, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    for ax in (ax_a, ax_b, ax_c):
        ax.set_facecolor("white")
    draw_randomization_panel(ax_a)
    draw_spatiotemporal_panel(ax_b)
    draw_gate_panel(ax_c)
    draw_fault_timeseries(ax_d, data)

    fig.text(
        0.055,
        0.985,
        "面向不准确柔性磁感知足底的结构化随机化与故障安全策略",
        ha="left",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        color=COLORS["ink"],
        fontfamily=CHINESE_FONT_FAMILY,
    )
    for text_item in fig.findobj(match=plt.Text):
        text_item.set_fontfamily(CHINESE_FONT_FAMILY)

    stem = args.output_dir / "hall_structured_randomization_fallback"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    save_source_data(stem.with_name(f"{stem.name}_source_data.csv"), data)
    print(stem)


if __name__ == "__main__":
    main()
