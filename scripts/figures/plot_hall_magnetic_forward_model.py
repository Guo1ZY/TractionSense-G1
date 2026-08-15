#!/usr/bin/env python3
"""Draw the Hall-only flexible magnetic sole forward-measurement schematic.

Figure contract
---------------
Core conclusion:
    Contact-induced compression, bending and shear alter four magnet poses;
    a dipole forward model maps those poses to Hall-local Bx/By/Bz and their
    baseline changes, without a magnetic-to-force inverse.
Archetype: schematic-led composite.
Panels: (a) mechanical/field forward chain; (b) absolute field components;
        (c) baseline-relative field changes and measurement boundary.
Output: editable SVG, vector PDF, 600-dpi PNG and numeric source CSV.

The numeric sweep is an illustrative kinematic loading path using the same SI
dipole equation as the simulator.  It is not force calibration data.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Mandatory editable-text rules from the project figure contract.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

import numpy as np
from matplotlib import patches
from matplotlib import font_manager
from matplotlib.path import Path as MplPath


CHINESE_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if CHINESE_FONT_PATH.exists():
    font_manager.fontManager.addfont(str(CHINESE_FONT_PATH))
CHINESE_FONT_FAMILY = (
    font_manager.FontProperties(fname=str(CHINESE_FONT_PATH)).get_name()
    if CHINESE_FONT_PATH.exists()
    else "Droid Sans Fallback"
)


MU0_OVER_4PI = 1.0e-7

COLORS = {
    "rigid": "#5B6472",
    "hall": "#0F4D92",
    "tpu": "#77D7D1",
    "tpu_edge": "#238F86",
    "magnet": "#B9A7E8",
    "magnet_edge": "#6653A9",
    "ground": "#D9D9D9",
    "field_x": "#D9544D",
    "field_y": "#5B7FCA",
    "field_z": "#2E9E68",
    "delta": "#7C6CCF",
    "ink": "#272727",
    "muted": "#767676",
    "warning": "#B64342",
    "panel": "#F7F8FA",
}


def dipole_field(
    hall_position: np.ndarray,
    magnet_positions: np.ndarray,
    magnetic_moments: np.ndarray,
    minimum_distance: float = 5.0e-4,
) -> np.ndarray:
    """Sum four magnetic dipoles at one Hall sample point, in tesla."""

    r = hall_position[None, :] - magnet_positions
    distance = np.linalg.norm(r, axis=1)
    safe_distance = np.maximum(distance, minimum_distance)
    r_hat = r / safe_distance[:, None]
    dot = np.sum(magnetic_moments * r_hat, axis=1)
    contribution = (
        MU0_OVER_4PI
        / safe_distance[:, None] ** 3
        * (3.0 * dot[:, None] * r_hat - magnetic_moments)
    )
    return contribution.sum(axis=0)


def loading_sweep(samples: int = 101) -> dict[str, np.ndarray]:
    """Create an illustrative compression/shear/bending trajectory in SI."""

    half_x = 0.0030
    half_y = 0.0030
    initial_distance = 0.0060
    positions_0 = np.array(
        [
            [-half_x, -half_y, -initial_distance],
            [-half_x, +half_y, -initial_distance],
            [+half_x, -half_y, -initial_distance],
            [+half_x, +half_y, -initial_distance],
        ],
        dtype=np.float64,
    )
    moment_magnitude = 0.0100  # A m^2, simulator engineering default
    hall = np.zeros(3, dtype=np.float64)
    deformation = np.linspace(0.0, 1.0, samples)
    fields = []
    poses = []
    for amount in deformation:
        # Unequal local compression makes bending visible; shear moves all four
        # magnets but includes a small gradient across the patch.
        compression_weights = np.array([0.55, 0.80, 0.70, 1.00])
        shear_weights = np.array([0.75, 0.90, 1.05, 1.20])
        position = positions_0.copy()
        position[:, 0] += amount * 0.0012 * shear_weights
        position[:, 1] += amount * 0.00035 * np.array([-1.0, 1.0, -1.0, 1.0])
        position[:, 2] += amount * 0.0016 * compression_weights

        # Local bending rotates the embedded magnetization direction.  The
        # common rotation plus a small corner-dependent term is purely a pose
        # update; no load-to-field inverse is used.
        angles = amount * np.deg2rad(np.array([5.0, 7.0, 9.0, 11.0]))
        moments = np.column_stack(
            (
                moment_magnitude * np.sin(angles),
                np.zeros(4),
                moment_magnitude * np.cos(angles),
            )
        )
        fields.append(dipole_field(hall, position, moments))
        poses.append(position)
    fields_array = np.asarray(fields)
    return {
        "deformation": deformation,
        "field": fields_array,
        "delta": fields_array - fields_array[0],
        "position_initial": positions_0,
        "position_final": np.asarray(poses)[-1],
    }


def arrow(ax, start, end, color, width=1.4, style="-|>", zorder=5, **kwargs):
    item = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=10,
        linewidth=width,
        color=color,
        zorder=zorder,
        **kwargs,
    )
    ax.add_patch(item)
    return item


def rounded_box(ax, xy, width, height, facecolor, edgecolor, radius=0.08, **kwargs):
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.1,
        **kwargs,
    )
    ax.add_patch(box)
    return box


def draw_state(ax, origin_x: float, deformed: bool) -> None:
    """Draw one side-view Hall/TPU/magnet state in schematic coordinates."""

    width = 2.55
    rigid_y = 3.15
    ground_y = 0.28
    ax.add_patch(
        patches.Rectangle(
            (origin_x, rigid_y),
            width,
            0.34,
            facecolor=COLORS["rigid"],
            edgecolor=COLORS["ink"],
            linewidth=0.8,
            zorder=2,
        )
    )
    ax.text(
        origin_x + width / 2,
        rigid_y + 0.17,
        "刚性足底",
        ha="center",
        va="center",
        color="white",
        fontsize=7.5,
        fontweight="bold",
    )
    rounded_box(
        ax,
        (origin_x + 1.02, 2.72),
        0.50,
        0.26,
        COLORS["hall"],
        COLORS["hall"],
        radius=0.04,
        zorder=5,
    )
    ax.plot(
        origin_x + 1.27,
        2.70,
        marker="o",
        markersize=3.4,
        color="white",
        markeredgecolor=COLORS["hall"],
        zorder=7,
    )
    ax.text(
        origin_x + 0.97,
        2.85,
        "Hall元件/采样点",
        ha="right",
        va="center",
        fontsize=7.0,
        color=COLORS["hall"],
        fontweight="bold",
    )

    ax.add_patch(
        patches.Rectangle(
            (origin_x - 0.05, ground_y - 0.22),
            width + 0.10,
            0.22,
            facecolor=COLORS["ground"],
            edgecolor=COLORS["muted"],
            linewidth=0.8,
            hatch="////",
            zorder=1,
        )
    )
    ax.text(
        origin_x + width / 2,
        ground_y - 0.11,
        "地面",
        ha="center",
        va="center",
        fontsize=6.7,
        color=COLORS["muted"],
    )

    if not deformed:
        tpu_top = 2.25
        tpu_bottom = 0.38
        rounded_box(
            ax,
            (origin_x, tpu_bottom),
            width,
            tpu_top - tpu_bottom,
            COLORS["tpu"],
            COLORS["tpu_edge"],
            radius=0.10,
            alpha=0.74,
            zorder=2,
        )
        magnet_centres = [
            (origin_x + 0.76, 1.88),
            (origin_x + 1.76, 1.88),
        ]
    else:
        x0 = origin_x
        verts = [
            (x0, 0.38),
            (x0 + width, 0.38),
            (x0 + width, 1.82),
            (x0 + 2.10, 1.96),
            (x0 + 1.55, 2.21),
            (x0 + 0.95, 2.08),
            (x0, 1.86),
            (x0, 0.38),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.LINETO,
            MplPath.LINETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CLOSEPOLY,
        ]
        ax.add_patch(
            patches.PathPatch(
                MplPath(verts, codes),
                facecolor=COLORS["tpu"],
                edgecolor=COLORS["tpu_edge"],
                linewidth=1.2,
                alpha=0.74,
                zorder=2,
            )
        )
        magnet_centres = [
            (origin_x + 0.93, 1.84),
            (origin_x + 1.98, 2.00),
        ]

    ax.text(
        origin_x + 0.12,
        0.62,
        "柔性TPU层（内嵌磁片）",
        ha="left",
        va="center",
        fontsize=7.1,
        color=COLORS["tpu_edge"],
        fontweight="bold",
    )
    for index, (mx, my) in enumerate(magnet_centres):
        ellipse = patches.Ellipse(
            (mx, my),
            0.40,
            0.20,
            angle=8.0 * index if deformed else 0.0,
            facecolor=COLORS["magnet"],
            edgecolor=COLORS["magnet_edge"],
            linewidth=1.0,
            zorder=6,
        )
        ax.add_patch(ellipse)
        arrow(
            ax,
            (mx, my),
            (mx + (0.09 if deformed else 0.0), my + 0.40),
            COLORS["magnet_edge"],
            width=1.0,
            zorder=7,
        )
        ax.plot(
            [mx, origin_x + 1.27],
            [my + 0.10, 2.70],
            color=COLORS["magnet_edge"],
            linewidth=0.7,
            linestyle=(0, (2, 2)),
            alpha=0.75,
            zorder=4,
        )
    ax.text(
        origin_x + width - 0.08,
        1.04,
        "磁片完整嵌入TPU（侧视2/4）\n共享局部材料形变、无独立自由度",
        ha="right",
        va="center",
        fontsize=5.8,
        color=COLORS["magnet_edge"],
    )

    if deformed:
        arrow(
            ax,
            (origin_x + 1.30, 0.10),
            (origin_x + 1.30, 0.62),
            COLORS["warning"],
            width=1.5,
        )
        arrow(
            ax,
            (origin_x + 0.25, 1.40),
            (origin_x + 0.78, 1.40),
            COLORS["warning"],
            width=1.5,
        )
        ax.text(
            origin_x + 0.18,
            1.18,
            "压缩",
            ha="left",
            color=COLORS["warning"],
            fontsize=6.8,
            fontweight="bold",
        )
        ax.text(
            origin_x + 0.50,
            1.56,
            "剪切",
            ha="center",
            color=COLORS["warning"],
            fontsize=6.8,
            fontweight="bold",
        )
        ax.add_patch(
            patches.Arc(
                (origin_x + 1.72, 2.17),
                0.72,
                0.55,
                theta1=25,
                theta2=145,
                color=COLORS["warning"],
                linewidth=1.2,
            )
        )
        ax.text(
            origin_x + 1.71,
            2.47,
            "弯曲/转角",
            ha="center",
            color=COLORS["warning"],
            fontsize=6.8,
            fontweight="bold",
        )


def draw_forward_chain(ax) -> None:
    ax.set_xlim(-0.15, 11.55)
    ax.set_ylim(-0.10, 4.10)
    ax.set_axis_off()
    ax.set_facecolor("white")
    draw_state(ax, 0.0, deformed=False)
    draw_state(ax, 3.35, deformed=True)
    ax.text(1.27, 3.78, "零载基线状态", ha="center", fontsize=8.5, fontweight="bold")
    ax.text(4.62, 3.78, "地面接触后的局部形变", ha="center", fontsize=8.5, fontweight="bold")
    arrow(ax, (2.72, 2.02), (3.20, 2.02), COLORS["ink"], width=1.3)

    rounded_box(
        ax,
        (6.55, 0.34),
        4.70,
        3.18,
        "#F5F1FC",
        COLORS["magnet_edge"],
        radius=0.12,
        zorder=1,
    )
    ax.text(
        8.90,
        3.26,
        "磁场正向计算（SI单位）",
        ha="center",
        fontsize=8.6,
        fontweight="bold",
        color=COLORS["magnet_edge"],
    )
    ax.text(
        6.83,
        2.75,
        r"$\mathbf{r}_{j}=\mathbf{p}_{Hall}-\mathbf{p}_{mag,j}$",
        fontsize=8.0,
        color=COLORS["ink"],
    )
    ax.text(
        6.83,
        2.15,
        r"$\mathbf{B}_{j}=\frac{\mu_0}{4\pi r_j^3}"
        r"[3(\mathbf{m}_j\cdot\hat{\mathbf{r}}_j)\hat{\mathbf{r}}_j-\mathbf{m}_j]$",
        fontsize=7.6,
        color=COLORS["ink"],
    )
    ax.text(
        6.83,
        1.57,
        r"$\mathbf{B}_{Hall}=\sum_{j=1}^{4}\mathbf{B}_{j}$",
        fontsize=8.2,
        color=COLORS["ink"],
        fontweight="bold",
    )
    rounded_box(
        ax,
        (6.84, 0.67),
        4.08,
        0.55,
        "white",
        COLORS["hall"],
        radius=0.06,
        zorder=2,
    )
    ax.text(
        7.10,
        0.94,
        "输出：",
        ha="left",
        va="center",
        fontsize=8.0,
        color=COLORS["hall"],
        fontweight="bold",
    )
    ax.text(
        7.92,
        0.94,
        r"$B_x, B_y, B_z, |B|, \mathbf{B}_0, \Delta\mathbf{B}$",
        ha="left",
        va="center",
        fontsize=8.0,
        color=COLORS["hall"],
        fontweight="bold",
    )
    arrow(ax, (6.05, 2.02), (6.48, 2.02), COLORS["ink"], width=1.3)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.06,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=COLORS["ink"],
    )


def build_figure(output_base: Path, samples: int = 101) -> list[Path]:
    data = loading_sweep(samples)
    deformation = data["deformation"]
    field_mt = data["field"] * 1.0e3
    delta_mt = data["delta"] * 1.0e3

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                CHINESE_FONT_FAMILY,
                "Arial",
                "DejaVu Sans",
                "Liberation Sans",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    fig = plt.figure(figsize=(7.2, 5.35), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.70, 1.0],
        hspace=0.28,
        wspace=0.28,
        left=0.065,
        right=0.985,
        top=0.955,
        bottom=0.13,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    draw_forward_chain(ax_a)
    add_panel_label(ax_a, "a")

    component_colors = [COLORS["field_x"], COLORS["field_y"], COLORS["field_z"]]
    component_labels = [r"$B_x$", r"$B_y$", r"$B_z$"]
    for axis, color, label in zip(range(3), component_colors, component_labels):
        ax_b.plot(
            deformation,
            field_mt[:, axis],
            color=color,
            linewidth=1.8,
            label=label,
        )
        ax_c.plot(
            deformation,
            delta_mt[:, axis],
            color=color,
            linewidth=1.8,
            label=rf"$\Delta B_{{{'xyz'[axis]}}}$",
        )
    ax_b.set_xlabel("归一化形变路径 λ（压缩+剪切+弯曲）")
    ax_b.set_ylabel("Hall局部磁场 B（mT）")
    ax_b.set_title("绝对三轴磁场", loc="left", fontsize=8.4, fontweight="bold")
    ax_b.axvline(0.0, color=COLORS["muted"], linewidth=0.8, linestyle="--")
    ax_b.text(
        0.03,
        0.94,
        r"$\mathbf{B}_0$",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=COLORS["muted"],
    )
    ax_b.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    ax_b.legend(ncol=3, loc="best", handlelength=1.5, columnspacing=0.9)
    add_panel_label(ax_b, "b")

    ax_c.axhline(0.0, color=COLORS["muted"], linewidth=0.8, linestyle="--")
    ax_c.set_xlabel("归一化形变路径 λ（压缩+剪切+弯曲）")
    ax_c.set_ylabel("相对基线变化 ΔB（mT）")
    ax_c.set_title("零载基线相对变化", loc="left", fontsize=8.4, fontweight="bold")
    ax_c.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    ax_c.legend(ncol=3, loc="best", handlelength=1.5, columnspacing=0.8)
    add_panel_label(ax_c, "c")

    fig.text(
        0.50,
        0.040,
        "测量边界：霍尔元件只提供三轴磁场及其基线变化；策略可学习时序风险，但不将磁场反解为法向力或切向力。",
        ha="center",
        va="center",
        fontsize=7.5,
        color=COLORS["warning"],
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#FFF4F2",
            "edgecolor": "#E9A6A1",
            "linewidth": 0.8,
        },
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in (".svg", ".pdf"):
        path = output_base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    png_path = output_base.with_suffix(".png")
    tiff_path = output_base.with_suffix(".tiff")
    fig.savefig(
        png_path, dpi=600, bbox_inches="tight", facecolor="white"
    )
    fig.savefig(
        tiff_path, dpi=600, bbox_inches="tight", facecolor="white"
    )
    outputs.extend((png_path, tiff_path))
    plt.close(fig)

    csv_path = output_base.with_name(output_base.name + "_source_data").with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "deformation_lambda",
                "Bx_T",
                "By_T",
                "Bz_T",
                "dBx_T",
                "dBy_T",
                "dBz_T",
            ]
        )
        for index, amount in enumerate(deformation):
            writer.writerow(
                [
                    f"{amount:.8g}",
                    *[f"{value:.10g}" for value in data["field"][index]],
                    *[f"{value:.10g}" for value in data["delta"][index]],
                ]
            )
    outputs.append(csv_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "doc/figures/hall_magnetic_forward_model"
            / "hall_magnetic_forward_model"
        ),
        help="Output path without extension.",
    )
    parser.add_argument("--samples", type=int, default=101)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples < 3:
        raise ValueError("--samples must be at least three")
    for path in build_figure(args.output, args.samples):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
