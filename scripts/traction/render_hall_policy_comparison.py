#!/usr/bin/env python3
"""Render synchronized Hall data and paired proprio-vs-Hall Isaac videos.

Hall panels contain only B/dB in tesla-derived units.  Contact slip is read
from Isaac CSV solely as an evaluation label and is explicitly marked as not
being a policy input or a quantity measured by the Hall chips.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


FALLBACK_LAYOUT = np.asarray(
    [
        (0.3723, -0.0050),
        (0.3359, 0.0978),
        (0.3336, -0.0030),
        (0.3347, -0.1058),
        (0.2968, -0.0050),
        (0.0449, -0.0130),
        (0.0085, 0.0958),
        (0.0070, -0.0090),
        (0.0097, -0.1168),
        (-0.0290, -0.0110),
        (-0.2813, -0.0030),
        (-0.3189, 0.1028),
        (-0.3197, -0.0060),
        (-0.3193, -0.1028),
        (-0.3553, 0.0030),
    ],
    dtype=np.float32,
)

BG = (20, 24, 31)
PANEL = (31, 37, 47)
GRID = (66, 74, 88)
WHITE = (238, 241, 245)
MUTED = (159, 169, 184)
GREEN = (90, 214, 128)
RED = (92, 96, 236)
CYAN = (230, 203, 74)
BLUE = (238, 133, 64)
AMBER = (45, 166, 244)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hall-video", type=Path, required=True)
    parser.add_argument("--hall-trace", type=Path, required=True)
    parser.add_argument("--hall-timeseries", type=Path, required=True)
    parser.add_argument("--hall-phases", type=Path, required=True)
    parser.add_argument("--baseline-video", type=Path, required=True)
    parser.add_argument("--baseline-timeseries", type=Path, required=True)
    parser.add_argument("--baseline-phases", type=Path, required=True)
    parser.add_argument("--output-hall-video", type=Path, required=True)
    parser.add_argument("--output-comparison-video", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--crf", type=int, default=18)
    return parser.parse_args()


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    result: dict[str, np.ndarray] = {}
    for key in rows[0]:
        try:
            result[key] = np.asarray([float(row[key]) for row in rows])
        except ValueError:
            result[key] = np.asarray([row[key] for row in rows])
    return result


def _aligned_series(
    trace_phase: np.ndarray, table: dict[str, np.ndarray], key: str
) -> np.ndarray:
    output = np.full(trace_phase.shape, np.nan, dtype=np.float32)
    indices = np.flatnonzero(trace_phase >= 0)
    values = np.asarray(table[key], dtype=np.float32)
    count = min(len(indices), len(values))
    output[indices[:count]] = values[:count]
    if count:
        output[: indices[0]] = values[0]
        output[indices[count - 1] + 1 :] = values[count - 1]
    return output


def _phase_name(mu: float, phase: int) -> str:
    if phase < 0:
        return "WARM-UP"
    return f"PHASE {phase + 1}  {'HIGH GRIP' if mu >= 0.75 else 'LOW GRIP'}"


def _put(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = WHITE,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _sole_outline(center: tuple[int, int], half_width: int, height: int) -> np.ndarray:
    # +x (toe) maps upward. Width variation follows the supplied A4 outline.
    shape = np.asarray(
        [
            (0.00, 0.50),
            (-0.60, 0.47),
            (-0.95, 0.35),
            (-1.00, 0.10),
            (-0.58, -0.05),
            (-0.65, -0.38),
            (-0.48, -0.49),
            (0.00, -0.52),
            (0.48, -0.49),
            (0.65, -0.38),
            (0.58, -0.05),
            (1.00, 0.10),
            (0.95, 0.35),
            (0.60, 0.47),
        ],
        dtype=np.float32,
    )
    points = np.empty_like(shape)
    points[:, 0] = center[0] + shape[:, 0] * half_width
    points[:, 1] = center[1] - shape[:, 1] * height
    return np.round(points).astype(np.int32)


def _turbo(value: float) -> tuple[int, int, int]:
    index = int(np.clip(round(value * 255.0), 0, 255))
    return tuple(
        int(x)
        for x in cv2.applyColorMap(
            np.asarray([[index]], dtype=np.uint8), cv2.COLORMAP_TURBO
        )[0, 0]
    )


def _draw_sole(
    image: np.ndarray,
    origin: tuple[int, int],
    size: tuple[int, int],
    positions: np.ndarray,
    values_mt: np.ndarray,
    vector_mt: np.ndarray,
    valid: np.ndarray,
    title: str,
    scale_mt: float,
    mirror_y: bool,
) -> None:
    x0, y0 = origin
    width, height = size
    cv2.rectangle(image, (x0, y0), (x0 + width, y0 + height), PANEL, -1)
    _put(image, title, (x0 + 12, y0 + 24), 0.55, WHITE, 1)
    _put(image, "toe +x", (x0 + width - 78, y0 + 22), 0.38, MUTED, 1)
    center = (x0 + width // 2, y0 + height // 2 + 6)
    outline = _sole_outline(center, int(width * 0.36), int(height * 0.39))
    cv2.fillPoly(image, [outline], (43, 49, 60))
    cv2.polylines(image, [outline], True, (165, 174, 187), 2, cv2.LINE_AA)
    for index, ((toe_x, lateral_y), value, keep) in enumerate(
        zip(positions, values_mt, valid, strict=True)
    ):
        local_y = -lateral_y if mirror_y else lateral_y
        px = int(center[0] - local_y * width * 0.70)
        py = int(center[1] - toe_x * height * 0.78)
        normalized = float(np.clip(value / max(scale_mt, 1.0e-9), 0.0, 1.0))
        color = _turbo(normalized) if keep > 0.5 else (85, 85, 85)
        cv2.circle(image, (px, py), 10, color, -1, cv2.LINE_AA)
        cv2.circle(image, (px, py), 10, WHITE, 1, cv2.LINE_AA)
        _put(image, str(index), (px - 6, py + 4), 0.30, (5, 5, 5), 1)
    base = (x0 + 20, y0 + height - 24)
    _put(
        image,
        f"mean dB [mT]  x={vector_mt[0]:+.3f}  y={vector_mt[1]:+.3f}  z={vector_mt[2]:+.3f}",
        base,
        0.34,
        MUTED,
        1,
    )
    arrow_start = (x0 + width - 45, y0 + height - 34)
    xy = vector_mt[:2]
    xy_norm = float(np.linalg.norm(xy))
    if xy_norm > 1.0e-8:
        direction = xy / xy_norm
        arrow_end = (
            int(arrow_start[0] - 30.0 * direction[1]),
            int(arrow_start[1] - 30.0 * direction[0]),
        )
        cv2.arrowedLine(image, arrow_start, arrow_end, CYAN, 2, cv2.LINE_AA, tipLength=0.3)


def _draw_line_chart(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    x: np.ndarray,
    series: list[tuple[np.ndarray, tuple[int, int, int], str]],
    current: int,
    y_range: tuple[float, float],
    title: str,
    ylabel: str,
    phase: np.ndarray | None = None,
    mu: np.ndarray | None = None,
) -> None:
    x0, y0, width, height = rect
    cv2.rectangle(image, (x0, y0), (x0 + width, y0 + height), PANEL, -1)
    left, right, top, bottom = 54, 12, 28, 30
    plot = (x0 + left, y0 + top, width - left - right, height - top - bottom)
    px0, py0, pw, ph = plot
    if phase is not None and mu is not None:
        last = min(current, len(phase) - 1)
        start = 0
        while start <= last:
            end = start
            while end + 1 <= last and phase[end + 1] == phase[start]:
                end += 1
            color = (36, 71, 42) if mu[start] >= 0.75 else (70, 48, 31)
            xa = px0 + int(start / max(len(x) - 1, 1) * pw)
            xb = px0 + int((end + 1) / max(len(x) - 1, 1) * pw)
            cv2.rectangle(image, (xa, py0), (xb, py0 + ph), color, -1)
            start = end + 1
    for fraction in (0.0, 0.5, 1.0):
        yy = py0 + int((1.0 - fraction) * ph)
        cv2.line(image, (px0, yy), (px0 + pw, yy), GRID, 1)
        value = y_range[0] + fraction * (y_range[1] - y_range[0])
        _put(image, f"{value:.2f}", (x0 + 4, yy + 4), 0.31, MUTED, 1)
    _put(image, title, (x0 + 10, y0 + 20), 0.48, WHITE, 1)
    _put(image, ylabel, (x0 + 4, y0 + height - 8), 0.31, MUTED, 1)
    end = min(current + 1, len(x))
    for series_index, (values, color, label) in enumerate(series):
        if end < 2:
            continue
        clipped = np.asarray(values[:end], dtype=np.float32)
        finite = np.isfinite(clipped)
        indices = np.flatnonzero(finite)
        if len(indices) >= 2:
            xs = px0 + np.round(indices / max(len(x) - 1, 1) * pw).astype(np.int32)
            ys = py0 + np.round(
                (1.0 - np.clip((clipped[indices] - y_range[0]) / (y_range[1] - y_range[0]), 0.0, 1.0)) * ph
            ).astype(np.int32)
            cv2.polylines(
                image,
                [np.column_stack((xs, ys)).reshape(-1, 1, 2)],
                False,
                color,
                2,
                cv2.LINE_AA,
            )
        legend_x = x0 + width - 145
        legend_y = y0 + 18 + 16 * series_index
        cv2.line(image, (legend_x, legend_y - 4), (legend_x + 18, legend_y - 4), color, 2)
        _put(image, label, (legend_x + 23, legend_y), 0.32, color, 1)
    marker_x = px0 + int(min(current, len(x) - 1) / max(len(x) - 1, 1) * pw)
    cv2.line(image, (marker_x, py0), (marker_x, py0 + ph), WHITE, 1)


def _field_panel(
    trace: dict[str, np.ndarray],
    speed: np.ndarray,
    index: int,
    width: int = 640,
    height: int = 720,
) -> np.ndarray:
    image = np.full((height, width, 3), BG, dtype=np.uint8)
    index = min(index, len(trace["time_s"]) - 1)
    delta_mt = trace["delta_tesla"][index] * 1.0e3
    norm_mt = np.linalg.norm(delta_mt, axis=-1)
    scale_mt = float(trace["display_scale_mt"])
    phase = int(trace["phase"][index])
    mu = float(trace["mu"][index])
    _put(image, "HALL MAGNETIC SOLE  |  measured Bx, By, Bz only", (14, 25), 0.53, WHITE, 1)
    _put(
        image,
        f"{_phase_name(mu, phase)}   mu={mu:.2f} (sim label)   t={trace['time_s'][index]:.2f}s",
        (14, 48),
        0.43,
        GREEN if mu >= 0.75 else BLUE,
        1,
    )
    positions = trace["positions"]
    _draw_sole(
        image,
        (10, 62),
        (300, 360),
        positions,
        norm_mt[0],
        delta_mt[0].mean(axis=0),
        trace["valid_mask"][index, 0],
        "LEFT FOOT  P00-P14",
        scale_mt,
        False,
    )
    _draw_sole(
        image,
        (320, 62),
        (310, 360),
        positions,
        norm_mt[1],
        delta_mt[1].mean(axis=0),
        trace["valid_mask"][index, 1],
        "RIGHT FOOT  P00-P14",
        scale_mt,
        bool(trace["mirror_right_y"]),
    )
    component_rms = np.sqrt(
        np.mean((trace["delta_tesla"] * 1.0e3) ** 2, axis=(1, 2))
    )
    ymax = max(float(np.percentile(component_rms, 99.5)) * 1.15, 0.05)
    _draw_line_chart(
        image,
        (10, 432, 620, 236),
        trace["time_s"],
        [
            (component_rms[:, 0], RED, "RMS dBx"),
            (component_rms[:, 1], GREEN, "RMS dBy"),
            (component_rms[:, 2], CYAN, "RMS dBz"),
        ],
        index,
        (0.0, ymax),
        "Three-axis field change across 30 Hall sites",
        "mT",
        trace["phase"],
        trace["mu"],
    )
    _put(
        image,
        f"actual vx={speed[index]:.3f} m/s    color scale |dB|: 0..{scale_mt:.3f} mT",
        (15, 701),
        0.39,
        MUTED,
        1,
    )
    return image


def _comparison_panel(
    trace: dict[str, np.ndarray],
    hall: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    index: int,
    metrics: dict[str, float | int | str],
) -> np.ndarray:
    image = np.full((540, 1920, 3), BG, dtype=np.uint8)
    compact = _field_panel(trace, hall["vx"], index, 700, 540)
    image[:, :700] = compact
    phase = trace["phase"]
    mu = trace["mu"]
    _draw_line_chart(
        image,
        (715, 10, 1188, 245),
        trace["time_s"],
        [
            (hall["vx"], GREEN, "Hall adaptive"),
            (baseline["vx"], RED, "original proprio"),
            (np.full_like(hall["vx"], 0.8), WHITE, "command"),
        ],
        index,
        (-0.05, 1.05),
        "Actual forward velocity under the SAME command and friction sequence",
        "m/s",
        phase,
        mu,
    )
    slip_max = max(
        float(np.nanpercentile(hall["slip"], 99)),
        float(np.nanpercentile(baseline["slip"], 99)),
        0.1,
    )
    _draw_line_chart(
        image,
        (715, 265, 790, 260),
        trace["time_s"],
        [
            (hall["slip"], GREEN, "Hall adaptive"),
            (baseline["slip"], RED, "original proprio"),
        ],
        index,
        (0.0, slip_max * 1.10),
        "Contact slip (Isaac evaluation truth only)",
        "m/s; NOT a Hall/policy input",
        phase,
        mu,
    )
    cv2.rectangle(image, (1520, 265), (1903, 525), PANEL, -1)
    _put(image, "PAIRED RESULT", (1540, 294), 0.62, WHITE, 2)
    _put(image, "low-grip contact slip", (1540, 329), 0.46, MUTED, 1)
    _put(
        image,
        f"{metrics['baseline_low_slip']:.3f} -> {metrics['hall_low_slip']:.3f} m/s",
        (1540, 357),
        0.55,
        GREEN,
        2,
    )
    _put(
        image,
        f"reduction  {metrics['low_slip_reduction_percent']:.1f}%",
        (1540, 385),
        0.56,
        GREEN,
        2,
    )
    _put(image, "low grip: deliberate slow-down", (1540, 421), 0.43, MUTED, 1)
    _put(
        image,
        f"vx {metrics['baseline_low_vx']:.3f} -> {metrics['hall_low_vx']:.3f} m/s",
        (1540, 447),
        0.48,
        CYAN,
        1,
    )
    _put(
        image,
        f"falls: Hall {metrics['hall_falls']} | base {metrics['baseline_falls']}",
        (1540, 480),
        0.46,
        WHITE,
        1,
    )
    _put(image, "policy never reads mu or slip truth", (1540, 509), 0.39, AMBER, 1)
    return image


def _open_video(path: Path) -> tuple[cv2.VideoCapture, float, int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return capture, fps, width, height, frames


def _encode_h264(temporary: Path, output: Path, crf: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(temporary),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    temporary.unlink()


def _make_hall_video(
    video: Path,
    output: Path,
    trace: dict[str, np.ndarray],
    hall: dict[str, np.ndarray],
    crf: int,
) -> None:
    capture, fps, _, _, frame_count = _open_video(video)
    temporary = output.with_name(output.stem + ".temporary.mp4")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 720)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {temporary}")
    for index in range(min(frame_count, len(trace["time_s"]))):
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        panel = _field_panel(trace, hall["vx"], index)
        writer.write(np.hstack((frame, panel)))
    capture.release()
    writer.release()
    _encode_h264(temporary, output, crf)


def _make_comparison_video(
    hall_video: Path,
    baseline_video: Path,
    output: Path,
    trace: dict[str, np.ndarray],
    hall: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    metrics: dict[str, float | int | str],
    crf: int,
) -> None:
    hall_capture, fps, _, _, hall_frames = _open_video(hall_video)
    base_capture, base_fps, _, _, base_frames = _open_video(baseline_video)
    if abs(fps - base_fps) > 1.0e-3:
        raise ValueError(f"video FPS mismatch: {fps} vs {base_fps}")
    temporary = output.with_name(output.stem + ".temporary.mp4")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {temporary}")
    count = min(hall_frames, base_frames, len(trace["time_s"]))
    for index in range(count):
        ok_hall, hall_frame = hall_capture.read()
        ok_base, base_frame = base_capture.read()
        if not ok_hall or not ok_base:
            break
        base_frame = cv2.resize(base_frame, (960, 540), interpolation=cv2.INTER_AREA)
        hall_frame = cv2.resize(hall_frame, (960, 540), interpolation=cv2.INTER_AREA)
        cv2.rectangle(base_frame, (0, 0), (960, 48), (10, 10, 10), -1)
        cv2.rectangle(hall_frame, (0, 0), (960, 48), (10, 10, 10), -1)
        _put(base_frame, "ORIGINAL PROPRIO POLICY  |  Hall ignored", (18, 32), 0.70, WHITE, 2)
        _put(hall_frame, "MAGNETIC-ADAPTIVE POLICY  |  Hall B history used", (18, 32), 0.70, GREEN, 2)
        bottom = _comparison_panel(trace, hall, baseline, index, metrics)
        writer.write(np.vstack((np.hstack((base_frame, hall_frame)), bottom)))
    hall_capture.release()
    base_capture.release()
    writer.release()
    _encode_h264(temporary, output, crf)


def _static_figure(
    output: Path,
    trace: dict[str, np.ndarray],
    hall_phases: dict[str, np.ndarray],
    base_phases: dict[str, np.ndarray],
    metrics: dict[str, float | int | str],
) -> None:
    candidates = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
    cjk = next((p for p in candidates if "NotoSansCJK" in Path(p).name), None)
    if cjk:
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cjk).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    phases = np.asarray(hall_phases["phase"], dtype=int)
    labels = [f"阶段 {p + 1}\nμ={hall_phases['mu'][p]:.1f}" for p in phases]
    x = np.arange(len(phases))
    width = 0.35
    axes[0, 0].bar(x - width / 2, hall_phases["steady_vx"], width, label="磁感知自适应", color="#2ca25f")
    axes[0, 0].bar(x + width / 2, base_phases["steady_vx"], width, label="原本体感策略", color="#de2d26")
    axes[0, 0].axhline(0.8, color="black", linestyle="--", linewidth=1, label="速度指令")
    axes[0, 0].set(xticks=x, xticklabels=labels, ylabel="实际前向速度 (m/s)", title="同一速度指令下的摩擦自适应")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].bar(x - width / 2, hall_phases["steady_contact_slip"], width, label="磁感知自适应", color="#2ca25f")
    axes[0, 1].bar(x + width / 2, base_phases["steady_contact_slip"], width, label="原本体感策略", color="#de2d26")
    axes[0, 1].set(xticks=x, xticklabels=labels, ylabel="接触脚滑移速度 (m/s)", title="低摩擦滑移显著降低（仅仿真评估真值）")
    axes[0, 1].text(0.02, 0.95, f"低摩擦滑移降低 {metrics['low_slip_reduction_percent']:.1f}%", transform=axes[0, 1].transAxes, va="top", color="#147d40", weight="bold")

    positions = trace["positions"]
    phase_index = np.flatnonzero(trace["phase"] == 1)
    low_field = np.linalg.norm(trace["delta_tesla"][phase_index], axis=-1).mean(axis=0) * 1e3
    vmax = max(float(np.percentile(low_field, 99)), 1e-6)
    for foot, marker, offset in ((0, "o", -0.20), (1, "s", 0.20)):
        lateral = positions[:, 1] * (-1.0 if foot == 1 and trace["mirror_right_y"] else 1.0)
        plot_x = offset + lateral
        scatter = axes[1, 0].scatter(plot_x, positions[:, 0], c=low_field[foot], s=180, cmap="turbo", vmin=0.0, vmax=vmax, marker=marker, edgecolors="black", label="左脚" if foot == 0 else "右脚")
        for sensor, (yy, xx) in enumerate(zip(positions[:, 0], lateral, strict=True)):
            axes[1, 0].text(offset + xx, yy, str(sensor), ha="center", va="center", fontsize=7)
    figure.colorbar(scatter, ax=axes[1, 0], label="低摩擦阶段平均 |dB| (mT)")
    axes[1, 0].set(xlabel="足底局部 y（机器人左为正）", ylabel="足底局部 x（脚尖为正）", title="真实 15 点布局上的磁场变化")
    axes[1, 0].set_xlim(-0.38, 0.38)
    axes[1, 0].axvline(0.0, color="#bbbbbb", linewidth=0.8)
    axes[1, 0].legend(frameon=False)
    axes[1, 0].set_aspect("equal", adjustable="box")

    axes[1, 1].axis("off")
    text = (
        "策略信息边界\n\n"
        "磁感知策略输入：本体感觉历史 + 左右脚 15×3 路 Bx/By/Bz 历史、采样周期和健康度。\n\n"
        "原策略输入：仅前 480 维本体感觉；霍尔通道到动作的路径严格为零。\n\n"
        "两者都不读取真实摩擦系数 μ、接触滑移真值或法/切向力。图中的 μ 和滑移只用于 Isaac 离线评价。\n\n"
        f"配对结果：低摩擦滑移 {metrics['baseline_low_slip']:.3f} → {metrics['hall_low_slip']:.3f} m/s，"
        f"无跌倒（磁感知 {metrics['hall_falls']}，基线 {metrics['baseline_falls']}）。\n\n"
        "当前限制：第二次进入高摩擦后的速度恢复仍偏慢，属于下一轮训练优化目标。"
    )
    axes[1, 1].text(0.02, 0.98, text, va="top", fontsize=13, linespacing=1.55, wrap=True)
    figure.suptitle("柔性磁感知足底：磁场观测与摩擦自适应策略配对对比", fontsize=18, weight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    loaded = np.load(args.hall_trace, allow_pickle=False)
    trace = {key: loaded[key] for key in loaded.files}
    trace["positions"] = (
        np.asarray(trace["hall_positions_normalized"], dtype=np.float32)
        if "hall_positions_normalized" in trace
        else FALLBACK_LAYOUT
    )
    trace["mirror_right_y"] = bool(
        trace["mirror_right_y"] if "mirror_right_y" in trace else True
    )
    if trace["delta_tesla"].shape[1:] != (2, 15, 3):
        raise ValueError(
            f"expected Hall trace [T,2,15,3], got {trace['delta_tesla'].shape}"
        )
    if not np.isfinite(trace["delta_tesla"]).all():
        raise ValueError("Hall trace contains NaN/Inf")
    norm_mt = np.linalg.norm(trace["delta_tesla"], axis=-1) * 1.0e3
    trace["display_scale_mt"] = np.asarray(
        max(float(np.percentile(norm_mt, 99.0)), 0.05), dtype=np.float32
    )

    hall_table = _read_csv(args.hall_timeseries)
    base_table = _read_csv(args.baseline_timeseries)
    hall = {
        "vx": _aligned_series(trace["phase"], hall_table, "mean_vx"),
        "slip": _aligned_series(trace["phase"], hall_table, "mean_contact_slip"),
    }
    baseline = {
        "vx": _aligned_series(trace["phase"], base_table, "mean_vx"),
        "slip": _aligned_series(trace["phase"], base_table, "mean_contact_slip"),
    }
    hall_phases = _read_csv(args.hall_phases)
    base_phases = _read_csv(args.baseline_phases)
    low = np.asarray(hall_phases["mu"]) <= 0.25
    high = np.asarray(hall_phases["mu"]) >= 0.75
    hall_low_slip = float(np.mean(hall_phases["steady_contact_slip"][low]))
    base_low_slip = float(np.mean(base_phases["steady_contact_slip"][low]))
    metrics: dict[str, float | int | str] = {
        "paired_seed": int(trace["seed"]) if "seed" in trace else -1,
        "command_vx_mps": float(hall_phases["cmd_vx"][0]),
        "hall_low_vx": float(np.mean(hall_phases["steady_vx"][low])),
        "baseline_low_vx": float(np.mean(base_phases["steady_vx"][low])),
        "hall_high_vx": float(np.mean(hall_phases["steady_vx"][high])),
        "baseline_high_vx": float(np.mean(base_phases["steady_vx"][high])),
        "hall_low_slip": hall_low_slip,
        "baseline_low_slip": base_low_slip,
        "low_slip_reduction_percent": 100.0 * (1.0 - hall_low_slip / max(base_low_slip, 1e-9)),
        "hall_high_slip": float(np.mean(hall_phases["steady_contact_slip"][high])),
        "baseline_high_slip": float(np.mean(base_phases["steady_contact_slip"][high])),
        "hall_falls": int(np.sum(hall_phases["falls"])),
        "baseline_falls": int(np.sum(base_phases["falls"])),
        "hall_observation": "proprio + dual-foot 15x3 Hall history/timing/health",
        "baseline_observation": "original 480-D proprio only; Hall slice ignored",
        "policy_reads_mu_or_slip_truth": False,
    }
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _static_figure(args.output_figure, trace, hall_phases, base_phases, metrics)
    _make_hall_video(
        args.hall_video,
        args.output_hall_video,
        trace,
        hall,
        args.crf,
    )
    _make_comparison_video(
        args.hall_video,
        args.baseline_video,
        args.output_comparison_video,
        trace,
        hall,
        baseline,
        metrics,
        args.crf,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[info] Hall data video: {args.output_hall_video}")
    print(f"[info] paired comparison video: {args.output_comparison_video}")
    print(f"[info] comparison figure: {args.output_figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
