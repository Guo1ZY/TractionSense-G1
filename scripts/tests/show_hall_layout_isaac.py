#!/usr/bin/env python3
"""Isaac GUI inspection scene for the 15-site A4 Hall layout.

This is a layout overlay, not a deformation or force test.  It places the
measured CAD sole under individually named P00..P14 square markers.  Marker XY
is the exact sensor-model XY; only marker Z is lifted above the opaque sole so
the complete layout remains visible from the top.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--frames",
    type=int,
    default=0,
    help="render this many frames; zero keeps the GUI open until the window is closed",
)
parser.add_argument(
    "--marker-lift-mm",
    type=float,
    default=6.5,
    help="display-only marker height above the centred 10 mm CAD sole",
)
parser.add_argument(
    "--report",
    default="logs/hall_foot_sensor/a4_layout_isaac.json",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import json
import math
from pathlib import Path

import isaaclab.sim as sim_utils

from unitree_rl_lab.sensors import (
    DEFAULT_HALL_POSITIONS_IMAGE_PX,
    HALL_LAYOUT_IMAGE_SIZE_PX,
    HALL_LAYOUT_SOLE_BOUNDS_PX,
    HALL_LAYOUT_SOURCE_IMAGE,
    HallFootSensorCfg,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "source" / "unitree_rl_lab" / "unitree_rl_lab"
TPU_USD = PACKAGE_ROOT / "assets" / "meshes" / "tpu_sole_a40_grid35.usd"


def _layout_xy(cfg: HallFootSensorCfg, foot: int) -> list[tuple[float, float]]:
    yaw_deg = cfg.sole_yaw_deg if foot == 0 else cfg.right_sole_yaw_deg
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    result: list[tuple[float, float]] = []
    for x_norm, y_norm in cfg.hall_positions_normalized:
        x = x_norm * cfg.sole_length
        y = y_norm * cfg.sole_width
        if foot == 1 and cfg.mirror_right_y:
            y = -y
        result.append(
            (
                cfg.sole_origin[0] + c * x - s * y,
                cfg.sole_origin[1] + s * x + c * y,
            )
        )
    return result


def _spawn_sole(path: str, position: tuple[float, float, float], color: tuple[float, float, float]) -> None:
    cfg = sim_utils.UsdFileCfg(
        usd_path=str(TPU_USD),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            roughness=0.72,
            opacity=1.0,
        ),
    )
    cfg.func(path, cfg, translation=position)


def _spawn_square(
    path: str,
    position: tuple[float, float, float],
    yaw_deg: float,
    size: tuple[float, float, float],
) -> None:
    yaw = math.radians(yaw_deg)
    cfg = sim_utils.CuboidCfg(
        size=size,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.025, 0.025),
            emissive_color=(0.30, 0.0, 0.0),
            roughness=0.40,
            opacity=1.0,
        ),
    )
    cfg.func(
        path,
        cfg,
        translation=position,
        orientation=(math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)),
    )


def main() -> None:
    if not TPU_USD.is_file():
        raise FileNotFoundError(TPU_USD)
    if args_cli.frames < 0 or args_cli.marker_lift_mm <= 5.0:
        raise ValueError("frames must be non-negative and marker-lift-mm must be above the 5 mm sole top")

    cfg = HallFootSensorCfg(enable_debug_vis=False)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 60.0, render_interval=1, device=args_cli.device)
    )
    sim.set_camera_view(eye=(0.035, -0.001, 0.52), target=(0.035, 0.0, 0.0))
    dome = sim_utils.DomeLightCfg(intensity=2200.0, color=(1.0, 1.0, 1.0))
    dome.func("/World/LayoutLight", dome)

    foot_y = (0.070, -0.070)
    side_names = ("left", "right")
    colors = ((0.06, 0.32, 0.95), (0.06, 0.72, 0.24))
    marker_z = args_cli.marker_lift_mm * 1.0e-3
    report_positions: dict[str, list[dict[str, object]]] = {}
    chip_yaws = cfg.hall_axis_yaw_deg
    if len(chip_yaws) == 1:
        chip_yaws = chip_yaws * cfg.num_hall_sensors

    for foot, (side, y_offset, color) in enumerate(zip(side_names, foot_y, colors, strict=True)):
        _spawn_sole(
            f"/World/HallLayout/{side}/CAD_magnetized_TPU_outline",
            (cfg.sole_origin[0], y_offset + cfg.sole_origin[1], 0.0),
            color,
        )
        report_positions[side] = []
        for index, ((x, y), image_px, chip_yaw) in enumerate(
            zip(_layout_xy(cfg, foot), DEFAULT_HALL_POSITIONS_IMAGE_PX, chip_yaws, strict=True)
        ):
            name = f"P{index:02d}"
            world = (x, y_offset + y, marker_z)
            sole_yaw = cfg.sole_yaw_deg if foot == 0 else cfg.right_sole_yaw_deg
            _spawn_square(
                f"/World/HallLayout/{side}/HallSquares/{name}",
                world,
                sole_yaw + chip_yaw,
                cfg.hall_package_size,
            )
            report_positions[side].append(
                {
                    "name": name,
                    "source_image_uv_px": list(image_px),
                    "normalized_xy": list(cfg.hall_positions_normalized[index]),
                    "foot_local_xy_m": [x, y],
                    "display_world_xyz_m": list(world),
                }
            )

    sim.reset()
    report_path = Path(args_cli.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "format": "hall-a4-layout-isaac-v1",
                "source_image": HALL_LAYOUT_SOURCE_IMAGE,
                "source_image_size_px": list(HALL_LAYOUT_IMAGE_SIZE_PX),
                "sole_ink_bounds_px": list(HALL_LAYOUT_SOLE_BOUNDS_PX),
                "hall_package_size_m": list(cfg.hall_package_size),
                "marker_z_is_display_only": True,
                "foot_order": list(side_names),
                "positions": report_positions,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "stage_root": "/World/HallLayout",
                "left_count": len(report_positions["left"]),
                "right_count": len(report_positions["right"]),
                "marker_shape_m": list(cfg.hall_package_size),
                "note": "marker Z is lifted for inspection; XY is the exact Hall model coordinate",
            },
            ensure_ascii=False,
        )
    )

    # Headless mode is a construction/schema smoke test; no viewport frames
    # are needed after the report has been written.
    if args_cli.headless:
        return
    frames = args_cli.frames
    rendered = 0
    while simulation_app.is_running() and (frames == 0 or rendered < frames):
        sim.step()
        rendered += 1


if __name__ == "__main__":
    main()
    simulation_app.close(skip_cleanup=True)
