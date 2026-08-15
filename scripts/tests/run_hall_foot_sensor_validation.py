#!/usr/bin/env python3
"""Independent fixed-foot rigid-platen validation for the Hall sole.

The platen pose is prescribed and converted into a deterministic local TPU
deformation field.  This isolates magnetic signs, layout locality, filtering,
mirror symmetry, and unloading before a tetrahedral TPU mesh is introduced.
The GUI shows the fixed rigid sole, TPU envelope, platen, Hall sites, four
magnets/site, connection lines, magnetic arrows, and compression colors.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Validate the dual-foot Hall magnetic model with a rigid platen sweep.")
parser.add_argument("--output", default="logs/hall_foot_sensor/platen_validation.csv")
parser.add_argument("--hold-frames", type=int, default=4)
parser.add_argument("--max-displacement-mm", type=float, default=2.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import csv
import json
import math
import os
from pathlib import Path
import sys
import traceback

import torch
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils

from unitree_rl_lab.sensors import HallFootSensor, HallFootSensorCfg


def _spawn_scene(cfg: HallFootSensorCfg) -> tuple[sim_utils.SimulationContext, tuple[UsdGeom.XformOp, UsdGeom.XformOp]]:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=cfg.sensor_period, device=args_cli.device))
    sim.set_camera_view(eye=(0.36, -0.32, 0.24), target=(0.035, 0.0, -0.045))
    sim_utils.DomeLightCfg(intensity=1800.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=1800.0))
    sim_utils.CuboidCfg(
        size=(cfg.sole_length, cfg.sole_width, cfg.sole_thickness),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.20, 0.24)),
    ).func(
        "/World/FixedFoot/RigidSole",
        sim_utils.CuboidCfg(
            size=(cfg.sole_length, cfg.sole_width, cfg.sole_thickness),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.20, 0.24)),
        ),
        translation=(cfg.sole_origin[0], cfg.sole_origin[1], cfg.hall_height + 0.5 * cfg.sole_thickness),
    )
    tpu_center_z = (
        cfg.hall_height
        - cfg.hall_to_tpu_top_distance
        - 0.5 * cfg.tpu_thickness
    )
    tpu_cfg = sim_utils.CuboidCfg(
        size=(cfg.sole_length, cfg.sole_width, cfg.tpu_thickness),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.12, 0.55), opacity=0.35),
    )
    tpu_cfg.func(
        "/World/FixedFoot/TpuEnvelope",
        tpu_cfg,
        translation=(cfg.sole_origin[0], cfg.sole_origin[1], tpu_center_z),
    )
    platen_cfg = sim_utils.CuboidCfg(
        size=(0.075, 0.060, 0.008),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.68, 0.72)),
    )
    platen_z = tpu_center_z - 0.5 * cfg.tpu_thickness - 0.006
    platen_cfg.func(
        "/World/Platen",
        platen_cfg,
        translation=(cfg.sole_origin[0], cfg.sole_origin[1], platen_z),
    )
    sim.reset()
    platen_xform = UsdGeom.Xformable(sim_utils.get_current_stage().GetPrimAtPath("/World/Platen"))
    operations = platen_xform.GetOrderedXformOps()
    translate_op = next(op for op in operations if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
    orient_op = next(op for op in operations if op.GetOpType() == UsdGeom.XformOp.TypeOrient)
    return sim, (translate_op, orient_op)


def _set_platen_pose(
    operations: tuple[UsdGeom.XformOp, UsdGeom.XformOp],
    xyz: tuple[float, float, float],
    tilt_deg: tuple[float, float],
) -> None:
    translate_op, orient_op = operations
    translate_op.Set(Gf.Vec3d(*xyz))
    roll, pitch = (math.radians(value) for value in tilt_deg)
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    orient_op.Set(Gf.Quatd(cr * cp, Gf.Vec3d(sr * cp, cr * sp, -sr * sp)))


def _platen_deformation(
    sensor: HallFootSensor,
    center_indices: tuple[int, int],
    normal_displacement: float,
    shear_xy: tuple[float, float] = (0.0, 0.0),
    tilt_xy: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    deformation = torch.zeros((1, 2, sensor.num_sensors, 6), device=sensor.device)
    sites = sensor.hall_positions_f
    centers = torch.stack((sites[0, center_indices[0]], sites[1, center_indices[1]]), dim=0)
    delta = sites[..., :2] - centers[:, None, :2]
    weight = torch.exp(-0.5 * torch.sum(delta * delta, dim=-1) / 0.030**2)
    tilt_x, tilt_y = tilt_xy
    plane = normal_displacement + math.tan(tilt_y) * delta[..., 0] - math.tan(tilt_x) * delta[..., 1]
    deformation[0, ..., 2] = torch.clamp(plane * weight, 0.0, sensor.cfg.max_normal_compression)
    deformation[0, ..., 0] = shear_xy[0] * weight
    deformation[0, ..., 1] = shear_xy[1] * weight
    deformation[0, ..., 3] = tilt_x * weight
    deformation[0, ..., 4] = tilt_y * weight
    return deformation


def _append_rows(
    rows: list[dict[str, object]],
    case: str,
    frame: int,
    sensor: HallFootSensor,
    normal_mm: float,
    shear_mm: tuple[float, float],
    tilt_deg: tuple[float, float],
) -> None:
    debug = sensor.get_debug_data()
    for foot_index, foot_name in enumerate(("left_foot", "right_foot")):
        for site in range(sensor.num_sensors):
            field = debug["filtered_magnetic_field"][0, foot_index, site]
            baseline = debug["zero_load_baseline"][0, foot_index, site]
            deformation = debug["local_deformation"][0, foot_index, site]
            rows.append(
                {
                    "case": case,
                    "frame": frame,
                    "foot": foot_name,
                    "hall_index": site,
                    "normal_displacement_mm": normal_mm,
                    "shear_x_mm": shear_mm[0],
                    "shear_y_mm": shear_mm[1],
                    "tilt_x_deg": tilt_deg[0],
                    "tilt_y_deg": tilt_deg[1],
                    "Bx_T": float(field[0]),
                    "By_T": float(field[1]),
                    "Bz_T": float(field[2]),
                    "dB_T": float(torch.linalg.vector_norm(field)),
                    "baseline_Bx_T": float(baseline[0]),
                    "baseline_By_T": float(baseline[1]),
                    "baseline_Bz_T": float(baseline[2]),
                    "dx_m": float(deformation[0]),
                    "dy_m": float(deformation[1]),
                    "dz_m": float(deformation[2]),
                    "roll_rad": float(deformation[3]),
                    "pitch_rad": float(deformation[4]),
                    "yaw_rad": float(deformation[5]),
                    "valid": bool(debug["valid_mask"][0, foot_index, site]),
                }
            )


def main() -> None:
    if args_cli.hold_frames < 1 or args_cli.max_displacement_mm <= 0.0:
        raise ValueError("hold-frames and max-displacement-mm must be positive")
    cfg = HallFootSensorCfg(
        enable_debug_vis=not args_cli.headless,
        debug_vis_max_envs=1,
        noise_std=(0.0, 0.0, 0.0),
        drift_std_per_sqrt_s=0.0,
    )
    sim, platen_operations = _spawn_scene(cfg)
    sensor = HallFootSensor(cfg)
    sensor.initialize(1, sim.device, seed=20260807)
    foot_pos = torch.zeros((1, 2, 3), device=sim.device)
    foot_quat = torch.zeros((1, 2, 4), device=sim.device)
    foot_quat[..., 0] = 1.0
    rows: list[dict[str, object]] = []

    def step_case(
        name: str,
        center: int,
        normal_mm: float,
        shear_mm: tuple[float, float] = (0.0, 0.0),
        tilt_deg: tuple[float, float] = (0.0, 0.0),
        frames: int | None = None,
    ) -> torch.Tensor:
        frames = frames or args_cli.hold_frames
        deformation = _platen_deformation(
            sensor,
            (center, center),
            normal_mm * 1.0e-3,
            (shear_mm[0] * 1.0e-3, shear_mm[1] * 1.0e-3),
            (math.radians(tilt_deg[0]), math.radians(tilt_deg[1])),
        )
        center_xy = sensor.hall_positions_f[0, center, :2]
        platen_z = (
            cfg.hall_height
            - cfg.hall_to_tpu_top_distance
            - cfg.tpu_thickness
            - 0.004
            + normal_mm * 1.0e-3
        )
        _set_platen_pose(
            platen_operations,
            (float(center_xy[0]), float(center_xy[1]), platen_z),
            tilt_deg,
        )
        for frame in range(frames):
            sensor.update(
                cfg.sensor_period,
                foot_positions_w=foot_pos,
                foot_quaternions_w=foot_quat,
                local_deformation=deformation,
            )
            _append_rows(rows, name, frame, sensor, normal_mm, shear_mm, tilt_deg)
            sim.step()
        return sensor.get_filtered_data().clone()

    # Auto-zero in the unloaded state.
    step_case("autozero_unloaded", 7, 0.0, frames=cfg.auto_zero_samples + 2)
    responses: dict[str, torch.Tensor] = {}
    regions = {"forefoot": 2, "midfoot": 7, "heel": 12}
    displacements = (0.25, 0.50, 1.00, 1.50, args_cli.max_displacement_mm)
    for region, center in regions.items():
        for displacement in displacements:
            responses[f"{region}_{displacement:.2f}"] = step_case(
                f"{region}_normal", center, displacement
            )
        step_case(f"{region}_shear", center, 1.0, shear_mm=(0.8, -0.5))
        step_case(f"{region}_tilt", center, 1.0, tilt_deg=(6.0, -4.0))

    unloaded = step_case("unload", 7, 0.0, frames=80)
    output_path = Path(args_cli.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    checks: dict[str, object] = {}
    for region, center in regions.items():
        field = responses[f"{region}_{args_cli.max_displacement_mm:.2f}"]
        magnitude = torch.linalg.vector_norm(field, dim=-1)
        far_index = {"forefoot": 14, "midfoot": 0, "heel": 0}[region]
        checks[f"{region}_locality_ratio"] = float(
            magnitude[0, :, center].mean() / magnitude[0, :, far_index].mean().clamp_min(1.0e-12)
        )
    symmetry_field = responses[f"midfoot_{args_cli.max_displacement_mm:.2f}"][0]
    right_corrected = symmetry_field[1] * torch.tensor((1.0, -1.0, 1.0), device=sim.device)
    checks["left_right_symmetry_max_error_T"] = float((symmetry_field[0] - right_corrected).abs().max())
    checks["unload_max_abs_T"] = float(unloaded.abs().max())
    checks["all_finite"] = all(
        math.isfinite(float(row[key]))
        for row in rows
        for key in ("Bx_T", "By_T", "Bz_T", "dB_T", "dx_m", "dy_m", "dz_m")
    )
    checks["pass"] = bool(
        checks["all_finite"]
        and all(checks[f"{region}_locality_ratio"] > 1.2 for region in regions)
        and checks["left_right_symmetry_max_error_T"] < 2.0e-5
        and checks["unload_max_abs_T"] < 5.0e-5
    )
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"csv": str(output_path), "summary": str(summary_path), **checks}, ensure_ascii=False, indent=2),
        flush=True,
    )
    if not checks["pass"]:
        raise RuntimeError("Hall platen validation failed; inspect the CSV and summary")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    # This script owns its Kit process; avoid a known Sim 5.1 graceful teardown
    # stall after a failed validation or CUDA context error.
    simulation_app.close(skip_cleanup=True)
