#!/usr/bin/env python3
"""True Scheme-B fixed-foot platen test for Isaac Sim 5.1 / Lab 2.3.2.

Visible/physical stack (top to bottom): rigid sole, rigid PCB enclosure with
the PCB inside, and one magnetized deformable TPU layer.  There is no connector
layer.  Four magnetic material frames per Hall IC are sampled from the live
PhysX nodes and passed to the dipole field model.  CSV output contains Hall
Bx/By/Bz only plus explicitly labelled simulation-debug deformation columns.
"""

from __future__ import annotations

import argparse
import faulthandler

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="logs/hall_foot_sensor/deformable_platen.csv")
parser.add_argument("--max-displacement-mm", type=float, default=2.0)
parser.add_argument("--settle-steps", type=int, default=64)
parser.add_argument("--ramp-steps", type=int, default=48)
parser.add_argument("--hold-steps", type=int, default=96)
parser.add_argument(
    "--diagnostic-only",
    action="store_true",
    help="write results and keep GUI visualization usable even when quantitative checks fail",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
faulthandler.dump_traceback_later(60.0, repeat=True)
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

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg

from unitree_rl_lab.assets.deformable_usd import DeformableUsdFileCfg
from unitree_rl_lab.sensors import HALL_LAYOUT_SOURCE_IMAGE, HallFootSensor, HallFootSensorCfg
from unitree_rl_lab.sensors.hall_deformable_sole import DeformableMagnetizedSoleAdapter


PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "source" / "unitree_rl_lab" / "unitree_rl_lab"
TPU_USD = PACKAGE_ROOT / "assets" / "meshes" / "tpu_sole_a40_grid35.usd"


class _FixedFootData:
    def __init__(self, position: torch.Tensor, quaternion: torch.Tensor) -> None:
        self.body_pos_w = position
        self.body_quat_w = quaternion


class _FixedFootRobot:
    def __init__(self, position: torch.Tensor, quaternion: torch.Tensor) -> None:
        self.data = _FixedFootData(position, quaternion)


def _tpu_cfg(path: str, cfg: HallFootSensorCfg, color: tuple[float, float, float]) -> DeformableObjectCfg:
    return DeformableObjectCfg(
        prim_path=path,
        spawn=DeformableUsdFileCfg(
            usd_path=str(TPU_USD),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                deformable_enabled=True,
                self_collision=cfg.tpu_self_collision,
                solver_position_iteration_count=cfg.tpu_solver_position_iteration_count,
                simulation_hexahedral_resolution=cfg.tpu_simulation_hexahedral_resolution,
                collision_simplification=True,
                collision_simplification_force_conforming=True,
                contact_offset=cfg.tpu_contact_offset,
                rest_offset=cfg.tpu_rest_offset,
            ),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                density=cfg.tpu_density,
                dynamic_friction=cfg.tpu_dynamic_friction,
                youngs_modulus=cfg.tpu_youngs_modulus,
                poissons_ratio=cfg.tpu_poisson_ratio,
                elasticity_damping=cfg.tpu_damping,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.75,
                opacity=1.0,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        debug_vis=False,
    )


def _spawn_stack_visuals(cfg: HallFootSensorCfg, foot_y: tuple[float, float]) -> None:
    """Show only the three real layers; PCB is contained by its rigid housing."""
    tpu_top_z = 0.5 * cfg.tpu_thickness
    pcb_center_z = tpu_top_z + 0.5 * cfg.pcb_enclosure_thickness
    rigid_sole_center_z = (
        tpu_top_z + cfg.pcb_enclosure_thickness + 0.5 * cfg.sole_thickness
    )
    for side, y in zip(("left", "right"), foot_y, strict=True):
        pcb_cfg = sim_utils.CuboidCfg(
            size=(cfg.sole_length, cfg.sole_width, cfg.pcb_enclosure_thickness),
            visual_material=sim_utils.PreviewSurfaceCfg(
                # Purple: rigid PCB enclosure, with the PCB contained inside.
                diffuse_color=(0.48, 0.16, 0.70),
                roughness=0.55,
                opacity=1.0,
            ),
        )
        pcb_cfg.func(
            f"/World/{side}_PCB_enclosure_with_PCB_inside",
            pcb_cfg,
            translation=(cfg.sole_origin[0], y + cfg.sole_origin[1], pcb_center_z),
        )
        sole_cfg = sim_utils.CuboidCfg(
            size=(cfg.sole_length, cfg.sole_width, cfg.sole_thickness),
            visual_material=sim_utils.PreviewSurfaceCfg(
                # Dark gray: rigid robot sole.
                diffuse_color=(0.20, 0.22, 0.25),
                roughness=0.72,
                opacity=1.0,
            ),
        )
        sole_cfg.func(
            f"/World/{side}_rigid_robot_sole",
            sole_cfg,
            translation=(cfg.sole_origin[0], y + cfg.sole_origin[1], rigid_sole_center_z),
        )


def _design_scene(cfg: HallFootSensorCfg):
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.0025, render_interval=4, device=args_cli.device)
    )
    sim.set_camera_view(eye=(0.38, -0.34, 0.24), target=(0.035, 0.0, 0.0))
    light = sim_utils.DomeLightCfg(intensity=1800.0)
    light.func("/World/Light", light)

    foot_y = (0.070, -0.070)
    _spawn_stack_visuals(cfg, foot_y)
    # Opaque blue/green keeps left and right deformable layers unmistakable.
    left_tpu = DeformableObject(_tpu_cfg("/World/left_magnetized_TPU", cfg, (0.06, 0.32, 0.95)))
    right_tpu = DeformableObject(_tpu_cfg("/World/right_magnetized_TPU", cfg, (0.06, 0.72, 0.24)))

    sim_utils.create_prim("/World/Presses/left_0", "Xform")
    sim_utils.create_prim("/World/Presses/right_1", "Xform")
    platen_cfg = RigidObjectCfg(
        prim_path="/World/Presses/.*/Platen",
        spawn=sim_utils.CuboidCfg(
            size=(0.065, 0.055, 0.008),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=5.0e-4,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                # Orange: external test platen, not part of the foot stack.
                diffuse_color=(0.95, 0.42, 0.04),
                roughness=0.40,
                metallic=0.15,
                opacity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.02)),
    )
    platens = RigidObject(platen_cfg)
    sim.reset()

    foot_origin_z = -(
        cfg.sole_origin[2]
        + cfg.hall_height
        - cfg.hall_to_tpu_top_distance
        - 0.5 * cfg.tpu_thickness
    )
    foot_positions = torch.tensor(
        [[[0.0, foot_y[0], foot_origin_z], [0.0, foot_y[1], foot_origin_z]]],
        device=sim.device,
    )
    foot_quaternions = torch.zeros((1, 2, 4), device=sim.device)
    foot_quaternions[..., 0] = 1.0
    fixed_robot = _FixedFootRobot(foot_positions, foot_quaternions)
    adapter = DeformableMagnetizedSoleAdapter(
        fixed_robot,
        left_tpu,
        right_tpu,
        (0, 1),
        cfg,
    )
    sensor = HallFootSensor(cfg)
    sensor.initialize(1, sim.device, magnet_pose_provider=adapter, seed=20260808)
    return sim, (left_tpu, right_tpu), platens, adapter, sensor, foot_positions, foot_quaternions


def _quat_from_tilt(roll: float, pitch: float, device: torch.device) -> torch.Tensor:
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    return torch.tensor((cr * cp, sr * cp, cr * sp, -sr * sp), device=device)


def main() -> None:
    if min(args_cli.settle_steps, args_cli.ramp_steps, args_cli.hold_steps) < 1:
        raise ValueError("step counts must be positive")
    if not TPU_USD.is_file():
        raise FileNotFoundError(TPU_USD)
    cfg = HallFootSensorCfg(
        implementation_mode="deformable",
        enable_debug_vis=not args_cli.headless,
        debug_vis_max_envs=1,
        noise_std=(0.0, 0.0, 0.0),
        drift_std_per_sqrt_s=0.0,
        tpu_simulation_hexahedral_resolution=96,
        # The test must run far enough to report the cooker error explicitly;
        # pass/fail below still rejects the inflated geometry.
        deformable_strict_geometry_check=False,
    )
    sim, tpu_assets, platens, adapter, sensor, foot_pos, foot_quat = _design_scene(cfg)
    dt = sim.get_physics_dt()
    rows: list[dict[str, object]] = []
    plate_gap = 2.0e-4
    plate_unloaded_z = -0.5 * cfg.tpu_thickness - plate_gap - 0.004

    def set_platens(
        site: int,
        displacement: float,
        shear: tuple[float, float],
        tilt: tuple[float, float],
    ) -> None:
        hall_f = sensor.hall_positions_f[:, site]
        pose = torch.zeros((2, 7), device=sim.device)
        pose[:, :3] = foot_pos[0] + hall_f
        pose[:, 2] = plate_unloaded_z + displacement
        pose[0, 0] += shear[0]
        pose[0, 1] += shear[1]
        pose[1, 0] += shear[0]
        pose[1, 1] -= shear[1]
        pose[0, 3:] = _quat_from_tilt(tilt[0], tilt[1], sim.device)
        pose[1, 3:] = _quat_from_tilt(-tilt[0], tilt[1], sim.device)
        platens.write_root_pose_to_sim(pose)
        platens.write_root_velocity_to_sim(torch.zeros((2, 6), device=sim.device))

    def physics_step() -> None:
        adapter.update_attachments()
        platens.write_data_to_sim()
        for asset in tpu_assets:
            asset.write_data_to_sim()
        sim.step()
        platens.update(dt)
        for asset in tpu_assets:
            asset.update(dt)
        sensor.update(
            dt,
            foot_positions_w=foot_pos,
            foot_quaternions_w=foot_quat,
        )

    set_platens(7, 0.0, (0.0, 0.0), (0.0, 0.0))
    for _ in range(max(args_cli.settle_steps, cfg.auto_zero_samples * 8 + 8)):
        physics_step()
    baseline_reference = sensor.get_raw_data().clone()
    unloaded_deformation_max_m = float(
        sensor.get_debug_data()["local_deformation"][..., :3].abs().max()
    )

    cases = (
        ("forefoot_normal", 2, (0.0, 0.0), (0.0, 0.0)),
        ("midfoot_normal", 7, (0.0, 0.0), (0.0, 0.0)),
        ("heel_normal", 12, (0.0, 0.0), (0.0, 0.0)),
        ("midfoot_shear", 7, (8.0e-4, -5.0e-4), (0.0, 0.0)),
        ("midfoot_tilt", 7, (0.0, 0.0), (math.radians(5.0), math.radians(-4.0))),
    )
    peak_by_case: dict[str, torch.Tensor] = {}
    for name, site, shear, tilt in cases:
        adapter.reset()
        set_platens(site, 0.0, (0.0, 0.0), (0.0, 0.0))
        for _ in range(args_cli.settle_steps):
            physics_step()
        max_displacement = args_cli.max_displacement_mm * 1.0e-3
        for ramp in range(args_cli.ramp_steps):
            fraction = (ramp + 1) / args_cli.ramp_steps
            set_platens(
                site,
                fraction * max_displacement,
                (fraction * shear[0], fraction * shear[1]),
                (fraction * tilt[0], fraction * tilt[1]),
            )
            physics_step()
        for frame in range(args_cli.hold_steps):
            physics_step()
            if frame % 8 != 0:
                continue
            debug = sensor.get_debug_data()
            for foot, foot_name in enumerate(("left", "right")):
                for hall in range(cfg.num_hall_sensors):
                    raw = debug["raw_magnetic_field"][0, foot, hall]
                    delta = debug["filtered_magnetic_field"][0, foot, hall]
                    deformation = debug["local_deformation"][0, foot, hall]
                    rows.append(
                        {
                            "case": name,
                            "frame": frame,
                            "foot": foot_name,
                            "hall_index": hall,
                            "Bx_T": float(raw[0]),
                            "By_T": float(raw[1]),
                            "Bz_T": float(raw[2]),
                            "dBx_T": float(delta[0]),
                            "dBy_T": float(delta[1]),
                            "dBz_T": float(delta[2]),
                            "dB_T": float(torch.linalg.vector_norm(delta)),
                            "sim_debug_dx_m": float(deformation[0]),
                            "sim_debug_dy_m": float(deformation[1]),
                            "sim_debug_dz_m": float(deformation[2]),
                            "valid": bool(debug["valid_mask"][0, foot, hall]),
                        }
                    )
        peak_by_case[name] = sensor.get_filtered_data().clone()

    # Unload without re-zeroing: a viscoelastic signal should return near the
    # original zero-load baseline after a sufficiently long relaxation.
    set_platens(7, 0.0, (0.0, 0.0), (0.0, 0.0))
    for _ in range(4 * args_cli.hold_steps):
        physics_step()
    unload = sensor.get_filtered_data().clone()

    output = Path(args_cli.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    checks: dict[str, object] = {
        "format": "hall-deformable-platen-validation-v1",
        "physical_stack": [
            "rigid_robot_sole",
            "rigid_PCB_enclosure_with_PCB_inside",
            "magnetized_deformable_TPU",
        ],
        "has_connector_layer": False,
        "hall_layout_source": HALL_LAYOUT_SOURCE_IMAGE,
        "hall_positions_normalized": [list(position) for position in cfg.hall_positions_normalized],
        "hall_package_size_m": list(cfg.hall_package_size),
        "material": {
            "effective_shore_a": cfg.tpu_effective_shore_a,
            "youngs_modulus_pa": cfg.tpu_youngs_modulus,
            "poissons_ratio": cfg.tpu_poisson_ratio,
            "density_kg_m3": cfg.tpu_density,
        },
        "embedding_rest_error_max_m": float(adapter.embedding_rest_error_m),
        "expected_tpu_thickness_m": cfg.tpu_thickness,
        "cooked_bbox_extent_left_right_m": [
            [float(value) for value in extent] for extent in adapter.cooked_bbox_extent_m
        ],
        "cooked_thickness_ratio_left_right": [
            float(value) for value in adapter.cooked_thickness_ratio
        ],
        "embedding_inside_fraction": float(adapter.embedding_inside_mask.float().mean()),
        "embedding_valid_sites_left_right": [
            int(adapter.embedding_inside_mask[foot].all(dim=(-1, -2)).sum()) for foot in range(2)
        ],
        "simulation_node_count": int(adapter.num_nodes),
        "valid_simulation_node_count_left_right": [
            int(adapter._valid_node_mask[foot].sum()) for foot in range(2)
        ],
        "rest_bbox_left_m": [
            [
                float(value)
                for value in adapter._rest_nodes_f[0, 0, adapter._valid_node_mask[0]].amin(dim=0)
            ],
            [
                float(value)
                for value in adapter._rest_nodes_f[0, 0, adapter._valid_node_mask[0]].amax(dim=0)
            ],
        ],
        "query_bbox_left_m": [
            [float(value) for value in adapter._query_points_f[0].reshape(-1, 3).amin(dim=0)],
            [float(value) for value in adapter._query_points_f[0].reshape(-1, 3).amax(dim=0)],
        ],
        "top_anchor_node_count_left_right": [
            int(adapter._top_node_mask[:, foot].sum()) for foot in range(2)
        ],
        "unloaded_deformation_max_m": unloaded_deformation_max_m,
        "baseline_raw_change_max_T": float((sensor.get_raw_data() - baseline_reference).abs().max()),
        "unload_delta_max_T": float(unload.abs().max()),
        "all_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in ("Bx_T", "By_T", "Bz_T", "dBx_T", "dBy_T", "dBz_T")
        ),
    }
    for region, site, far in (("forefoot", 2, 14), ("midfoot", 7, 0), ("heel", 12, 0)):
        magnitude = torch.linalg.vector_norm(peak_by_case[f"{region}_normal"], dim=-1)
        checks[f"{region}_locality_ratio"] = float(
            magnitude[0, :, site].mean() / magnitude[0, :, far].mean().clamp_min(1.0e-12)
        )
    checks["pass"] = bool(
        checks["all_finite"]
        and max(checks["cooked_thickness_ratio_left_right"])
        <= cfg.deformable_max_cooked_thickness_ratio
        and checks["unloaded_deformation_max_m"] < 3.0e-3
        and all(checks[f"{region}_locality_ratio"] > 1.05 for region in ("forefoot", "midfoot", "heel"))
    )
    summary = output.with_suffix(".summary.json")
    summary.write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(output), "summary": str(summary), **checks}, ensure_ascii=False, indent=2))
    faulthandler.cancel_dump_traceback_later()
    if not checks["pass"] and not args_cli.diagnostic_only:
        raise RuntimeError("Scheme-B platen validation failed; inspect CSV and summary")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close(skip_cleanup=True)
