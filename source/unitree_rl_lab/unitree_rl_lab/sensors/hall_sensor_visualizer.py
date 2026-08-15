"""Optional Isaac Lab viewport markers for the magnetic sole."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_matrix

from .hall_sensor_config import HallFootSensorCfg


def _quaternion_from_axis(axis: tuple[float, float, float], vector: torch.Tensor) -> torch.Tensor:
    """Return scalar-first quaternions rotating ``axis`` onto ``vector``."""
    target = vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1.0e-12)
    source = torch.tensor(axis, device=vector.device, dtype=vector.dtype).expand_as(target)
    dot = torch.sum(source * target, dim=-1, keepdim=True)
    cross = torch.linalg.cross(source, target, dim=-1)
    quaternion = torch.cat((1.0 + dot, cross), dim=-1)
    opposite = dot.squeeze(-1) < -0.999999
    if opposite.any():
        fallback_axis = torch.zeros_like(target[opposite])
        fallback_axis[:, 1] = 1.0
        quaternion[opposite, 0] = 0.0
        quaternion[opposite, 1:] = fallback_axis
    return quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-12)


class HallSensorVisualizer:
    """Draw only the first configured environments to protect RL throughput."""

    def __init__(self, cfg: HallFootSensorCfg) -> None:
        self.cfg = cfg
        self.halls = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/HallFoot/Halls",
                markers={
                    "hall": sim_utils.CuboidCfg(
                        size=cfg.hall_package_size,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.04, 0.04),
                            roughness=0.45,
                            opacity=1.0,
                        ),
                    )
                },
            )
        )
        self.magnets = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/HallFoot/Magnets",
                markers={
                    "magnet": sim_utils.CylinderCfg(
                        radius=0.5 * cfg.magnet_size,
                        height=cfg.magnet_thickness,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 1.0)),
                    )
                },
            )
        )
        self.connections = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/HallFoot/Connections",
                markers={
                    "line": sim_utils.CylinderCfg(
                        radius=0.00025,
                        height=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1)),
                    )
                },
            )
        )
        self.field_arrows = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/HallFoot/FieldArrows",
                markers={
                    "field": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                        # X is the vector-length axis.  Keep Y/Z very thin so
                        # 15 nearby arrows do not merge into a solid block.
                        scale=(1.0, cfg.debug_field_arrow_width, cfg.debug_field_arrow_width),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 1.0),
                            emissive_color=(0.0, 0.35, 0.35),
                            opacity=1.0,
                        ),
                    )
                },
            )
        )
        self.compression = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/HallFoot/Compression",
                markers={
                    "low": sim_utils.SphereCfg(
                        radius=0.0010,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.2)),
                    ),
                    "medium": sim_utils.SphereCfg(
                        radius=0.0013,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.75, 0.05)),
                    ),
                    "high": sim_utils.SphereCfg(
                        radius=0.0016,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.05)),
                    ),
                },
            )
        )

    def update(self, data: dict[str, torch.Tensor]) -> None:
        limit = min(self.cfg.debug_vis_max_envs, data["hall_positions_w"].shape[0])
        hall = data["hall_positions_w"][:limit].reshape(-1, 3)
        hall_rotation = data["hall_rotations_w"][:limit].reshape(-1, 3, 3)
        magnet = data["magnet_positions_w"][:limit].reshape(-1, self.cfg.magnets_per_hall, 3)
        magnet_flat = magnet.reshape(-1, 3)
        self.halls.visualize(
            translations=hall,
            orientations=quat_from_matrix(hall_rotation),
        )
        self.magnets.visualize(translations=magnet_flat)

        hall_for_magnets = hall.unsqueeze(1).expand(-1, self.cfg.magnets_per_hall, -1).reshape(-1, 3)
        connection_vector = magnet_flat - hall_for_magnets
        connection_length = torch.linalg.vector_norm(connection_vector, dim=-1)
        self.connections.visualize(
            translations=0.5 * (hall_for_magnets + magnet_flat),
            orientations=_quaternion_from_axis((0.0, 0.0, 1.0), connection_vector),
            scales=torch.stack(
                (torch.ones_like(connection_length), torch.ones_like(connection_length), connection_length),
                dim=-1,
            ),
        )

        field = data["filtered_magnetic_field"][:limit].reshape(-1, 3)
        field_norm = torch.linalg.vector_norm(field, dim=-1)
        arrow_length = torch.clamp(
            field_norm * self.cfg.debug_field_scale,
            self.cfg.debug_field_arrow_min_length,
            self.cfg.debug_field_arrow_max_length,
        )
        safe_field = torch.where(
            (field_norm > 1.0e-12).unsqueeze(-1),
            field,
            torch.tensor((1.0, 0.0, 0.0), device=field.device),
        )
        self.field_arrows.visualize(
            translations=hall,
            orientations=_quaternion_from_axis((1.0, 0.0, 0.0), safe_field),
            scales=torch.stack(
                (arrow_length, torch.ones_like(arrow_length), torch.ones_like(arrow_length)), dim=-1
            ),
        )

        compression = data["local_deformation"][:limit, ..., 2].reshape(-1)
        fraction = compression / max(self.cfg.max_normal_compression, 1.0e-9)
        indices = torch.where(fraction < 0.25, 0, torch.where(fraction < 0.65, 1, 2)).to(torch.int32)
        self.compression.visualize(
            translations=hall + torch.tensor((0.0, 0.0, 0.003), device=hall.device),
            marker_indices=indices,
        )

    def set_visibility(self, visible: bool) -> None:
        for marker in (self.halls, self.magnets, self.connections, self.field_arrows, self.compression):
            marker.set_visibility(visible)
