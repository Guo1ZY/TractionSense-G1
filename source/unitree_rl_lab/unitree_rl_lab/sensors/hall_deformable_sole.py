"""Isaac Lab 2.3 adapter for the Scheme-B magnetized TPU sole.

The physical stack represented here is exactly

``rigid robot sole -> rigid PCB enclosure (PCB inside) -> magnetized TPU``.

Only the final magnetized TPU object is deformable.  A thin selection of its
top simulation nodes is kinematically tied directly to the rigid foot/PCB
assembly; that selection is a boundary condition and is not another material
layer.  Four virtual magnet frames per Hall site are embedded into the current
TPU nodal field with inverse-distance interpolation.  No magnetic body is
created and no force is reconstructed from the resulting Hall signal.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence
import warnings

import torch

from isaaclab.assets import Articulation, DeformableObject
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .hall_foot_sensor import DeformableMagnetPoseSample, quaternion_to_matrix
from .hall_sensor_config import HallFootSensorCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _normalize(value: torch.Tensor, eps: float = 1.0e-10) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(eps)


def _rotation_z(angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    result = torch.zeros((*angle.shape, 3, 3), device=angle.device, dtype=angle.dtype)
    result[..., 0, 0] = c
    result[..., 0, 1] = -s
    result[..., 1, 0] = s
    result[..., 1, 1] = c
    result[..., 2, 2] = 1.0
    return result


class DeformableMagnetizedSoleAdapter:
    """Attach two deformable TPU meshes and expose embedded magnet poses."""

    def __init__(
        self,
        robot: Articulation,
        left_tpu: DeformableObject,
        right_tpu: DeformableObject,
        foot_body_ids: Sequence[int],
        cfg: HallFootSensorCfg,
    ) -> None:
        if cfg.implementation_mode != "deformable":
            raise ValueError("DeformableMagnetizedSoleAdapter requires implementation_mode='deformable'")
        if len(foot_body_ids) != 2:
            raise ValueError("foot_body_ids must be [left, right]")
        if left_tpu.num_instances != right_tpu.num_instances:
            raise ValueError("left and right magnetized TPU assets must have equal batch size")
        if left_tpu.max_sim_vertices_per_body != right_tpu.max_sim_vertices_per_body:
            raise ValueError("left and right magnetized TPU meshes must share one topology")

        self.robot = robot
        self.tpu_assets = (left_tpu, right_tpu)
        self.foot_body_ids = list(foot_body_ids)
        self.cfg = cfg
        self.device = left_tpu.device
        self.dtype = left_tpu.data.default_nodal_state_w.dtype
        self.num_envs = left_tpu.num_instances
        self.num_nodes = left_tpu.max_sim_vertices_per_body

        self._simulation_elements, self._valid_node_mask = self._read_simulation_topology()
        self._rest_nodes_f = self._build_rest_nodes_in_foot_frames()
        self._validate_cooked_geometry()
        self._top_node_mask = self._build_top_node_mask()
        self._query_points_f = self._build_embedded_query_points()
        self._embedding_indices, self._embedding_weights = self._build_embedding()
        # K-nearest material interpolation is deliberately approximate.  Its
        # reconstructed rest state is the zero-deformation reference so mesh
        # discretization error can never appear as a fictitious TPU strain.
        self._nominal_embedded_points_f = self._interpolate_queries(self._rest_nodes_f)
        self._nominal_magnet_centers_f = self._nominal_embedded_points_f[..., 0, :]
        self.embedding_rest_error_m = torch.linalg.vector_norm(
            self._nominal_embedded_points_f - self._query_points_f.unsqueeze(0),
            dim=-1,
        ).amax()
        self._targets = [asset.data.nodal_kinematic_target.clone() for asset in self.tpu_assets]
        self.reset()

    def _validate_cooked_geometry(self) -> None:
        """Reject an auto-cooked volume that no longer matches the 10 mm CAD.

        This guards the physical stack, not merely rendering.  In particular,
        it prevents an artificial intermediate thickness from appearing
        between the rigid PCB enclosure and the magnetized TPU.
        """
        extents: list[torch.Tensor] = []
        for foot in range(2):
            nodes = self._rest_nodes_f[0, foot, self._valid_node_mask[foot]]
            extents.append(nodes.amax(dim=0) - nodes.amin(dim=0))
        self.cooked_bbox_extent_m = torch.stack(extents, dim=0)
        self.cooked_thickness_ratio = self.cooked_bbox_extent_m[:, 2] / self.cfg.tpu_thickness
        worst_ratio = float(self.cooked_thickness_ratio.max())
        if worst_ratio <= self.cfg.deformable_max_cooked_thickness_ratio:
            return
        message = (
            "Isaac Sim auto-cooked TPU thickness is inconsistent with the measured magnetized layer: "
            f"expected {self.cfg.tpu_thickness:.6f} m, got up to "
            f"{float(self.cooked_bbox_extent_m[:, 2].max()):.6f} m "
            f"({worst_ratio:.2f}x). The physical stack has no connector layer. "
            "Use an explicit volume mesh/current supported deformable pipeline before quantitative "
            "Scheme-B training. Set deformable_strict_geometry_check=False only for visualization diagnostics."
        )
        if self.cfg.deformable_strict_geometry_check:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _read_simulation_topology(self) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return non-padded tetrahedra and the real-node masks for both feet."""
        tables: list[torch.Tensor] = []
        node_masks: list[torch.Tensor] = []
        for asset in self.tpu_assets:
            raw = asset.root_physx_view.get_sim_element_indices()[0].to(
                device=self.device,
                dtype=torch.long,
            )
            in_range = ((raw >= 0) & (raw < self.num_nodes)).all(dim=-1)
            raw = raw[in_range]
            distinct = (
                (raw[:, 0] != raw[:, 1])
                & (raw[:, 0] != raw[:, 2])
                & (raw[:, 0] != raw[:, 3])
                & (raw[:, 1] != raw[:, 2])
                & (raw[:, 1] != raw[:, 3])
                & (raw[:, 2] != raw[:, 3])
            )
            elements = raw[distinct]
            if elements.numel() == 0:
                raise RuntimeError("deformable TPU has no non-padded simulation tetrahedra")
            mask = torch.zeros(self.num_nodes, device=self.device, dtype=torch.bool)
            mask[elements.reshape(-1)] = True
            tables.append(elements)
            node_masks.append(mask)
        return (tables[0], tables[1]), torch.stack(node_masks, dim=0)

    def _build_rest_nodes_in_foot_frames(self) -> torch.Tensor:
        """Map the centered CAD mesh into each canonical foot-local frame."""
        result: list[torch.Tensor] = []
        center_f = torch.tensor(
            (
                self.cfg.sole_origin[0],
                self.cfg.sole_origin[1],
                self.cfg.sole_origin[2]
                + self.cfg.hall_height
                - self.cfg.hall_to_tpu_top_distance
                - 0.5 * self.cfg.tpu_thickness,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        yaw = torch.deg2rad(
            torch.tensor(
                (self.cfg.sole_yaw_deg, self.cfg.right_sole_yaw_deg),
                device=self.device,
                dtype=self.dtype,
            )
        )
        rotations = _rotation_z(yaw)
        for foot, asset in enumerate(self.tpu_assets):
            default = asset.data.default_nodal_state_w[..., :3]
            valid = self._valid_node_mask[foot]
            lower = default[:, valid].amin(dim=1, keepdim=True)
            upper = default[:, valid].amax(dim=1, keepdim=True)
            centered = default - 0.5 * (lower + upper)
            if foot == 1 and self.cfg.mirror_right_y:
                centered = centered.clone()
                centered[..., 1].neg_()
            local = torch.einsum("ij,nvj->nvi", rotations[foot], centered)
            result.append(local + center_f)
        return torch.stack(result, dim=1)

    def _build_top_node_mask(self) -> torch.Tensor:
        # The measured magnetized layer is not a flat cuboid.  A global max-Z threshold
        # would constrain only the few highest heel/toe nodes and let the rest
        # of the layer sag.  Select the local upper envelope in XY cells so the
        # whole curved top face is bonded directly to the rigid PCB enclosure.
        masks: list[torch.Tensor] = []
        for foot in range(2):
            valid_indices = torch.nonzero(self._valid_node_mask[foot], as_tuple=False).flatten()
            nodes = self._rest_nodes_f[0, foot, valid_indices]
            xy = nodes[:, :2]
            lower = xy.amin(dim=0)
            cell = self.cfg.tpu_top_anchor_grid_size
            ij = torch.floor((xy - lower) / cell).to(torch.long)
            ny = int(ij[:, 1].max().item()) + 1
            linear = ij[:, 0] * ny + ij[:, 1]
            bin_count = int(linear.max().item()) + 1
            local_top = torch.full(
                (bin_count,),
                -torch.inf,
                device=self.device,
                dtype=self.dtype,
            )
            local_top.scatter_reduce_(0, linear, nodes[:, 2], reduce="amax", include_self=True)
            selected = nodes[:, 2] >= local_top[linear] - self.cfg.tpu_top_anchor_depth
            one_mask = torch.zeros(self.num_nodes, device=self.device, dtype=torch.bool)
            one_mask[valid_indices[selected]] = True
            masks.append(one_mask.unsqueeze(0).expand(self.num_envs, -1))
        mask = torch.stack(masks, dim=1)
        if not bool(mask.any(dim=-1).all()):
            raise RuntimeError("TPU top-anchor selection found no nodes")
        return mask

    def _hall_positions_f(self) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = torch.tensor(
            self.cfg.hall_positions_normalized,
            device=self.device,
            dtype=self.dtype,
        )
        xy = normalized * torch.tensor(
            (self.cfg.sole_length, self.cfg.sole_width),
            device=self.device,
            dtype=self.dtype,
        )
        xy = xy.unsqueeze(0).expand(2, -1, -1).clone()
        if self.cfg.mirror_right_y:
            xy[1, :, 1].neg_()
        yaw = torch.deg2rad(
            torch.tensor(
                (self.cfg.sole_yaw_deg, self.cfg.right_sole_yaw_deg),
                device=self.device,
                dtype=self.dtype,
            )
        )
        rotation = _rotation_z(yaw)
        xyz = torch.zeros((2, self.cfg.num_hall_sensors, 3), device=self.device, dtype=self.dtype)
        xyz[..., :2] = xy
        xyz = torch.einsum("fij,fsj->fsi", rotation, xyz)
        xyz += torch.tensor(self.cfg.sole_origin, device=self.device, dtype=self.dtype)
        xyz[..., 2] = self.cfg.sole_origin[2] + self.cfg.hall_height
        return xyz, rotation

    def _build_embedded_query_points(self) -> torch.Tensor:
        """Return center/+X/+Y material points for every embedded magnet."""
        hall, sole_rotation = self._hall_positions_f()
        sx = 0.5 * self.cfg.magnet_spacing_x
        sy = 0.5 * self.cfg.magnet_spacing_y
        planar_offsets = torch.tensor(
            ((-sx, -sy, 0.0), (-sx, sy, 0.0), (sx, -sy, 0.0), (sx, sy, 0.0)),
            device=self.device,
            dtype=self.dtype,
        )
        offsets_f = torch.einsum("fij,mj->fmi", sole_rotation, planar_offsets)
        centers = hall.unsqueeze(-2) + offsets_f.unsqueeze(1)
        centers[..., 2] -= self.cfg.initial_hall_magnet_distance

        radius = self.cfg.deformable_frame_sample_radius
        x_step = sole_rotation[..., :, 0].unsqueeze(1).unsqueeze(1) * radius
        y_step = sole_rotation[..., :, 1].unsqueeze(1).unsqueeze(1) * radius
        points = torch.stack((centers, centers + x_step, centers + y_step), dim=-2)

        # Hexa/tetra cooking approximates the curved 10 mm CAD skin with a
        # stepped simulation mesh.  Put every material sample at the requested
        # depth below its *local* cooked upper envelope, not below one global Z
        # plane that may lie outside the local tetrahedra.
        for foot in range(2):
            flat_xy = points[foot, ..., :2].reshape(-1, 2)
            nodes = self._rest_nodes_f[0, foot, self._valid_node_mask[foot]]
            k = min(32, nodes.shape[0])
            nearest = torch.topk(
                torch.cdist(flat_xy, nodes[:, :2]),
                k=k,
                dim=-1,
                largest=False,
            ).indices
            candidate_z = nodes[:, 2][nearest]
            local_top = candidate_z.amax(dim=-1)
            local_bottom = candidate_z.amin(dim=-1)
            material_z = local_top - self.cfg.magnet_embedding_depth
            half_magnet = 0.5 * self.cfg.magnet_thickness
            material_z = torch.maximum(material_z, local_bottom + half_magnet)
            material_z = torch.minimum(material_z, local_top - half_magnet)
            points[foot, ..., 2] = material_z.reshape(points[foot, ..., 2].shape)
        return points

    def _build_embedding(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Bind points to simulation tetrahedra with material barycentrics."""
        index_tables: list[torch.Tensor] = []
        weight_tables: list[torch.Tensor] = []
        inside_tables: list[torch.Tensor] = []
        for foot, asset in enumerate(self.tpu_assets):
            nodes = self._rest_nodes_f[0, foot]
            elements = self._simulation_elements[foot]
            tetra = nodes[elements]
            matrix = torch.stack(
                (
                    tetra[:, 1] - tetra[:, 0],
                    tetra[:, 2] - tetra[:, 0],
                    tetra[:, 3] - tetra[:, 0],
                ),
                dim=-1,
            )
            nondegenerate = torch.linalg.det(matrix).abs() > 1.0e-12
            elements = elements[nondegenerate]
            tetra = tetra[nondegenerate]
            inverse = torch.linalg.inv(matrix[nondegenerate])
            if elements.numel() == 0:
                raise RuntimeError("deformable TPU simulation mesh has no valid tetrahedra")

            query_shape = self._query_points_f[foot].shape[:-1]
            query = self._query_points_f[foot].reshape(-1, 3)
            relative = query[:, None, :] - tetra[None, :, 0, :]
            bary_123 = torch.einsum("eij,qej->qei", inverse, relative)
            bary = torch.cat((1.0 - bary_123.sum(dim=-1, keepdim=True), bary_123), dim=-1)
            violation = torch.relu(-bary).amax(dim=-1) + torch.relu(bary - 1.0).amax(dim=-1)
            best_element = violation.argmin(dim=-1)
            query_index = torch.arange(query.shape[0], device=self.device)
            best_weight = bary[query_index, best_element]
            inside = violation[query_index, best_element] <= 2.0e-3
            # If a point lands just outside the cooked stair-step boundary,
            # project it onto the closest tetrahedron instead of extrapolating.
            projected = torch.clamp(best_weight, 0.0, 1.0)
            projected /= projected.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
            best_weight = torch.where(inside.unsqueeze(-1), best_weight, projected)
            index_tables.append(elements[best_element].reshape(*query_shape, 4))
            weight_tables.append(best_weight.reshape(*query_shape, 4))
            inside_tables.append(inside.reshape(query_shape))

        self.embedding_inside_mask = torch.stack(inside_tables, dim=0)
        return torch.stack(index_tables, dim=0), torch.stack(weight_tables, dim=0)

    def _foot_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        position = self.robot.data.body_pos_w[:, self.foot_body_ids, :]
        quaternion = self.robot.data.body_quat_w[:, self.foot_body_ids, :]
        return position, quaternion_to_matrix(quaternion)

    def _world_from_foot(
        self,
        points_f: torch.Tensor,
        env_ids: torch.Tensor,
        foot: int,
    ) -> torch.Tensor:
        position, rotation = self._foot_pose()
        position = position[env_ids, foot]
        rotation = rotation[env_ids, foot]
        return position.unsqueeze(-2) + torch.einsum("nij,nvj->nvi", rotation, points_f)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Restore the TPU rest state and its direct top-face attachment."""
        if env_ids is None or isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            ids = env_ids.to(device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        for foot, asset in enumerate(self.tpu_assets):
            rest_f = self._rest_nodes_f[ids, foot]
            world = self._world_from_foot(rest_f, ids, foot)
            state = torch.zeros((ids.numel(), self.num_nodes, 6), device=self.device, dtype=self.dtype)
            state[..., :3] = world
            asset.write_nodal_state_to_sim(state, env_ids=ids)
            target = self._targets[foot][ids].clone()
            target[..., :3] = world
            target[..., 3] = 1.0
            mask = self._top_node_mask[ids, foot]
            target[..., 3][mask] = 0.0
            self._targets[foot][ids] = target
            asset.write_nodal_kinematic_target_to_sim(target, env_ids=ids)
            asset.reset(ids)

    def update_attachments(self) -> None:
        """Move only the TPU top boundary with the rigid PCB/foot assembly."""
        ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        for foot, asset in enumerate(self.tpu_assets):
            desired = self._world_from_foot(self._rest_nodes_f[:, foot], ids, foot)
            target = self._targets[foot]
            mask = self._top_node_mask[:, foot]
            target[..., :3][mask] = desired[mask]
            target[..., 3] = 1.0
            target[..., 3][mask] = 0.0
            asset.write_nodal_kinematic_target_to_sim(target)

    def _current_nodes(self) -> torch.Tensor:
        return torch.stack(tuple(asset.data.nodal_pos_w for asset in self.tpu_assets), dim=1)

    def _interpolate_queries(self, nodes: torch.Tensor) -> torch.Tensor:
        indices = self._embedding_indices.unsqueeze(0).expand(self.num_envs, -1, -1, -1, -1, -1)
        env_index = torch.arange(self.num_envs, device=self.device).view(-1, 1, 1, 1, 1, 1)
        foot_index = torch.arange(2, device=self.device).view(1, 2, 1, 1, 1, 1)
        samples = nodes[env_index, foot_index, indices]
        weights = self._embedding_weights.unsqueeze(0).unsqueeze(-1)
        return torch.sum(samples * weights, dim=-2)

    @staticmethod
    def _frames_from_points(points: torch.Tensor) -> torch.Tensor:
        center = points[..., 0, :]
        x_axis = _normalize(points[..., 1, :] - center)
        y_seed = points[..., 2, :] - center
        y_axis = _normalize(y_seed - torch.sum(y_seed * x_axis, dim=-1, keepdim=True) * x_axis)
        z_axis = _normalize(torch.linalg.cross(x_axis, y_axis, dim=-1))
        y_axis = _normalize(torch.linalg.cross(z_axis, x_axis, dim=-1))
        return torch.stack((x_axis, y_axis, z_axis), dim=-1)

    def sample(
        self,
        foot_positions_w: torch.Tensor,
        foot_quaternions_w: torch.Tensor,
        dt: float,
    ) -> DeformableMagnetPoseSample:
        """Sample four embedded magnet frames per Hall site from current nodes."""
        del dt
        points = self._interpolate_queries(self._current_nodes())
        center = points[..., 0, :]
        rotations_w = self._frames_from_points(points)

        foot_rotation_w = quaternion_to_matrix(foot_quaternions_w)
        site_center_w = center.mean(dim=-2)
        current_center_f = torch.einsum(
            "nfji,nfsj->nfsi",
            foot_rotation_w,
            site_center_w - foot_positions_w.unsqueeze(-2),
        )
        nominal_center_f = self._nominal_magnet_centers_f.mean(dim=-2)
        translation = current_center_f - nominal_center_f

        current_rotation_w = rotations_w.mean(dim=-3)
        current_rotation_f = torch.einsum("nfji,nfsjk->nfsik", foot_rotation_w, current_rotation_w)
        nominal_rotation_f = self._frames_from_points(self._nominal_embedded_points_f).mean(dim=-3)
        relative_rotation = torch.einsum(
            "nfsji,nfsjk->nfsik",
            nominal_rotation_f,
            current_rotation_f,
        )
        roll = torch.atan2(relative_rotation[..., 2, 1], relative_rotation[..., 2, 2])
        pitch = torch.asin(torch.clamp(-relative_rotation[..., 2, 0], -1.0, 1.0))
        yaw = torch.atan2(relative_rotation[..., 1, 0], relative_rotation[..., 0, 0])
        deformation = torch.cat((translation, torch.stack((roll, pitch, yaw), dim=-1)), dim=-1)

        valid = (
            torch.isfinite(center).all(dim=-1).all(dim=-1)
            & torch.isfinite(rotations_w).all(dim=(-1, -2, -3))
        )
        embedded = self.embedding_inside_mask.all(dim=(-1, -2)).unsqueeze(0)
        valid &= embedded
        return DeformableMagnetPoseSample(
            positions_w=torch.nan_to_num(center),
            rotations_w=torch.nan_to_num(rotations_w),
            local_deformation=torch.nan_to_num(deformation),
            valid_mask=valid,
        )


class HallSoleAttachmentAction(ActionTerm):
    """Zero-dimensional per-physics-step TPU-to-PCB attachment update."""

    cfg: "HallSoleAttachmentActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "HallSoleAttachmentActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        body_ids, body_names = self._asset.find_bodies(cfg.foot_body_names, preserve_order=True)
        if body_names != list(cfg.foot_body_names):
            raise RuntimeError(f"failed to resolve ordered foot bodies: {body_names}")
        left_tpu: DeformableObject = env.scene[cfg.left_tpu_asset_name]
        right_tpu: DeformableObject = env.scene[cfg.right_tpu_asset_name]
        self.adapter = DeformableMagnetizedSoleAdapter(
            self._asset,
            left_tpu,
            right_tpu,
            body_ids,
            cfg.hall_cfg,
        )
        env._hall_magnet_pose_provider = self.adapter
        env._hall_deformable_sole_adapter = self.adapter
        self._empty = torch.empty((env.num_envs, 0), device=env.device)
        self._export_IO_descriptor = False

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._empty

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._empty

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._empty.shape:
            raise ValueError(f"Hall attachment action must be [N,0], got {tuple(actions.shape)}")

    def apply_actions(self) -> None:
        self.adapter.update_attachments()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self.adapter.reset(env_ids)


@configclass
class HallSoleAttachmentActionCfg(ActionTermCfg):
    """Configuration for the zero-dimensional Scheme-B attachment hook."""

    class_type: type[ActionTerm] = HallSoleAttachmentAction
    asset_name: str = "robot"
    left_tpu_asset_name: str = "left_magnetized_tpu"
    right_tpu_asset_name: str = "right_magnetized_tpu"
    foot_body_names: tuple[str, str] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    hall_cfg: HallFootSensorCfg = MISSING
