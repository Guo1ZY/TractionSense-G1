"""Isaac friction events whose privileged label exactly matches physics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.events import randomize_rigid_body_material
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg


class CoherentFrictionWithBuffer(ManagerTermBase):
    """Assign one static==dynamic μ per environment and store that exact μ."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._material_term = randomize_rigid_body_material(cfg, env)
        env.ground_friction_mu_buf = torch.full(
            (env.scene.num_envs,),
            0.8,
            device=env.device,
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = True,
    ) -> None:
        del dynamic_friction_range, num_buckets, asset_cfg, make_consistent
        if env_ids is None:
            env_ids_cpu = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids_cpu = env_ids.to("cpu")
        low, high = map(float, static_friction_range)
        restitution_low, restitution_high = map(float, restitution_range)
        mu = torch.empty(len(env_ids_cpu), device="cpu").uniform_(low, high)
        restitution = torch.empty(len(env_ids_cpu), device="cpu").uniform_(
            restitution_low,
            restitution_high,
        )
        per_environment_material = torch.stack(
            (mu, mu, restitution),
            dim=-1,
        )
        total_shapes = self._material_term.asset.root_physx_view.max_shapes
        material_samples = per_environment_material[:, None, :].expand(
            -1,
            total_shapes,
            -1,
        )
        materials = (
            self._material_term.asset.root_physx_view.get_material_properties()
        )
        if self._material_term.num_shapes_per_body is None:
            materials[env_ids_cpu] = material_samples
        else:
            for body_id in self._material_term.asset_cfg.body_ids:
                start = sum(self._material_term.num_shapes_per_body[:body_id])
                stop = start + self._material_term.num_shapes_per_body[body_id]
                materials[env_ids_cpu, start:stop] = material_samples[:, start:stop]
        self._material_term.asset.root_physx_view.set_material_properties(
            materials,
            env_ids_cpu,
        )
        env_ids_device = env_ids_cpu.to(env.device)
        env.ground_friction_mu_buf[env_ids_device] = mu.to(env.device)


class CoherentFootFrictionWithBuffer(ManagerTermBase):
    """Apply exact per-foot μ, including asymmetric and interval transitions."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._material_term = randomize_rigid_body_material(cfg, env)
        if not hasattr(env, "ground_friction_mu_buf"):
            env.ground_friction_mu_buf = torch.full(
                (env.scene.num_envs, 2),
                0.8,
                device=env.device,
            )
        counts = self._material_term.num_shapes_per_body
        if counts is None:
            raise RuntimeError(
                "per-foot friction requires resolved per-body collision shape counts"
            )
        body_names = self._material_term.asset.body_names
        parameters = cfg.params
        foot_names = (
            parameters["left_body_name"],
            parameters["right_body_name"],
        )
        slices = []
        for name in foot_names:
            if name not in body_names:
                raise ValueError(f"foot body {name!r} was not found")
            body_id = body_names.index(name)
            start = sum(counts[:body_id])
            slices.append(slice(start, start + counts[body_id]))
        self._foot_shape_slices = tuple(slices)
        env._canonical_foot_material_slices = self._foot_shape_slices

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        left_body_name: str,
        right_body_name: str,
        asymmetric_probability: float = 0.5,
        make_consistent: bool = True,
    ) -> None:
        del (
            dynamic_friction_range,
            num_buckets,
            asset_cfg,
            left_body_name,
            right_body_name,
            make_consistent,
        )
        if not 0.0 <= asymmetric_probability <= 1.0:
            raise ValueError("asymmetric_probability must be within [0,1]")
        env_ids_cpu = (
            torch.arange(env.scene.num_envs, device="cpu")
            if env_ids is None
            else env_ids.to("cpu")
        )
        count = len(env_ids_cpu)
        low, high = map(float, static_friction_range)
        left_mu = torch.empty(count, device="cpu").uniform_(low, high)
        independent_right = torch.empty(count, device="cpu").uniform_(low, high)
        asymmetric = (
            torch.rand(count, device="cpu") < asymmetric_probability
        )
        right_mu = torch.where(asymmetric, independent_right, left_mu)
        restitution = torch.empty(count, device="cpu").uniform_(
            *map(float, restitution_range)
        )
        materials = (
            self._material_term.asset.root_physx_view.get_material_properties()
        )
        for foot, foot_slice in enumerate(self._foot_shape_slices):
            mu = left_mu if foot == 0 else right_mu
            sample = torch.stack((mu, mu, restitution), dim=-1)
            materials[env_ids_cpu, foot_slice] = sample[:, None, :]
        self._material_term.asset.root_physx_view.set_material_properties(
            materials,
            env_ids_cpu,
        )
        env.ground_friction_mu_buf[env_ids_cpu.to(env.device)] = torch.stack(
            (left_mu, right_mu),
            dim=-1,
        ).to(env.device)
