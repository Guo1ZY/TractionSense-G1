"""Isaac Lab adapters for the canonical Teacher and Student schemas."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from unitree_rl_lab.traction.diagnostics import (
    TractionDiagnostics,
    TractionDiagnosticsCfg,
    TractionDiagnosticsState,
)
from unitree_rl_lab.traction.schema import (
    PRIVILEGED_TRACTION_SCHEMA,
    TEMPORAL_STUDENT_FRAME_SCHEMA,
)
from unitree_rl_lab.traction.tactile import (
    TactileDomainRandomizationCfg,
    TactileObservationModel,
)

if TYPE_CHECKING:
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.envs import ManagerBasedRLEnv


FootForceMode = Literal[
    "proprio_only",
    "ideal_raw_force",
    "randomized_tactile_force",
]


def _select_joints(asset, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    if asset_cfg.joint_ids is None or asset_cfg.joint_ids == slice(None):
        return asset.data.joint_pos, asset.data.joint_vel
    return (
        asset.data.joint_pos[:, asset_cfg.joint_ids],
        asset.data.joint_vel[:, asset_cfg.joint_ids],
    )


def canonical_current_proprio(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Return the audited 96-D current-frame proprioception in policy units."""
    asset = env.scene[asset_cfg.name]
    joint_pos, joint_vel = _select_joints(asset, asset_cfg)
    default_joint_pos = asset.data.default_joint_pos
    if asset_cfg.joint_ids is not None and asset_cfg.joint_ids != slice(None):
        default_joint_pos = default_joint_pos[:, asset_cfg.joint_ids]
    previous_action = env.action_manager.action
    command = env.command_manager.get_command(command_name)
    value = torch.cat(
        (
            asset.data.root_ang_vel_b * 0.2,
            asset.data.projected_gravity_b,
            command,
            joint_pos - default_joint_pos,
            joint_vel * 0.05,
            previous_action,
        ),
        dim=-1,
    )
    if value.shape[-1] != 96:
        raise RuntimeError(f"canonical proprio dimension changed to {value.shape[-1]}")
    return torch.nan_to_num(value)


def _true_force_local_n(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    from isaaclab.sensors import ContactSensor
    from isaaclab.utils.math import quat_apply_inverse

    def filtered_force(sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        normal = sensor.data.force_matrix_w
        friction = sensor.data.friction_forces_w
        if normal is None or friction is None:
            raise RuntimeError(
                f"{sensor_cfg.name} requires a ground filter and "
                "track_friction_forces=True"
            )
        if normal.shape[1] != 1 or friction.shape[1] != 1:
            raise RuntimeError(
                f"{sensor_cfg.name} must cover exactly one ankle-roll body"
            )
        return (normal + friction).sum(dim=(1, 2))

    force_world = torch.stack(
        (filtered_force(left_sensor_cfg), filtered_force(right_sensor_cfg)),
        dim=1,
    )
    asset = env.scene[asset_cfg.name]
    foot_quaternion_world = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    if foot_quaternion_world.shape[1] != 2:
        raise RuntimeError("asset_cfg must resolve left then right ankle-roll bodies")
    return quat_apply_inverse(
        foot_quaternion_world,
        force_world,
    ).reshape(env.num_envs, 6)


def _robot_mass_kg(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    mass = getattr(env, "_canonical_robot_mass_kg", None)
    if mass is None or not hasattr(env, "termination_manager"):
        asset = env.scene[asset_cfg.name]
        mass = asset.root_physx_view.get_masses().sum(dim=1, keepdim=True).to(
            device=env.device, dtype=torch.float32
        )
        if hasattr(env, "termination_manager"):
            env._canonical_robot_mass_kg = mass
    return mass


def _traction_diagnostics(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    cfg: TractionDiagnosticsCfg,
) -> TractionDiagnostics:
    step = int(getattr(env, "common_step_counter", 0))
    cached_step = getattr(env, "_canonical_traction_diagnostics_step", -1)
    if cached_step == step and hasattr(env, "_canonical_traction_diagnostics"):
        return env._canonical_traction_diagnostics
    state = getattr(env, "_canonical_traction_diagnostics_state", None)
    if state is None or state.num_envs != env.num_envs or state.cfg != cfg:
        state = TractionDiagnosticsState(env.num_envs, cfg=cfg, device=env.device)
        env._canonical_traction_diagnostics_state = state
    if hasattr(env, "episode_length_buf"):
        reset_ids = torch.nonzero(env.episode_length_buf <= 1, as_tuple=False).flatten()
        if reset_ids.numel():
            state.reset(reset_ids)
    force = _true_force_local_n(
        env,
        left_sensor_cfg,
        right_sensor_cfg,
        asset_cfg,
    )
    asset = env.scene[asset_cfg.name]
    foot_velocity = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    diagnostics = state.update(
        force,
        foot_velocity,
        velocity_is_proxy=True,
    )
    env._canonical_traction_diagnostics = diagnostics
    env._canonical_traction_diagnostics_step = step
    return diagnostics


def canonical_privileged_traction(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    diagnostics_cfg: TractionDiagnosticsCfg = TractionDiagnosticsCfg(),
) -> torch.Tensor:
    """Return current physically available privileged traction representation."""
    asset = env.scene[asset_cfg.name]
    proprio = canonical_current_proprio(env, asset_cfg)
    force = _true_force_local_n(
        env,
        left_sensor_cfg,
        right_sensor_cfg,
        asset_cfg,
    )
    diagnostics = _traction_diagnostics(
        env,
        left_sensor_cfg,
        right_sensor_cfg,
        asset_cfg,
        diagnostics_cfg,
    )
    mu = getattr(env, "ground_friction_mu_buf", None)
    if mu is None:
        mu = torch.full((env.num_envs,), 0.8, device=env.device)
    if mu.ndim == 1:
        mu = mu[:, None].expand(-1, 2)
    elif mu.shape[-1] == 1:
        mu = mu.expand(-1, 2)

    terrain_level = torch.zeros((env.num_envs, 1), device=env.device)
    terrain = getattr(env.scene, "terrain", None)
    if terrain is not None and hasattr(terrain, "terrain_levels"):
        terrain_level = terrain.terrain_levels.to(torch.float32).view(-1, 1)
    # Non-randomized/unavailable fields are explicit nominal zeros, not hidden
    # simulated truth. The first two contact fields retain actual left/right μ.
    terrain_contact = torch.cat(
        (
            terrain_level,
            torch.zeros_like(terrain_level),
            mu[:, 0:1],
            mu[:, 1:2],
        ),
        dim=-1,
    )
    mass_ratio = _robot_mass_kg(env, asset_cfg) / 35.2793
    dynamics = torch.cat(
        (mass_ratio, torch.zeros((env.num_envs, 7), device=env.device)),
        dim=-1,
    )
    value = torch.cat(
        (
            proprio,
            mu,
            force,
            diagnostics.force_normal,
            diagnostics.force_tangent,
            diagnostics.friction_utilization,
            diagnostics.contact.float(),
            diagnostics.foot_tangent_velocity.reshape(env.num_envs, 4),
            diagnostics.slip_speed,
            diagnostics.slip_label.float(),
            asset.data.root_lin_vel_b,
            terrain_contact,
            dynamics,
        ),
        dim=-1,
    )
    expected = PRIVILEGED_TRACTION_SCHEMA.flat_dimension
    if value.shape[-1] != expected:
        raise RuntimeError(
            f"privileged traction dimension changed to {value.shape[-1]}, expected {expected}"
        )
    return torch.nan_to_num(value)


def canonical_teacher_observation(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    diagnostics_cfg: TractionDiagnosticsCfg = TractionDiagnosticsCfg(),
) -> torch.Tensor:
    """Return ``96 current + 3 adjusted command + 135 privileged``."""
    proprio = canonical_current_proprio(env, asset_cfg, command_name)
    raw_command = env.command_manager.get_command(command_name)
    adjusted_command = getattr(env, "traction_adjusted_command_buf", raw_command)
    privileged = canonical_privileged_traction(
        env,
        left_sensor_cfg,
        right_sensor_cfg,
        asset_cfg,
        diagnostics_cfg,
    )
    return torch.cat((proprio, adjusted_command, privileged), dim=-1)


def _tactile_observation(
    env: ManagerBasedRLEnv,
    true_force_n: torch.Tensor,
    mode: FootForceMode,
    tactile_cfg: TactileDomainRandomizationCfg,
    tactile_seed: int,
    tactile_curriculum_stage: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode == "proprio_only":
        return (
            torch.zeros_like(true_force_n),
            torch.zeros((env.num_envs, 2), device=env.device),
            torch.full((env.num_envs, 2), 1.0e6, device=env.device),
        )
    if mode == "ideal_raw_force":
        return (
            true_force_n,
            torch.ones((env.num_envs, 2), device=env.device),
            torch.zeros((env.num_envs, 2), device=env.device),
        )
    if mode != "randomized_tactile_force":
        raise ValueError(f"unsupported foot force mode {mode!r}")

    model = getattr(env, "_canonical_tactile_model", None)
    if (
        model is None
        or model.num_envs != env.num_envs
        or model.cfg != tactile_cfg
        or model.curriculum_stage != tactile_curriculum_stage
    ):
        model = TactileObservationModel(
            env.num_envs,
            cfg=tactile_cfg,
            device=env.device,
            seed=tactile_seed,
            curriculum_stage=tactile_curriculum_stage,
        )
        env._canonical_tactile_model = model
    step = int(getattr(env, "common_step_counter", 0))
    if getattr(env, "_canonical_tactile_step", -1) != step:
        if hasattr(env, "episode_length_buf"):
            reset_ids = torch.nonzero(env.episode_length_buf <= 1, as_tuple=False).flatten()
            if reset_ids.numel():
                model.reset(reset_ids)
        env._canonical_tactile_value = model(true_force_n)
        env._canonical_tactile_step = step
    observation = env._canonical_tactile_value
    return (
        observation.force_xyz_n,
        observation.valid.float(),
        observation.sample_age_s,
    )


def canonical_student_frame(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    mode: FootForceMode = "randomized_tactile_force",
    command_name: str = "base_velocity",
    force_clip: tuple[float, float] = (-2.0, 2.0),
    tactile_cfg: TactileDomainRandomizationCfg = TactileDomainRandomizationCfg(),
    tactile_seed: int = 20260731,
    tactile_curriculum_stage: int = 5,
) -> torch.Tensor:
    """Return one canonical 106-D Student frame for 15-frame history."""
    proprio = canonical_current_proprio(env, asset_cfg, command_name)
    # Move command from the audited current-proprio position into the canonical
    # Student term order after previous_action.
    base_ang_gravity = proprio[:, :6]
    command = proprio[:, 6:9]
    joints_and_action = proprio[:, 9:]
    true_force = _true_force_local_n(
        env,
        left_sensor_cfg,
        right_sensor_cfg,
        asset_cfg,
    )
    observed_force, valid, age = _tactile_observation(
        env,
        true_force,
        mode,
        tactile_cfg,
        tactile_seed,
        tactile_curriculum_stage,
    )
    normalized_force = (
        observed_force / (_robot_mass_kg(env, asset_cfg) * 9.81)
    ).clamp(*force_clip)
    value = torch.cat(
        (
            base_ang_gravity,
            joints_and_action,
            command,
            normalized_force,
            valid,
            age,
        ),
        dim=-1,
    )
    expected = TEMPORAL_STUDENT_FRAME_SCHEMA.frame_dimension
    if value.shape[-1] != expected:
        raise RuntimeError(
            f"Student frame dimension changed to {value.shape[-1]}, expected {expected}"
        )
    return torch.nan_to_num(value)


def canonical_slip_penalty(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    diagnostics_cfg: TractionDiagnosticsCfg = TractionDiagnosticsCfg(),
) -> torch.Tensor:
    diagnostics = _traction_diagnostics(
        env,
        left_sensor_cfg,
        right_sensor_cfg,
        asset_cfg,
        diagnostics_cfg,
    )
    return (diagnostics.slip_speed.square() * diagnostics.contact).sum(dim=1)


def canonical_tangential_push_penalty(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    diagnostics_cfg: TractionDiagnosticsCfg = TractionDiagnosticsCfg(),
) -> torch.Tensor:
    diagnostics = _traction_diagnostics(
        env,
        left_sensor_cfg,
        right_sensor_cfg,
        asset_cfg,
        diagnostics_cfg,
    )
    normalized = diagnostics.force_tangent / (
        _robot_mass_kg(env, asset_cfg) * 9.81
    )
    return normalized.square().sum(dim=1)


def high_traction_unnecessary_slowdown(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    high_mu_threshold: float = 0.7,
) -> torch.Tensor:
    """Penalize unsupported slowdown only on known high-traction training ground."""
    asset = env.scene["robot"]
    command = env.command_manager.get_command(command_name)[:, :2]
    actual = asset.data.root_lin_vel_b[:, :2]
    mu = getattr(env, "ground_friction_mu_buf", None)
    if mu is None:
        high_mu = torch.ones(env.num_envs, device=env.device)
    else:
        high_mu = (mu.reshape(env.num_envs, -1).min(dim=1).values > high_mu_threshold).float()
    shortfall = (
        torch.linalg.vector_norm(command, dim=1)
        - torch.linalg.vector_norm(actual, dim=1)
    ).clamp_min(0.0)
    return high_mu * shortfall.square()
