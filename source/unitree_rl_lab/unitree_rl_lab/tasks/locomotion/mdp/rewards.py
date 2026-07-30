from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


"""
Foot friction / anti-slip (soft, no hard if-μ rules).
"""


def feet_anti_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 5.0,
    soft_scale: float = 0.5,
    slip_ratio_coef: float = 0.15,
) -> torch.Tensor:
    """Soft anti-slip penalty (continuous, no discrete friction if-rules).

    Combines:
      1. contact-weighted foot planar speed (classic slide)
      2. soft high tangential/normal force ratio while moving (friction-cone style)

    Friction coefficient μ is *not* read here. Domain randomization randomizes
    surface μ; the policy must learn to slow down / plant carefully from
    contact/force observations + this continuous cost.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    fn = torch.abs(forces[:, :, 2])
    ft = torch.linalg.norm(forces[:, :, :2], dim=-1)
    soft_contact = torch.sigmoid((fn - force_threshold) * soft_scale)

    asset: Articulation = env.scene[asset_cfg.name]
    foot_vel = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)

    # Clamp ratio so a near-zero Fn + large Ft spike cannot explode the term.
    slip_ratio = torch.clamp(ft / (fn + 1.0), 0.0, 5.0)
    foot_vel = torch.clamp(foot_vel, 0.0, 3.0)
    # Continuous cost: more load + more slip velocity / more cone stress → higher penalty
    cost = soft_contact * (foot_vel + slip_ratio_coef * slip_ratio * foot_vel)
    # Final bound per-env (protect value targets from rare contact spikes)
    return torch.clamp(torch.sum(cost, dim=1), 0.0, 20.0)


def feet_force_rate(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_delta_clip: float = 200.0,
    force_scale: float = 100.0,
) -> torch.Tensor:
    """Penalize high-frequency contact force change (sensor-like jerk).

    Uses ContactSensor history (needs history_length >= 2). Forces are clipped
    and scaled **before** squaring — raw ΔF² can explode (1e10+) on hard
    impacts and poison value targets / PPO (NaN → invalid action std).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    hist = contact_sensor.data.net_forces_w_history  # (N, T, B, 3)
    if hist is None or hist.shape[1] < 2:
        return torch.zeros(env.num_envs, device=env.device)
    # most recent minus previous
    delta = hist[:, 0, sensor_cfg.body_ids, :] - hist[:, 1, sensor_cfg.body_ids, :]
    delta = torch.clamp(delta, -force_delta_clip, force_delta_clip) / max(force_scale, 1e-6)
    return torch.sum(torch.square(delta), dim=(1, 2))


def _mean_contact_foot_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_threshold: float = 5.0,
    soft_scale: float = 0.5,
) -> torch.Tensor:
    """Mean soft-contact-weighted planar foot speed (m/s). Shape (num_envs,)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    fn = torch.abs(forces[:, :, 2])
    soft_contact = torch.sigmoid((fn - force_threshold) * soft_scale)

    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids if asset_cfg.body_ids is not None else sensor_cfg.body_ids
    foot_vel = torch.linalg.norm(asset.data.body_lin_vel_w[:, body_ids, :2], dim=-1)
    foot_vel = torch.clamp(foot_vel, 0.0, 5.0)
    # Average over feet (soft weights); avoid /0 when airborne
    weight = soft_contact.sum(dim=1).clamp(min=1e-3)
    return (soft_contact * foot_vel).sum(dim=1) / weight


def track_lin_vel_xy_slip_aware(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 5.0,
    soft_scale: float = 0.5,
    slip_vel_scale: float = 0.45,
    min_track_scale: float = 0.35,
) -> torch.Tensor:
    """Linear velocity tracking with soft weight when feet are slipping.

    High μ / planted feet → full track reward (encourage fast cmd following).
    High contact slip → scale track down so low-μ envs are not forced to 1.2 m/s.
    """
    # Local import avoids circular import through mdp package re-exports.
    from isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards import (
        track_lin_vel_xy_yaw_frame_exp,
    )

    base = track_lin_vel_xy_yaw_frame_exp(
        env, std=std, command_name=command_name, asset_cfg=asset_cfg
    )
    slip = _mean_contact_foot_slip(
        env, sensor_cfg, asset_cfg, force_threshold=force_threshold, soft_scale=soft_scale
    )
    # 1 at slip≈0 → min_track_scale as slip grows
    scale = min_track_scale + (1.0 - min_track_scale) * torch.exp(-slip / max(slip_vel_scale, 1e-6))
    return base * scale


def track_ang_vel_z_slip_aware(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 5.0,
    soft_scale: float = 0.5,
    slip_vel_scale: float = 0.45,
    min_track_scale: float = 0.40,
) -> torch.Tensor:
    """Yaw tracking with soft weight under foot slip (same idea as lin track)."""
    from isaaclab.envs.mdp.rewards import track_ang_vel_z_exp

    base = track_ang_vel_z_exp(env, std=std, command_name=command_name, asset_cfg=asset_cfg)
    slip = _mean_contact_foot_slip(
        env, sensor_cfg, asset_cfg, force_threshold=force_threshold, soft_scale=soft_scale
    )
    scale = min_track_scale + (1.0 - min_track_scale) * torch.exp(-slip / max(slip_vel_scale, 1e-6))
    return base * scale


def feet_motion_when_idle(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.12,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """Penalize foot planar motion when velocity command is near zero (stop stomping).

    After adaptive/yaw finetunes the policy often keeps marching in place with zero stick.
    """
    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.linalg.norm(cmd, dim=1)
    idle = (cmd_norm < cmd_threshold).float()

    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids if asset_cfg.body_ids is not None else sensor_cfg.body_ids
    foot_vel = torch.linalg.norm(asset.data.body_lin_vel_w[:, body_ids, :2], dim=-1)
    foot_vel = torch.clamp(foot_vel, 0.0, 3.0)

    # Also penalize air time if available (swinging feet while idle).
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = (contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] <= 0).float()
    cost = torch.sum(foot_vel + 0.5 * in_air, dim=1)
    return cost * idle


def base_still_when_idle(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_threshold: float = 0.12,
) -> torch.Tensor:
    """Penalize base xy / yaw speed when command is idle (should stand still)."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_norm = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    idle = (cmd_norm < cmd_threshold).float()
    vxy = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    wz = torch.abs(asset.data.root_ang_vel_b[:, 2])
    return idle * (vxy + 0.5 * wz)


# ---------------------------------------------------------------------------
# Foot-Adaptive-V2: outcome-based multi-objective (no if-μ rules, no slip-gated track floor)
# ---------------------------------------------------------------------------


def track_lin_vel_xy_exp_full(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Full linear velocity tracking (no slip-scale). Prefer this over slip_aware for V2.

    Low-μ slow-down must come from survival + slip penalties + stable_speed_bonus,
    not from artificially lowering the tracking target when feet slide.
    """
    from isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards import (
        track_lin_vel_xy_yaw_frame_exp,
    )

    return track_lin_vel_xy_yaw_frame_exp(
        env, std=std, command_name=command_name, asset_cfg=asset_cfg
    )


def track_ang_vel_z_exp_full(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Full yaw tracking without slip soft-down (pair with track_lin_vel_xy_exp_full)."""
    from isaaclab.envs.mdp.rewards import track_ang_vel_z_exp

    return track_ang_vel_z_exp(env, std=std, command_name=command_name, asset_cfg=asset_cfg)


def stable_speed_bonus(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 5.0,
    soft_scale: float = 0.5,
    slip_vel_scale: float = 0.35,
    cmd_threshold: float = 0.15,
) -> torch.Tensor:
    """Bonus only when tracking is good AND feet are planted (low slip).

    High μ → low slip at high cmd → bonus fires → incentive to go fast when grip
    allows. Low μ → slip rises → bonus vanishes while full track + slip pen remain
    → optimal is slower. Avoids slip_aware's min_track_scale "give up tracking" hole.
    """
    from isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards import (
        track_lin_vel_xy_yaw_frame_exp,
    )

    track = track_lin_vel_xy_yaw_frame_exp(
        env, std=std, command_name=command_name, asset_cfg=asset_cfg
    )
    slip = _mean_contact_foot_slip(
        env, sensor_cfg, asset_cfg, force_threshold=force_threshold, soft_scale=soft_scale
    )
    plant = torch.exp(-slip / max(slip_vel_scale, 1e-6))
    cmd = env.command_manager.get_command(command_name)
    cmd_xy = torch.linalg.norm(cmd[:, :2], dim=1)
    active = (cmd_xy > cmd_threshold).float()
    return track * plant * active


def slip_under_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 5.0,
    soft_scale: float = 0.5,
    cmd_scale: float = 1.0,
) -> torch.Tensor:
    """Penalize contact slip scaled by command magnitude.

    High commanded speed while skating is expensive; standing still has no cost.
    Continuous, no μ threshold.
    """
    slip = _mean_contact_foot_slip(
        env, sensor_cfg, asset_cfg, force_threshold=force_threshold, soft_scale=soft_scale
    )
    cmd = env.command_manager.get_command(command_name)
    cmd_mag = torch.linalg.norm(cmd, dim=1)
    return torch.clamp(slip * (1.0 + cmd_mag / max(cmd_scale, 1e-6)), 0.0, 10.0)


def accumulated_slip_step(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 5.0,
    soft_scale: float = 0.5,
    dt: float | None = None,
) -> torch.Tensor:
    """One-step contact-weighted slip distance proxy (m) for outcome cost."""
    slip_speed = _mean_contact_foot_slip(
        env, sensor_cfg, asset_cfg, force_threshold=force_threshold, soft_scale=soft_scale
    )
    step_dt = float(env.step_dt if dt is None else dt)
    return torch.clamp(slip_speed * step_dt, 0.0, 0.5)


def lateral_slip_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_x_threshold: float = 0.15,
    vy_clip: float = 1.5,
) -> torch.Tensor:
    """Penalize body lateral velocity when command is mostly forward.

    MuJoCo tests showed ICE often keeps high |v_xy| via **sideways skid** (vy),
    not useful forward vx. This pushes straight-line tracking without if-μ rules.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_x = torch.abs(cmd[:, 0])
    cmd_y = torch.abs(cmd[:, 1])
    # Gate: only when forward command dominates lateral command.
    forward_dom = (cmd_x > cmd_x_threshold).float() * (cmd_x / (cmd_x + cmd_y + 1e-3))
    vy = torch.clamp(torch.abs(asset.data.root_lin_vel_b[:, 1]), 0.0, vy_clip)
    return vy * forward_dom


def track_lin_vel_x_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward forward (x) velocity tracking in yaw frame — emphasizes vx over |v|."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    # root lin vel in body / yaw frame: use body x as forward (same convention as cmd)
    vx = asset.data.root_lin_vel_b[:, 0]
    err = torch.square(cmd[:, 0] - vx)
    return torch.exp(-err / (std**2 + 1e-8))


# ---------------------------------------------------------------------------
# Traction-adaptive speed teacher
# ---------------------------------------------------------------------------


def _traction_speed_cap(
    env: ManagerBasedRLEnv,
    low_speed: float = 0.20,
    high_speed: float = 1.50,
    mu_midpoint: float = 0.55,
    mu_width: float = 0.14,
    very_high_speed: float | None = None,
    very_high_mu_midpoint: float = 1.00,
    very_high_mu_width: float = 0.06,
    default_mu: float = 0.8,
) -> torch.Tensor:
    """Smooth μ→safe-speed map used only by rewards/critic-side teaching.

    The actor is never given ``ground_friction_mu_buf``.  It must infer the
    regime from proprioception and the temporal foot-force channels.  A smooth
    logistic cap avoids a brittle if-μ switch while encoding the requested
    behavior: roughly 0.25--0.4 m/s on ice and up to 1.5 m/s on high grip.
    """
    if hasattr(env, "ground_friction_mu_buf"):
        mu = env.ground_friction_mu_buf
    else:
        mu = torch.full((env.num_envs,), default_mu, device=env.device)
    blend = torch.sigmoid((mu - mu_midpoint) / max(mu_width, 1e-6))
    cap = low_speed + (high_speed - low_speed) * blend
    if very_high_speed is not None:
        # Very high grip permits speed but also converts a forward action
        # transient into a large pitching moment.  A second smooth shoulder
        # can therefore retain the μ≈0.8 running target while selecting a
        # slightly slower, more stable gait near the μ=1.2 upper boundary.
        shoulder = torch.sigmoid(
            (mu - very_high_mu_midpoint) / max(very_high_mu_width, 1e-6)
        )
        cap = cap + (very_high_speed - high_speed) * shoulder
    return cap


def traction_limited_track_lin_vel_x_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    low_speed: float = 0.20,
    high_speed: float = 1.50,
    mu_midpoint: float = 0.55,
    mu_width: float = 0.14,
    very_high_speed: float | None = None,
    very_high_mu_midpoint: float = 1.00,
    very_high_mu_width: float = 0.06,
) -> torch.Tensor:
    """Track ``min(|command|, traction_speed_cap(μ))`` in body-forward x.

    Large commands therefore remain meaningful on high friction, while low
    friction explicitly teaches a slower outcome instead of rewarding a
    1.5-m/s skid.  μ is privileged reward information, not actor input.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cap = _traction_speed_cap(
        env,
        low_speed=low_speed,
        high_speed=high_speed,
        mu_midpoint=mu_midpoint,
        mu_width=mu_width,
        very_high_speed=very_high_speed,
        very_high_mu_midpoint=very_high_mu_midpoint,
        very_high_mu_width=very_high_mu_width,
    )
    target_mag = torch.minimum(torch.abs(cmd_x), cap)
    target_x = torch.sign(cmd_x) * target_mag
    err = torch.square(target_x - asset.data.root_lin_vel_b[:, 0])
    return torch.exp(-err / (std**2 + 1e-8))


def traction_overspeed_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    low_speed: float = 0.20,
    high_speed: float = 1.50,
    mu_midpoint: float = 0.55,
    mu_width: float = 0.14,
    very_high_speed: float | None = None,
    very_high_mu_midpoint: float = 1.00,
    very_high_mu_width: float = 0.06,
    cmd_threshold: float = 0.10,
    excess_clip: float = 2.0,
) -> torch.Tensor:
    """Penalize body-forward speed above the smooth traction-dependent cap."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cap = _traction_speed_cap(
        env,
        low_speed=low_speed,
        high_speed=high_speed,
        mu_midpoint=mu_midpoint,
        mu_width=mu_width,
        very_high_speed=very_high_speed,
        very_high_mu_midpoint=very_high_mu_midpoint,
        very_high_mu_width=very_high_mu_width,
    )
    excess = torch.relu(torch.abs(asset.data.root_lin_vel_b[:, 0]) - cap)
    excess = torch.clamp(excess, 0.0, excess_clip)
    return torch.square(excess) * (torch.abs(cmd_x) > cmd_threshold).float()


def high_traction_underspeed_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_speed: float = 1.0,
    mu_midpoint: float = 0.78,
    mu_width: float = 0.08,
    command_midpoint: float = 0.70,
    command_width: float = 0.08,
    tolerance: float = 0.03,
    error_clip: float = 1.0,
    default_mu: float = 0.8,
) -> torch.Tensor:
    """Penalize high-grip forward-speed deficit without affecting low grip.

    This is privileged Teacher shaping: true friction only gates the reward.
    The deployable Student must infer the same regime from magnetic histories.
    A smooth friction/command gate avoids a discontinuous mode switch.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    command_mag = torch.abs(command_x)
    desired_mag = torch.minimum(
        command_mag,
        torch.full_like(command_mag, target_speed),
    )
    realized_forward = torch.sign(command_x) * asset.data.root_lin_vel_b[:, 0]
    deficit = torch.relu(desired_mag - realized_forward - tolerance)
    deficit = torch.clamp(deficit, 0.0, error_clip)
    if hasattr(env, "ground_friction_mu_buf"):
        mu = env.ground_friction_mu_buf
    else:
        mu = torch.full(
            (env.num_envs,), default_mu, device=env.device, dtype=deficit.dtype
        )
    traction_gate = torch.sigmoid(
        (mu - mu_midpoint) / max(mu_width, 1.0e-6)
    )
    command_gate = torch.sigmoid(
        (command_mag - command_midpoint) / max(command_width, 1.0e-6)
    )
    return traction_gate * command_gate * torch.square(deficit)


def friction_cone_margin_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    safe_utilization: float = 0.75,
    force_threshold: float = 5.0,
    soft_scale: float = 0.5,
    force_eps: float = 5.0,
    utilization_clip: float = 3.0,
) -> torch.Tensor:
    """Penalize use of the foot friction cone beyond a safety margin.

    Utilization is ``||Ft|| / (μ Fn + eps)``.  True μ is teacher-only; the
    actor observes Fn/Ft and their temporal ratio, so it can learn the contact
    signatures that precede saturation and slip.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    fn = torch.abs(forces[:, :, 2])
    ft = torch.linalg.norm(forces[:, :, :2], dim=-1)
    if hasattr(env, "ground_friction_mu_buf"):
        mu = env.ground_friction_mu_buf[:, None]
    else:
        mu = torch.full((env.num_envs, 1), 0.8, device=env.device)
    utilization = ft / (mu * fn + force_eps)
    utilization = torch.clamp(utilization, 0.0, utilization_clip)
    contact = torch.sigmoid((fn - force_threshold) * soft_scale)
    excess = torch.square(torch.relu(utilization - safe_utilization)) * contact
    return excess.sum(dim=1) / contact.sum(dim=1).clamp(min=1e-3)


def straight_line_motion_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_x_threshold: float = 0.10,
    yaw_rate_scale: float = 1.0,
    lateral_clip: float = 2.0,
    yaw_clip: float = 2.0,
) -> torch.Tensor:
    """Squared body-lateral speed and yaw-rate cost under forward commands."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_x = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    active = (cmd_x > cmd_x_threshold).float()
    vy = torch.clamp(asset.data.root_lin_vel_b[:, 1], -lateral_clip, lateral_clip)
    wz = torch.clamp(asset.data.root_ang_vel_b[:, 2], -yaw_clip, yaw_clip)
    return active * (torch.square(vy) + yaw_rate_scale * torch.square(wz))


def straight_cross_track_error(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_x_threshold: float = 0.10,
    error_clip: float = 1.0,
) -> torch.Tensor:
    """Squared lateral displacement relative to each episode's initial heading.

    A velocity-only cost permits a small persistent ``vy`` to accumulate into
    a large path error. This term stores the reset position and initial planar
    heading, then penalizes displacement perpendicular to that heading. It is
    invariant to randomized world yaw and therefore safe for vectorized resets.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    position_xy = asset.data.root_pos_w[:, :2]
    quat = asset.data.root_quat_w
    needs_init = (
        not hasattr(env, "straight_track_origin_xy")
        or env.straight_track_origin_xy.shape[0] != env.num_envs
    )
    if needs_init:
        env.straight_track_origin_xy = position_xy.clone()
        env.straight_track_lateral_axis = torch.zeros_like(position_xy)
        env.straight_track_initialized = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )

    # Isaac quaternion convention is (w, x, y, z). The world projection of
    # body-X is (1-2(y²+z²), 2(xy+wz)); rotate it +90° for the lateral axis.
    heading_x = 1.0 - 2.0 * (torch.square(quat[:, 2]) + torch.square(quat[:, 3]))
    heading_y = 2.0 * (quat[:, 1] * quat[:, 2] + quat[:, 0] * quat[:, 3])
    norm = torch.sqrt(torch.square(heading_x) + torch.square(heading_y)).clamp(
        min=1.0e-6
    )
    lateral_axis = torch.stack((-heading_y / norm, heading_x / norm), dim=-1)

    reset = (env.episode_length_buf <= 1) | (~env.straight_track_initialized)
    if reset.any():
        env.straight_track_origin_xy[reset] = position_xy[reset]
        env.straight_track_lateral_axis[reset] = lateral_axis[reset]
        env.straight_track_initialized[reset] = True

    displacement = position_xy - env.straight_track_origin_xy
    cross_track = torch.sum(
        displacement * env.straight_track_lateral_axis, dim=-1
    )
    cross_track = torch.clamp(cross_track, -error_clip, error_clip)
    cmd_x = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    active = (cmd_x > cmd_x_threshold).float()
    return active * torch.square(cross_track)


def traction_adaptive_feet_gait(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    slow_period: float = 0.85,
    fast_period: float = 0.50,
    stance_threshold: float = 0.55,
    low_speed: float = 0.20,
    high_speed: float = 1.50,
    mu_midpoint: float = 0.55,
    mu_width: float = 0.14,
    very_high_speed: float | None = None,
    very_high_mu_midpoint: float = 1.00,
    very_high_mu_width: float = 0.06,
) -> torch.Tensor:
    """Alternate-foot gait cue whose cadence follows the taught safe speed.

    This is deliberately a weak shaping term: low traction receives a slower
    walking cadence, while high-traction high-command samples can discover a
    fast walk/light run instead of being locked to the old fixed 0.8-s gait.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    cmd_x = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    cap = _traction_speed_cap(
        env,
        low_speed=low_speed,
        high_speed=high_speed,
        mu_midpoint=mu_midpoint,
        mu_width=mu_width,
        very_high_speed=very_high_speed,
        very_high_mu_midpoint=very_high_mu_midpoint,
        very_high_mu_width=very_high_mu_width,
    )
    target_speed = torch.minimum(cmd_x, cap)
    speed_fraction = torch.clamp(target_speed / max(high_speed, 1e-6), 0.0, 1.0)
    period = slow_period + (fast_period - slow_period) * speed_fraction
    phase = torch.remainder(env.episode_length_buf * env.step_dt / period, 1.0)
    left_stance = phase < stance_threshold
    right_stance = torch.remainder(phase + 0.5, 1.0) < stance_threshold
    reward = (~(left_stance ^ is_contact[:, 0])).float()
    reward += (~(right_stance ^ is_contact[:, 1])).float()
    return reward * (cmd_x > 0.1).float()


def low_traction_touchdown_rate_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    mu_midpoint: float = 0.30,
    mu_width: float = 0.06,
    minimum_air_time: float = 0.08,
    command_threshold: float = 0.10,
    default_mu: float = 0.8,
) -> torch.Tensor:
    """Penalize deliberate touchdown events only while traction is low.

    The phase-matching gait reward can be satisfied by short, rapid steps.
    This sparse term directly prices each new step after a real swing interval,
    so the low-traction solution must lengthen its stance/swing cycle instead
    of compensating a shorter stride with higher cadence.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_contact = contact_sensor.data.current_contact_time[
        :, sensor_cfg.body_ids
    ]
    last_air = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    touchdown = (
        (current_contact > 0.0)
        & (current_contact <= 1.5 * env.step_dt)
        & (last_air >= minimum_air_time)
    )
    if hasattr(env, "ground_friction_mu_buf"):
        mu = env.ground_friction_mu_buf
    else:
        mu = torch.full(
            (env.num_envs,), default_mu, device=env.device
        )
    low_traction = torch.sigmoid(
        (mu_midpoint - mu) / max(mu_width, 1.0e-6)
    )
    command = torch.abs(
        env.command_manager.get_command(command_name)[:, 0]
    )
    active = (command > command_threshold).float()
    return low_traction * active * touchdown.float().sum(dim=1)
