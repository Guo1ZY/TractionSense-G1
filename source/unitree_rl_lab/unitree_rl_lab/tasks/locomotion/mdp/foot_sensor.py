# Copyright (c) 2022-2025, The Isaac Lab Project Developers / local foot-sensor extension.
# SPDX-License-Identifier: BSD-3-Clause
"""Foot-sensor observations aligned with zorn ``foot_sensor`` schema.

Simulation training uses Isaac Lab :class:`ContactSensor` on ankle-roll bodies
(vectorized, multi-env). The field semantics mirror the zorn ContactView /
ROS2 pipeline so that Real2Sim / Sim2Real can share the same observation
layout.

Zorn reference (host: ``/home/mosense/docker/zorn/workspace/foot_sensor``):
  - ContactView on ``left_ankle_roll_link`` / ``right_ankle_roll_link``
  - Physics dt = 0.005 s; collector rate ~20 Hz; ROS publisher ~50 Hz
  - Topics:
      /g1/left_foot/frame      Float32MultiArray[35]
      /g1/right_foot/frame     Float32MultiArray[35]
      /g1/left_foot/sensor15   Float32MultiArray[15]
      /g1/right_foot/sensor15  Float32MultiArray[15]
  - Units: forces in Newtons (N), torques in N·m, positions in m
  - frame35 layout (per foot):
      [0]    normal_force_mag
      [1]    tangent_force_mag
      [2]    total_force_mag
      [3:6]  cop_local xyz
      [6:9]  force_local_total
      [9:12] normal_force_local
      [12:15] tangent_force_local
      [15:18] torque_local
      [18]   contact_count
      [19]   friction_count
      [20:35] sensor15 (normal-force distribution over 15 virtual pads)

RL observation (this module) uses net contact forces on ankle bodies — the
same physical quantities as frame35[0:2] / force vectors — not the full 15-pt
map (kept for ROS schema alignment; optional later).

Left / right order is always ``[left, right]`` when body names are resolved
as ``["left_ankle_roll_link", "right_ankle_roll_link"]``.
"""

from __future__ import annotations

import operator
import sys
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

from unitree_rl_lab.sensors import HallFootSensor, HallFootSensorCfg
from unitree_rl_lab.sensors.hall_contact_distribution import (
    indexed_buffer_indices,
    sum_vectors_by_index,
)

_DETAILED_AUDIT_WARNED_ONCE: set[str] = set()

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Schema constants (shared with docs / deploy)
# ---------------------------------------------------------------------------

FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
"""Canonical L→R ankle body names (G1 29DoF)."""

FOOT_SIDE_ORDER = ("left", "right")

# Force normalization for policy obs. Standing load ~ body_weight/2 ≈ 150–250 N.
# scale=0.01 maps 100 N → 1.0 so typical magnitudes sit in ~[0, 3].
DEFAULT_FORCE_SCALE = 0.01

# Contact threshold in Newtons (soft / hard).
DEFAULT_CONTACT_FORCE_THRESHOLD = 5.0

# Zorn-aligned physics / topic metadata (documentation + Real2Sim).
ZORN_FOOT_SCHEMA = {
    "sim_dt": 0.005,
    "contact_update_period": 0.005,
    "collector_rate_hz": 20.0,
    "ros_publish_rate_hz": 50.0,
    "policy_step_dt": 0.02,  # sim_dt * decimation(4)
    "units": {"force": "N", "torque": "N.m", "position": "m"},
    "topics": {
        "left_frame": "/g1/left_foot/frame",
        "right_frame": "/g1/right_foot/frame",
        "left_sensor15": "/g1/left_foot/sensor15",
        "right_sensor15": "/g1/right_foot/sensor15",
    },
    "frame35_len": 35,
    "sensor15_len": 15,
    "rl_terms": {
        "foot_contact": "soft contact state L,R ∈ [0,1]",
        "foot_normal_force": "scaled |F_z| (world) L,R",
        "foot_tangent_force": "scaled |F_xy| (world) L,R (sim shear; avoid on actor if HW lacks shear)",
        "foot_load_ratio": "Fn share L,R (deployable from normal only)",
        "foot_planar_vel": "ankle planar speed L,R (proprio/FK)",
        "foot_sensor_valid": "1=fresh, 0=failed/stale",
        "foot_sensor_age": "normalized age of last frame",
        "foot_force_history": "short net-force history (optional compact)",
        "foot_friction_ratio": "ρ=||Ft||/(||Fn||+eps) critic-privileged (≠ μ)",
        "foot_slip_proxy": "contact-weighted foot planar speed L,R",
        "ground_friction_mu": "true μ critic/teacher only",
    },
    "schema_version": "foot_obs_v2",
}


def _foot_forces_w(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Net contact forces in world frame for configured foot bodies.

    Returns:
        Tensor of shape (num_envs, num_feet, 3).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]


def _filtered_total_contact_force_w(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return one dedicated foot sensor's filtered normal + friction force."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    normal_forces_w = contact_sensor.data.force_matrix_w
    friction_forces_w = contact_sensor.data.friction_forces_w
    if normal_forces_w is None or friction_forces_w is None:
        raise RuntimeError(
            f"ContactSensor '{sensor_cfg.name}' must configure a ground filter "
            "and track_friction_forces=True"
        )
    # Dedicated sensor layout is (env, one foot body, ground filters, xyz).
    if normal_forces_w.shape[1] != 1 or friction_forces_w.shape[1] != 1:
        raise RuntimeError(
            f"ContactSensor '{sensor_cfg.name}' must cover exactly one foot body; "
            f"got normal {tuple(normal_forces_w.shape)} and friction "
            f"{tuple(friction_forces_w.shape)}"
        )
    return (normal_forces_w + friction_forces_w).sum(dim=(1, 2))


def raw_foot_force_world_n(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return signed total ground contact forces in world frame, left then right.

    Returns:
        Tensor ``(num_envs, 2, 3)`` in Newtons.  Unlike
        :attr:`ContactSensor.data.net_forces_w`, this includes the filtered
        friction force as well as the filtered normal force.
    """
    left_force_w = _filtered_total_contact_force_w(env, left_sensor_cfg)
    right_force_w = _filtered_total_contact_force_w(env, right_sensor_cfg)
    return torch.stack((left_force_w, right_force_w), dim=1)


def raw_foot_force_local_n(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return signed left/right net contact force in each ankle-roll link frame.

    The input is the sum of Isaac Lab ContactSensor filtered normal and
    friction forces, in Newtons. ``quat_apply_inverse`` rotates each world-frame
    force into the matching ankle-roll link frame.  ``asset_cfg`` must resolve
    bodies in explicit
    ``[left_ankle_roll_link, right_ankle_roll_link]`` order.

    Returns:
        Tensor ``(num_envs, 6)`` in this exact order, still in Newtons:
        ``[left_Fx, left_Fy, left_Fz, right_Fx, right_Fy, right_Fz]``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    forces_w = raw_foot_force_world_n(env, left_sensor_cfg, right_sensor_cfg)
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    if forces_w.shape[1] != 2 or foot_quat_w.shape[1] != 2:
        raise RuntimeError(
            "raw_foot_force_local_n requires exactly two feet in left/right order; "
            f"got sensor shape {tuple(forces_w.shape)} and body quaternion shape "
            f"{tuple(foot_quat_w.shape)}"
        )
    return quat_apply_inverse(foot_quat_w, forces_w).reshape(env.num_envs, 6)


def normalized_raw_foot_force_local(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    robot_mass_kg: float | None = None,
    gravity_m_s2: float = 9.81,
) -> torch.Tensor:
    """Return local signed foot forces normalized by ``robot_mass * gravity``.

    With ``robot_mass_kg=None`` (default), the current total articulation mass
    is read after startup mass randomization and cached independently for every
    environment.  Supplying ``robot_mass_kg`` provides an explicit configurable
    normalization mass.  Clipping is intentionally configured on the
    corresponding ``ObservationTermCfg`` so it occurs immediately before the
    value is written to policy/critic history.
    """
    if gravity_m_s2 <= 0.0:
        raise ValueError(f"gravity_m_s2 must be positive, got {gravity_m_s2}")

    force_local = raw_foot_force_local_n(
        env, left_sensor_cfg, right_sensor_cfg, asset_cfg
    )
    asset: Articulation = env.scene[asset_cfg.name]
    if robot_mass_kg is None:
        # ObservationManager probes this function once before startup events.
        # Do not retain that nominal mass.  Once all RL managers exist, startup
        # mass DR has run before the first rollout observation, so the cached
        # value is the actual per-environment robot mass.
        robot_mass = getattr(env, "_raw_foot_force_robot_mass_kg", None)
        if robot_mass is None or not hasattr(env, "termination_manager"):
            robot_mass = asset.root_physx_view.get_masses().sum(dim=1, keepdim=True).to(
                device=force_local.device, dtype=force_local.dtype
            )
            if hasattr(env, "termination_manager"):
                env._raw_foot_force_robot_mass_kg = robot_mass
    else:
        if robot_mass_kg <= 0.0:
            raise ValueError(f"robot_mass_kg must be positive, got {robot_mass_kg}")
        robot_mass = torch.full(
            (env.num_envs, 1),
            float(robot_mass_kg),
            device=force_local.device,
            dtype=force_local.dtype,
        )
    return force_local / (robot_mass * float(gravity_m_s2))


def _foot_sensor_validity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return episode-level sensor validity as ``(N, 1)``."""
    if hasattr(env, "foot_sensor_valid_buf"):
        return env.foot_sensor_valid_buf.view(env.num_envs, 1)
    return torch.ones((env.num_envs, 1), device=env.device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Stateful/structured sensor domain randomization
# ---------------------------------------------------------------------------


def sample_structured_foot_sensor_noise(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
    gain_range: tuple[float, float] = (0.80, 1.20),
    normal_bias_range: tuple[float, float] = (-0.05, 0.05),
    tangent_bias_range: tuple[float, float] = (-0.03, 0.03),
    lowpass_alpha_range: tuple[float, float] = (0.25, 1.00),
    delay_steps_range: tuple[int, int] = (0, 3),
    sample_dropout_prob_range: tuple[float, float] = (0.0, 0.05),
    burst_dropout_prob_range: tuple[float, float] = (0.0, 0.02),
    burst_length_range: tuple[int, int] = (2, 10),
) -> None:
    """Sample episode-correlated foot-sensor errors.

    Unlike independent observation noise, this model keeps gain, zero bias,
    bandwidth and latency fixed for an episode, then generates short random
    dropouts over time.  Force values and biases use the policy's normalized
    force unit (``100 N == 1`` for the default scale).
    """

    n = env.num_envs
    device = env.device
    delay_lo, delay_hi = int(delay_steps_range[0]), int(delay_steps_range[1])
    burst_lo, burst_hi = int(burst_length_range[0]), int(burst_length_range[1])
    if delay_lo < 0 or delay_hi < delay_lo:
        raise ValueError(f"invalid delay_steps_range={delay_steps_range}")
    if burst_lo < 1 or burst_hi < burst_lo:
        raise ValueError(f"invalid burst_length_range={burst_length_range}")

    history_size = delay_hi + 1
    needs_init = (
        not hasattr(env, "structured_foot_gain_buf")
        or env.structured_foot_delay_history.shape[1] != history_size
    )
    if needs_init:
        env.structured_foot_gain_buf = torch.ones((n, 2, 2), device=device)
        env.structured_foot_bias_buf = torch.zeros((n, 2, 2), device=device)
        env.structured_foot_lowpass_alpha_buf = torch.ones((n, 1, 1), device=device)
        env.structured_foot_delay_steps_buf = torch.zeros(n, device=device, dtype=torch.long)
        env.structured_foot_sample_dropout_prob_buf = torch.zeros(n, device=device)
        env.structured_foot_burst_dropout_prob_buf = torch.zeros(n, device=device)
        env.structured_foot_burst_length_min_buf = torch.full(
            (n,), burst_lo, device=device, dtype=torch.long
        )
        env.structured_foot_burst_length_max_buf = torch.full(
            (n,), burst_hi, device=device, dtype=torch.long
        )
        env.structured_foot_burst_remaining_buf = torch.zeros(n, device=device, dtype=torch.long)
        env.structured_foot_filter_buf = torch.zeros((n, 2, 2), device=device)
        env.structured_foot_delay_history = torch.zeros(
            (n, history_size, 2, 2), device=device
        )
        env.structured_foot_initialized_buf = torch.zeros(n, device=device, dtype=torch.bool)
        env.structured_foot_last_step_buf = torch.full(
            (n,), -1, device=device, dtype=torch.long
        )
        env.structured_foot_current_valid_buf = torch.ones(n, device=device)
        env.structured_foot_current_age_buf = torch.zeros(n, device=device)
        env.structured_foot_packet_cache = {}

    if env_ids is None:
        env_ids = torch.arange(n, device=device)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    def uniform(bounds: tuple[float, float], shape: tuple[int, ...]) -> torch.Tensor:
        lo, hi = float(bounds[0]), float(bounds[1])
        return lo + (hi - lo) * torch.rand(shape, device=device)

    count = env_ids.numel()
    env.structured_foot_gain_buf[env_ids] = uniform(gain_range, (count, 2, 2))
    normal_bias = uniform(normal_bias_range, (count, 2))
    tangent_bias = uniform(tangent_bias_range, (count, 2))
    env.structured_foot_bias_buf[env_ids, :, 0] = normal_bias
    env.structured_foot_bias_buf[env_ids, :, 1] = tangent_bias
    env.structured_foot_lowpass_alpha_buf[env_ids] = uniform(
        lowpass_alpha_range, (count, 1, 1)
    )
    env.structured_foot_delay_steps_buf[env_ids] = torch.randint(
        delay_lo, delay_hi + 1, (count,), device=device
    )
    env.structured_foot_sample_dropout_prob_buf[env_ids] = uniform(
        sample_dropout_prob_range, (count,)
    )
    env.structured_foot_burst_dropout_prob_buf[env_ids] = uniform(
        burst_dropout_prob_range, (count,)
    )
    env.structured_foot_burst_length_min_buf[env_ids] = burst_lo
    env.structured_foot_burst_length_max_buf[env_ids] = burst_hi
    env.structured_foot_burst_remaining_buf[env_ids] = 0
    env.structured_foot_filter_buf[env_ids] = 0.0
    env.structured_foot_delay_history[env_ids] = 0.0
    env.structured_foot_initialized_buf[env_ids] = False
    env.structured_foot_last_step_buf[env_ids] = -1
    env.structured_foot_current_valid_buf[env_ids] = 1.0
    env.structured_foot_current_age_buf[env_ids] = 0.0

    # Keep the existing deploy-health observation buffers synchronized.
    if not hasattr(env, "foot_sensor_valid_buf"):
        env.foot_sensor_valid_buf = torch.ones(n, device=device)
        env.foot_sensor_age_buf = torch.zeros(n, device=device)
    env.foot_sensor_valid_buf[env_ids] = 1.0
    env.foot_sensor_age_buf[env_ids] = 0.0


def _structured_foot_packet(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_scale: float = DEFAULT_FORCE_SCALE,
) -> dict[str, torch.Tensor]:
    """Return one coherent delayed/noisy packet, updating at most once per step."""

    if not hasattr(env, "structured_foot_gain_buf"):
        sample_structured_foot_sensor_noise(env)

    step = getattr(env, "episode_length_buf", None)
    if step is None:
        step = torch.full(
            (env.num_envs,), int(getattr(env, "episode_length", 0)),
            device=env.device, dtype=torch.long,
        )
    else:
        step = step.to(device=env.device, dtype=torch.long)
    changed = env.structured_foot_last_step_buf != step
    ids = torch.nonzero(changed, as_tuple=False).squeeze(-1)

    if ids.numel() > 0:
        forces = _foot_forces_w(env, sensor_cfg)
        fn = torch.abs(forces[:, :, 2]) * force_scale
        ft = torch.linalg.norm(forces[:, :, :2], dim=-1) * force_scale
        calibrated = torch.stack((fn, ft), dim=-1)
        calibrated = torch.clamp(
            calibrated * env.structured_foot_gain_buf + env.structured_foot_bias_buf,
            min=0.0,
        )

        initialized = env.structured_foot_initialized_buf[ids]
        previous = env.structured_foot_filter_buf[ids]
        alpha = env.structured_foot_lowpass_alpha_buf[ids]
        filtered = alpha * calibrated[ids] + (1.0 - alpha) * previous
        filtered = torch.where(initialized[:, None, None], filtered, calibrated[ids])
        env.structured_foot_filter_buf[ids] = filtered

        history = env.structured_foot_delay_history
        if history.shape[1] > 1:
            history[ids, 1:] = history[ids, :-1].clone()
        history[ids, 0] = filtered
        fresh_ids = ids[~initialized]
        if fresh_ids.numel() > 0:
            history[fresh_ids] = env.structured_foot_filter_buf[fresh_ids, None].expand(
                -1, history.shape[1], -1, -1
            )
        env.structured_foot_initialized_buf[ids] = True

        remaining = env.structured_foot_burst_remaining_buf[ids]
        continuing = remaining > 0
        remaining = torch.clamp(remaining - 1, min=0)
        start_burst = (~continuing) & (
            torch.rand(ids.numel(), device=env.device)
            < env.structured_foot_burst_dropout_prob_buf[ids]
        )
        if start_burst.any():
            lo = env.structured_foot_burst_length_min_buf[ids[start_burst]]
            hi = env.structured_foot_burst_length_max_buf[ids[start_burst]]
            span = hi - lo + 1
            length = lo + torch.floor(
                torch.rand(int(start_burst.sum().item()), device=env.device) * span
            ).to(torch.long)
            remaining[start_burst] = torch.clamp(length - 1, min=0)
        env.structured_foot_burst_remaining_buf[ids] = remaining
        sample_drop = (
            torch.rand(ids.numel(), device=env.device)
            < env.structured_foot_sample_dropout_prob_buf[ids]
        )
        dropped = continuing | start_burst | sample_drop
        valid = (~dropped).to(torch.float32)
        env.structured_foot_current_valid_buf[ids] = valid
        dt = float(getattr(env, "step_dt", 0.02))
        env.structured_foot_current_age_buf[ids] = torch.where(
            valid > 0.5,
            torch.zeros_like(valid),
            env.structured_foot_current_age_buf[ids] + dt,
        )
        env.foot_sensor_valid_buf[ids] = valid
        env.foot_sensor_age_buf[ids] = env.structured_foot_current_age_buf[ids]
        env.structured_foot_last_step_buf[ids] = step[ids]

        row = torch.arange(env.num_envs, device=env.device)
        delayed = history[row, env.structured_foot_delay_steps_buf]
        valid_all = env.structured_foot_current_valid_buf[:, None]
        delayed_fn = delayed[:, :, 0] * valid_all
        delayed_ft = delayed[:, :, 1] * valid_all
        contact = torch.sigmoid(
            ((delayed_fn / max(force_scale, 1.0e-8)) - DEFAULT_CONTACT_FORCE_THRESHOLD) * 2.0
        ) * valid_all
        total = delayed_fn.sum(dim=-1, keepdim=True)
        load = delayed_fn / (total + force_scale)
        load = valid_all * load + (1.0 - valid_all) * 0.5
        env.structured_foot_packet_cache = {
            "normal": delayed_fn,
            "tangent": delayed_ft,
            "contact": contact,
            "load": torch.clamp(load, 0.0, 1.0),
            "valid": env.structured_foot_current_valid_buf[:, None],
            "age": env.structured_foot_current_age_buf[:, None],
        }

    return env.structured_foot_packet_cache


def structured_foot_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = DEFAULT_CONTACT_FORCE_THRESHOLD,
    soft: bool = True,
    soft_scale: float = 2.0,
) -> torch.Tensor:
    packet = _structured_foot_packet(env, sensor_cfg)
    if soft and threshold == DEFAULT_CONTACT_FORCE_THRESHOLD and soft_scale == 2.0:
        return packet["contact"]
    fn_n = packet["normal"] / max(DEFAULT_FORCE_SCALE, 1.0e-8)
    if soft:
        return torch.sigmoid((fn_n - threshold) * soft_scale) * packet["valid"]
    return (fn_n > threshold).to(torch.float32) * packet["valid"]


def structured_foot_normal_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    scale: float = DEFAULT_FORCE_SCALE,
) -> torch.Tensor:
    if abs(scale - DEFAULT_FORCE_SCALE) > 1.0e-9:
        return _structured_foot_packet(env, sensor_cfg)["normal"] * (scale / DEFAULT_FORCE_SCALE)
    return _structured_foot_packet(env, sensor_cfg)["normal"]


def structured_foot_tangent_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    scale: float = DEFAULT_FORCE_SCALE,
) -> torch.Tensor:
    if abs(scale - DEFAULT_FORCE_SCALE) > 1.0e-9:
        return _structured_foot_packet(env, sensor_cfg)["tangent"] * (scale / DEFAULT_FORCE_SCALE)
    return _structured_foot_packet(env, sensor_cfg)["tangent"]


def structured_foot_friction_ratio(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    eps: float = 5.0,
    clip_max: float = 2.0,
) -> torch.Tensor:
    packet = _structured_foot_packet(env, sensor_cfg)
    ratio = packet["tangent"] / (packet["normal"] + eps * DEFAULT_FORCE_SCALE)
    return torch.clamp(ratio, 0.0, clip_max) * packet["valid"]


def structured_foot_load_ratio(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    eps: float = 5.0,
) -> torch.Tensor:
    del eps
    return _structured_foot_packet(env, sensor_cfg)["load"]


def structured_foot_sensor_valid(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    default_valid: float = 1.0,
) -> torch.Tensor:
    del default_valid
    return _structured_foot_packet(env, sensor_cfg)["valid"]


def structured_foot_sensor_age(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    age_scale: float = 0.25,
    clip_max: float = 1.0,
) -> torch.Tensor:
    age = _structured_foot_packet(env, sensor_cfg)["age"] / max(age_scale, 1.0e-6)
    return torch.clamp(age, 0.0, clip_max)


def foot_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = DEFAULT_CONTACT_FORCE_THRESHOLD,
    soft: bool = True,
    soft_scale: float = 2.0,
    respect_sensor_valid: bool = False,
) -> torch.Tensor:
    """Per-foot contact state (left, right).

    Soft mode uses a sigmoid of force magnitude for differentiable-ish DR;
    hard mode is a binary threshold. Both map to zorn ``contact_count > 0``.

    Shape: (num_envs, num_feet)
    """
    forces = _foot_forces_w(env, sensor_cfg)
    mag = torch.linalg.norm(forces, dim=-1)
    if soft:
        value = torch.sigmoid((mag - threshold) * soft_scale)
    else:
        value = (mag > threshold).to(dtype=torch.float32)
    if respect_sensor_valid:
        value = value * _foot_sensor_validity(env)
    return value


def foot_normal_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    scale: float = DEFAULT_FORCE_SCALE,
    respect_sensor_valid: bool = False,
) -> torch.Tensor:
    """Scaled world-Z contact force magnitude per foot (proxy for normal load).

    Aligns with zorn ``normal_force_mag`` under flat-ground assumption.
    Shape: (num_envs, num_feet)
    """
    forces = _foot_forces_w(env, sensor_cfg)
    # ContactSensor net force is primarily normal; |Fz| is a stable flat-ground proxy.
    value = torch.abs(forces[:, :, 2]) * scale
    if respect_sensor_valid:
        value = value * _foot_sensor_validity(env)
    return value


def foot_tangent_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    scale: float = DEFAULT_FORCE_SCALE,
    respect_sensor_valid: bool = False,
) -> torch.Tensor:
    """Scaled world-XY force magnitude per foot (proxy for tangential / friction).

    Aligns with zorn ``tangent_force_mag``.
    Shape: (num_envs, num_feet)
    """
    forces = _foot_forces_w(env, sensor_cfg)
    value = torch.linalg.norm(forces[:, :, :2], dim=-1) * scale
    if respect_sensor_valid:
        value = value * _foot_sensor_validity(env)
    return value


def foot_force_history(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    scale: float = DEFAULT_FORCE_SCALE,
    history_steps: int = 3,
) -> torch.Tensor:
    """Short net-force history from ContactSensor (most recent first).

    Flattens (T, feet, 3) → (T * feet * 3). When sensor history is shorter
    than ``history_steps``, available steps are used (padded by repeating
    oldest if needed).

    Aligns with zorn short temporal context without full 15-pt map.
    Shape: (num_envs, history_steps * num_feet * 3)
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # (N, T, B, 3) — index 0 is most recent
    hist = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    t_avail = hist.shape[1]
    t_use = min(history_steps, t_avail)
    chunk = hist[:, :t_use] * scale
    if t_use < history_steps:
        # pad with oldest available sample
        pad = chunk[:, -1:].expand(-1, history_steps - t_use, -1, -1)
        chunk = torch.cat([chunk, pad], dim=1)
    return chunk.reshape(env.num_envs, -1)


def foot_force_vector(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    scale: float = DEFAULT_FORCE_SCALE,
) -> torch.Tensor:
    """Scaled 3D net force per foot, flattened L then R (Fx,Fy,Fz, ...).

    Shape: (num_envs, num_feet * 3)
    """
    forces = _foot_forces_w(env, sensor_cfg) * scale
    return forces.reshape(env.num_envs, -1)


def foot_friction_ratio(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    eps: float = 1.0,
    clip_max: float = 5.0,
    respect_sensor_valid: bool = False,
) -> torch.Tensor:
    """Friction-cone utilization ρ = ||F_t|| / (||F_n|| + eps) per foot (L,R).

    Aligns with zorn ``tangent_force_mag / normal_force_mag``. Not μ itself.
    Prefer **critic-only** if real hardware has no reliable shear channel.

    Shape: (num_envs, num_feet)
    """
    forces = _foot_forces_w(env, sensor_cfg)
    fn = torch.abs(forces[:, :, 2])
    ft = torch.linalg.norm(forces[:, :, :2], dim=-1)
    rho = ft / (fn + eps)
    value = torch.clamp(rho, 0.0, clip_max)
    if respect_sensor_valid:
        value = value * _foot_sensor_validity(env)
    return value


def foot_slip_proxy(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = DEFAULT_CONTACT_FORCE_THRESHOLD,
    soft_scale: float = 0.5,
    vel_scale: float = 1.0,
    clip_max: float = 3.0,
) -> torch.Tensor:
    """Contact-weighted planar foot speed (slip proxy) per foot (L,R).

    ``soft_contact * ||v_foot_xy|| / vel_scale``. Deployable if foot/body
    velocity is available; in sim uses ankle body linear velocity.

    Shape: (num_envs, num_feet)
    """
    from isaaclab.assets import Articulation

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    fn = torch.abs(forces[:, :, 2])
    soft_contact = torch.sigmoid((fn - force_threshold) * soft_scale)

    asset: Articulation = env.scene[asset_cfg.name]
    # Prefer sensor body_ids order when asset uses the same foot bodies.
    body_ids = asset_cfg.body_ids if asset_cfg.body_ids is not None else sensor_cfg.body_ids
    foot_vel = torch.linalg.norm(asset.data.body_lin_vel_w[:, body_ids, :2], dim=-1)
    slip = soft_contact * foot_vel / max(vel_scale, 1e-6)
    return torch.clamp(slip, 0.0, clip_max)


def foot_load_ratio(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    eps: float = 1.0,
    respect_sensor_valid: bool = False,
) -> torch.Tensor:
    """Per-foot load share of total normal force (L, R). Deployable from Fn only.

    ``Fn_i / (sum_j Fn_j + eps)``. Useful gait / weight-shift cue without shear.

    Shape: (num_envs, num_feet)
    """
    forces = _foot_forces_w(env, sensor_cfg)
    fn = torch.abs(forces[:, :, 2])
    total = torch.sum(fn, dim=-1, keepdim=True) + eps
    value = torch.clamp(fn / total, 0.0, 1.0)
    if respect_sensor_valid:
        valid = _foot_sensor_validity(env)
        value = valid * value + (1.0 - valid) * 0.5
    return value


def foot_planar_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vel_scale: float = 1.0,
    clip_max: float = 3.0,
) -> torch.Tensor:
    """Ankle planar speed ||v_xy|| / vel_scale per foot (L,R).

    Deployable via FK/body velocity on sim; real robot needs ankle kinematics.
    Prefer over true Ft for actor when hardware has only normal pressure.

    Shape: (num_envs, num_feet)
    """
    from isaaclab.assets import Articulation

    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    foot_vel = torch.linalg.norm(asset.data.body_lin_vel_w[:, body_ids, :2], dim=-1)
    return torch.clamp(foot_vel / max(vel_scale, 1e-6), 0.0, clip_max)


def foot_sensor_valid(
    env: ManagerBasedRLEnv,
    default_valid: float = 1.0,
) -> torch.Tensor:
    """Sensor validity flag in [0,1]. Train: always 1.0 unless DR injects dropout.

    Deploy: bridge sets 0 when stale/missing so policy can distinguish
    "zero force" from "sensor failed" (zeros alone are ambiguous).

    Shape: (num_envs, 1)
    """
    if hasattr(env, "foot_sensor_valid_buf"):
        return env.foot_sensor_valid_buf.view(env.num_envs, 1)
    return torch.full((env.num_envs, 1), default_valid, device=env.device, dtype=torch.float32)


def foot_sensor_age(
    env: ManagerBasedRLEnv,
    age_scale: float = 0.25,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """Normalized sensor age (0=fresh, 1=stale≥age_scale seconds).

    Train default 0. Deploy: ``min(age_sec / age_scale, 1)``.

    Shape: (num_envs, 1)
    """
    if hasattr(env, "foot_sensor_age_buf"):
        age = env.foot_sensor_age_buf.view(env.num_envs, 1) / max(age_scale, 1e-6)
        return torch.clamp(age, 0.0, clip_max)
    return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)


def ground_friction_mu(
    env: ManagerBasedRLEnv,
    default_mu: float = 0.8,
    clip_max: float = 2.0,
) -> torch.Tensor:
    """Privileged ground / effective friction μ (critic/teacher only).

    Filled by friction DR event into ``env.ground_friction_mu_buf``.
    This is true μ, NOT Ft/Fn utilization.

    Shape: (num_envs, 1)
    """
    if hasattr(env, "ground_friction_mu_buf"):
        return torch.clamp(env.ground_friction_mu_buf.view(env.num_envs, 1), 0.0, clip_max)
    return torch.full((env.num_envs, 1), default_mu, device=env.device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Flexible magnetic Hall sole (physical Scheme A / deformable Scheme B)
# ---------------------------------------------------------------------------


def _dedicated_hall_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read total normal+friction force and mean contact point for one foot."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    data = sensor.data
    if data.force_matrix_w is None:
        normal = data.net_forces_w.sum(dim=1)
    else:
        normal = torch.nan_to_num(data.force_matrix_w).sum(dim=(1, 2))
    friction = (
        torch.nan_to_num(data.friction_forces_w).sum(dim=(1, 2))
        if data.friction_forces_w is not None
        else torch.zeros_like(normal)
    )
    force = normal + friction
    if data.contact_pos_w is None:
        point = torch.full_like(force, torch.nan)
    else:
        positions = data.contact_pos_w.reshape(env.num_envs, -1, 3)
        finite = torch.isfinite(positions).all(dim=-1, keepdim=True)
        count = finite.sum(dim=1).clamp_min(1)
        point = torch.where(finite, positions, 0.0).sum(dim=1) / count
        point = torch.where((finite.sum(dim=1) > 0), point, torch.full_like(point, torch.nan))
    return force, point


def _audit_detailed_force_sum(
    *,
    label: str,
    raw_forces_w: torch.Tensor,
    sensor_rows: torch.Tensor,
    reported_forces_w: torch.Tensor,
    num_envs: int,
    atol: float,
    rtol: float,
    fail: bool = True,
) -> None:
    """Fail closed if detailed and ContactSensor aggregate forces disagree."""

    detailed_sum = sum_vectors_by_index(
        raw_forces_w,
        sensor_rows,
        output_count=num_envs,
    )
    reported = torch.nan_to_num(reported_forces_w).sum(dim=(1, 2))
    if reported.shape != detailed_sum.shape:
        raise RuntimeError(
            f"{label}: reported force shape {tuple(reported.shape)} does not match "
            f"detailed sum {tuple(detailed_sum.shape)}"
        )
    close = torch.isclose(detailed_sum, reported, atol=atol, rtol=rtol)
    if not bool(torch.all(close).item()):
        max_abs = float(torch.max(torch.abs(detailed_sum - reported)).item())
        if not fail:
            if label not in _DETAILED_AUDIT_WARNED_ONCE:
                _DETAILED_AUDIT_WARNED_ONCE.add(label)
                print(
                    f"[hall-detailed-contact-warning] {label}: raw detailed "
                    f"force disagrees with ContactSensor aggregate "
                    f"(max_abs_error={max_abs:.6g} N, atol={atol:g}, "
                    f"rtol={rtol:g}); continuing because the evaluation-only "
                    "audit mismatch guard is disabled",
                    file=sys.stderr,
                    flush=True,
                )
            return
        raise RuntimeError(
            f"{label}: raw detailed force does not reproduce ContactSensor aggregate "
            f"(max_abs_error={max_abs:.6g} N, atol={atol:g}, rtol={rtol:g})"
        )


def _detailed_hall_contact_samples(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    *,
    foot_index: int,
    hall_cfg: HallFootSensorCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read one foot's raw PhysX normal patches and friction anchors.

    Returns ``(forces_w, points_w, env_indices, foot_indices)``.  Normal and
    friction streams are deliberately gathered independently: Isaac Sim 5.1
    does not promise a one-to-one index relation between contact patches and
    friction anchors.
    """

    if foot_index not in (0, 1):
        raise ValueError("foot_index must be 0 (left) or 1 (right)")
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if sensor.num_bodies != 1:
        raise RuntimeError(
            f"detailed Hall ContactSensor {sensor_cfg.name!r} must resolve exactly one body"
        )
    if not sensor.cfg.track_contact_points or not sensor.cfg.track_friction_forces:
        raise RuntimeError(
            f"detailed Hall ContactSensor {sensor_cfg.name!r} requires "
            "track_contact_points=True and track_friction_forces=True"
        )

    # Accessing data first advances the ContactSensor's timestamp/buffers.  The
    # raw RigidContactView queries below then read the exact same physics step.
    data = sensor.data
    if data.force_matrix_w is None or data.friction_forces_w is None:
        raise RuntimeError(
            f"detailed Hall ContactSensor {sensor_cfg.name!r} requires filtered "
            "normal and friction force buffers"
        )
    view = sensor.contact_physx_view
    if view.filter_count < 1 or view.max_contact_data_count < 1:
        raise RuntimeError(
            f"detailed Hall ContactSensor {sensor_cfg.name!r} has no raw contact capacity"
        )
    if view.sensor_count != env.num_envs:
        raise RuntimeError(
            f"detailed Hall ContactSensor {sensor_cfg.name!r} has sensor_count="
            f"{view.sensor_count}, expected {env.num_envs}"
        )
    physics_dt = float(getattr(sensor, "_sim_physics_dt", 0.0))
    if not physics_dt > 0.0:
        raise RuntimeError("ContactSensor physics dt is unavailable for SI impulse-to-force conversion")

    normal_raw = view.get_contact_data(dt=physics_dt)
    if not isinstance(normal_raw, tuple) or len(normal_raw) != 6:
        raise RuntimeError("Isaac Sim 5.1 get_contact_data() contract changed")
    normal_magnitudes, normal_points_w, normal_directions_w, _, normal_counts, normal_starts = normal_raw
    expected_count_shape = (env.num_envs, view.filter_count)
    if tuple(normal_counts.shape) != expected_count_shape or normal_starts.shape != normal_counts.shape:
        raise RuntimeError(
            "unexpected detailed normal count/start shape: "
            f"{tuple(normal_counts.shape)}, expected {expected_count_shape}"
        )
    if hall_cfg.detailed_contact_fail_on_buffer_saturation:
        used = int(normal_counts.to(dtype=torch.long).sum().item())
        if used >= int(view.max_contact_data_count):
            raise RuntimeError(
                f"{sensor_cfg.name}: detailed normal buffer is full ({used}/"
                f"{view.max_contact_data_count}); increase max_contact_data_count_per_prim"
            )
    normal_rows, normal_indices = indexed_buffer_indices(
        normal_counts,
        normal_starts,
        buffer_length=normal_magnitudes.shape[0],
    )
    normal_magnitudes = normal_magnitudes.index_select(0, normal_indices).reshape(-1, 1)
    normal_points_w = normal_points_w.index_select(0, normal_indices)
    normal_directions_w = normal_directions_w.index_select(0, normal_indices)
    normal_forces_w = normal_magnitudes * normal_directions_w
    if not (
        torch.isfinite(normal_forces_w).all()
        and torch.isfinite(normal_points_w).all()
    ):
        raise RuntimeError(f"{sensor_cfg.name}: non-finite referenced normal contact sample")
    _audit_detailed_force_sum(
        label=f"{sensor_cfg.name} normal",
        raw_forces_w=normal_forces_w,
        sensor_rows=normal_rows,
        reported_forces_w=data.force_matrix_w,
        num_envs=env.num_envs,
        atol=hall_cfg.detailed_contact_force_atol,
        rtol=hall_cfg.detailed_contact_force_rtol,
        fail=hall_cfg.detailed_contact_fail_on_audit_mismatch,
    )

    friction_raw = view.get_friction_data(dt=physics_dt)
    if not isinstance(friction_raw, tuple) or len(friction_raw) != 4:
        raise RuntimeError("Isaac Sim 5.1 get_friction_data() contract changed")
    friction_forces_buffer, friction_points_buffer, friction_counts, friction_starts = friction_raw
    if tuple(friction_counts.shape) != expected_count_shape or friction_starts.shape != friction_counts.shape:
        raise RuntimeError(
            "unexpected detailed friction count/start shape: "
            f"{tuple(friction_counts.shape)}, expected {expected_count_shape}"
        )
    if hall_cfg.detailed_contact_fail_on_buffer_saturation:
        used = int(friction_counts.to(dtype=torch.long).sum().item())
        if used >= int(view.max_contact_data_count):
            raise RuntimeError(
                f"{sensor_cfg.name}: detailed friction buffer is full ({used}/"
                f"{view.max_contact_data_count}); increase max_contact_data_count_per_prim"
            )
    friction_rows, friction_indices = indexed_buffer_indices(
        friction_counts,
        friction_starts,
        buffer_length=friction_forces_buffer.shape[0],
    )
    friction_forces_w = friction_forces_buffer.index_select(0, friction_indices)
    friction_points_w = friction_points_buffer.index_select(0, friction_indices)
    if not (
        torch.isfinite(friction_forces_w).all()
        and torch.isfinite(friction_points_w).all()
    ):
        raise RuntimeError(f"{sensor_cfg.name}: non-finite referenced friction contact sample")
    _audit_detailed_force_sum(
        label=f"{sensor_cfg.name} friction",
        raw_forces_w=friction_forces_w,
        sensor_rows=friction_rows,
        reported_forces_w=data.friction_forces_w,
        num_envs=env.num_envs,
        atol=hall_cfg.detailed_contact_force_atol,
        rtol=hall_cfg.detailed_contact_force_rtol,
        fail=hall_cfg.detailed_contact_fail_on_audit_mismatch,
    )

    forces_w = torch.cat((normal_forces_w, friction_forces_w), dim=0)
    points_w = torch.cat((normal_points_w, friction_points_w), dim=0)
    env_indices = torch.cat((normal_rows, friction_rows), dim=0).to(dtype=torch.long)
    foot_indices = torch.full_like(env_indices, foot_index)
    return forces_w, points_w, env_indices, foot_indices


def _hall_foot_packet(
    env: ManagerBasedRLEnv,
    hall_cfg: HallFootSensorCfg,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    left_contact_sensor_cfg: SceneEntityCfg | None = None,
    right_contact_sensor_cfg: SceneEntityCfg | None = None,
) -> dict[str, torch.Tensor]:
    """Return one coherent Hall packet per policy step, cached across terms."""
    sensor = getattr(env, "_hall_foot_sensor", None)
    if (
        sensor is None
        or sensor.cfg != hall_cfg
        or sensor.num_envs != env.num_envs
        or sensor.device != torch.device(env.device)
    ):
        configured_seed = getattr(getattr(env, "cfg", None), "seed", None)
        if configured_seed is None:
            raise RuntimeError(
                "HallFootSensor requires env.cfg.seed so sensor noise/domain "
                "randomization follows the effective training/evaluation seed"
            )
        try:
            hall_seed = operator.index(configured_seed)
        except TypeError as exc:
            raise TypeError(
                "env.cfg.seed must be an integer for deterministic Hall sensor "
                f"randomization, got {configured_seed!r}"
            ) from exc
        if hall_seed < 0:
            raise ValueError(
                f"env.cfg.seed must be non-negative after CLI seed resolution, got {hall_seed}"
            )
        sensor = HallFootSensor(hall_cfg)
        magnet_pose_provider = None
        if hall_cfg.implementation_mode == "deformable":
            magnet_pose_provider = getattr(env, "_hall_magnet_pose_provider", None)
            if magnet_pose_provider is None:
                raise RuntimeError(
                    "Scheme B requires HallSoleAttachmentAction and the left/right "
                    "magnetized TPU DeformableObject assets; no pose provider was initialized"
                )
        sensor.initialize(
            env.num_envs,
            env.device,
            magnet_pose_provider=magnet_pose_provider,
            seed=hall_seed,
        )
        env._hall_foot_sensor = sensor
        # Expose the effective seed for runtime audits and run provenance.  In
        # distributed training ``train.py`` has already folded local_rank into
        # ``env.cfg.seed``, so every rank now receives an independent stream.
        env._hall_foot_sensor_seed = hall_seed
        env._hall_foot_sensor_step = -1
        env._hall_foot_prev_episode_length = env.episode_length_buf.clone()

    step = int(getattr(env, "common_step_counter", 0))
    if getattr(env, "_hall_foot_sensor_step", -1) != step:
        previous_length = env._hall_foot_prev_episode_length
        reset_ids = torch.nonzero(env.episode_length_buf < previous_length, as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            sensor.reset(reset_ids)
        env._hall_foot_prev_episode_length.copy_(env.episode_length_buf)

        asset: Articulation = env.scene[asset_cfg.name]
        foot_positions_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
        foot_quaternions_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
        if foot_positions_w.shape[1] != 2:
            raise RuntimeError("asset_cfg must resolve left then right ankle-roll bodies")

        local_contact_force_f = None
        if left_contact_sensor_cfg is not None and right_contact_sensor_cfg is not None:
            if hall_cfg.contact_distribution_mode == "detailed":
                left_samples = _detailed_hall_contact_samples(
                    env,
                    left_contact_sensor_cfg,
                    foot_index=0,
                    hall_cfg=hall_cfg,
                )
                right_samples = _detailed_hall_contact_samples(
                    env,
                    right_contact_sensor_cfg,
                    foot_index=1,
                    hall_cfg=hall_cfg,
                )
                point_forces_w = torch.cat((left_samples[0], right_samples[0]), dim=0)
                contact_points_w = torch.cat((left_samples[1], right_samples[1]), dim=0)
                contact_env_indices = torch.cat((left_samples[2], right_samples[2]), dim=0)
                contact_foot_indices = torch.cat((left_samples[3], right_samples[3]), dim=0)
                local_contact_force_f = sensor.distribute_detailed_contact_forces(
                    foot_positions_w=foot_positions_w,
                    foot_quaternions_w=foot_quaternions_w,
                    point_forces_w=point_forces_w,
                    contact_points_w=contact_points_w,
                    contact_env_indices=contact_env_indices,
                    contact_foot_indices=contact_foot_indices,
                )
                contact_force_w = None
                contact_point_w = None
            else:
                left_force, left_point = _dedicated_hall_contact(env, left_contact_sensor_cfg)
                right_force, right_point = _dedicated_hall_contact(env, right_contact_sensor_cfg)
                contact_force_w = torch.stack((left_force, right_force), dim=1)
                contact_point_w = torch.stack((left_point, right_point), dim=1)
        else:
            if hall_cfg.contact_distribution_mode == "detailed":
                raise RuntimeError(
                    "contact_distribution_mode='detailed' requires one dedicated "
                    "filtered ContactSensor per foot; refusing aggregate fallback"
                )
            contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
            contact_force_w = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :]
            contact_point_w = None

        sensor.update(
            env.step_dt,
            foot_positions_w=foot_positions_w,
            foot_quaternions_w=foot_quaternions_w,
            contact_force_w=contact_force_w,
            contact_point_w=contact_point_w,
            local_contact_force_f=local_contact_force_f,
        )
        debug = sensor.get_debug_data()
        env._hall_foot_packet_cache = {
            "raw": sensor.get_raw_data(),
            "filtered": sensor.get_filtered_data(),
            "normalized": sensor.get_policy_observation(),
            "norm": debug["magnetic_norm"],
            "delta": debug["magnetic_delta"],
            "deformation": debug["local_deformation"],
            "valid": sensor.get_policy_valid_mask(),
            "age": debug["sample_age"],
            "period": sensor.get_reported_sample_period(),
        }
        env._hall_foot_sensor_step = step
    return env._hall_foot_packet_cache


def hall_magnetic_array(
    env: ManagerBasedRLEnv,
    hall_cfg: HallFootSensorCfg,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    left_contact_sensor_cfg: SceneEntityCfg | None = None,
    right_contact_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Normalized Hall delta, flattened left ``SxXYZ`` then right ``SxXYZ``."""
    packet = _hall_foot_packet(
        env,
        hall_cfg,
        asset_cfg,
        contact_sensor_cfg,
        left_contact_sensor_cfg,
        right_contact_sensor_cfg,
    )
    return packet["normalized"].reshape(env.num_envs, -1)


def hall_magnetic_raw(
    env: ManagerBasedRLEnv,
    hall_cfg: HallFootSensorCfg,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    left_contact_sensor_cfg: SceneEntityCfg | None = None,
    right_contact_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    return _hall_foot_packet(
        env, hall_cfg, asset_cfg, contact_sensor_cfg, left_contact_sensor_cfg, right_contact_sensor_cfg
    )["raw"]


def hall_magnetic_delta(
    env: ManagerBasedRLEnv,
    hall_cfg: HallFootSensorCfg,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    left_contact_sensor_cfg: SceneEntityCfg | None = None,
    right_contact_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    return _hall_foot_packet(
        env, hall_cfg, asset_cfg, contact_sensor_cfg, left_contact_sensor_cfg, right_contact_sensor_cfg
    )["delta"]


def hall_local_deformation(
    env: ManagerBasedRLEnv,
    hall_cfg: HallFootSensorCfg,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    left_contact_sensor_cfg: SceneEntityCfg | None = None,
    right_contact_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    return _hall_foot_packet(
        env, hall_cfg, asset_cfg, contact_sensor_cfg, left_contact_sensor_cfg, right_contact_sensor_cfg
    )["deformation"]


def hall_sensor_valid_lr(
    env: ManagerBasedRLEnv,
    hall_cfg: HallFootSensorCfg,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    left_contact_sensor_cfg: SceneEntityCfg | None = None,
    right_contact_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    valid = _hall_foot_packet(
        env, hall_cfg, asset_cfg, contact_sensor_cfg, left_contact_sensor_cfg, right_contact_sensor_cfg
    )["valid"]
    return valid.all(dim=-1).to(torch.float32)


def hall_sensor_age_lr(
    env: ManagerBasedRLEnv,
    hall_cfg: HallFootSensorCfg,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    left_contact_sensor_cfg: SceneEntityCfg | None = None,
    right_contact_sensor_cfg: SceneEntityCfg | None = None,
    age_scale: float = 0.25,
) -> torch.Tensor:
    age = _hall_foot_packet(
        env, hall_cfg, asset_cfg, contact_sensor_cfg, left_contact_sensor_cfg, right_contact_sensor_cfg
    )["age"]
    return torch.clamp(age / max(age_scale, 1.0e-6), 0.0, 1.0)


def hall_sample_period_lr(
    env: ManagerBasedRLEnv,
    hall_cfg: HallFootSensorCfg,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    left_contact_sensor_cfg: SceneEntityCfg | None = None,
    right_contact_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    return _hall_foot_packet(
        env, hall_cfg, asset_cfg, contact_sensor_cfg, left_contact_sensor_cfg, right_contact_sensor_cfg
    )["period"]


def reset_hall_foot_sensor(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Reset Hall baseline/filter/drift state for selected RL environments.

    The Hall object is created lazily by the observation manager, so the first
    environment reset may legitimately arrive before it exists.  Subsequent
    partial resets are forwarded exactly and invalidate the per-step packet.
    """
    sensor: HallFootSensor | None = getattr(env, "_hall_foot_sensor", None)
    if sensor is None:
        return
    sensor.reset(env_ids)
    env._hall_foot_sensor_step = -1
    env._hall_foot_packet_cache = {}
    if hasattr(env, "_hall_foot_prev_episode_length"):
        if env_ids is None:
            env._hall_foot_prev_episode_length.copy_(env.episode_length_buf)
        else:
            ids = env_ids.to(device=env.device, dtype=torch.long)
            env._hall_foot_prev_episode_length[ids] = env.episode_length_buf[ids]


def sample_magnetic_array_proxy(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
    sensor_gain_range: tuple[float, float] = (0.72, 1.28),
    axis_gain_range: tuple[float, float] = (0.75, 1.25),
    zero_residual_std: float = 0.06,
    dead_channel_prob: float = 0.015,
    foot_dropout_prob: float = 0.02,
    period_range_s: tuple[float, float] = (0.018, 0.048),
) -> None:
    """Sample episode-correlated parameters for the 15-point Hall-array proxy."""

    n, device = env.num_envs, env.device
    needs_init = (
        not hasattr(env, "magnetic_sensor_gain_buf")
        or env.magnetic_sensor_gain_buf.shape[0] != n
    )
    if needs_init:
        env.magnetic_sensor_gain_buf = torch.ones((n, 2, 15, 1), device=device)
        env.magnetic_axis_gain_buf = torch.ones((n, 2, 1, 3), device=device)
        env.magnetic_normal_mix_buf = torch.zeros((n, 2, 1, 3), device=device)
        env.magnetic_tangent_mix_buf = torch.zeros((n, 2, 1, 3), device=device)
        env.magnetic_zero_residual_buf = torch.zeros((n, 2, 15, 3), device=device)
        env.magnetic_channel_keep_buf = torch.ones((n, 2, 15, 1), device=device)
        env.magnetic_period_buf = torch.full((n, 2), 0.02, device=device)
        env.magnetic_episode_valid_lr_buf = torch.ones((n, 2), device=device)
        env.magnetic_episode_age_lr_buf = torch.zeros((n, 2), device=device)
        env.magnetic_valid_lr_buf = torch.ones((n, 2), device=device)
        env.magnetic_age_lr_buf = torch.zeros((n, 2), device=device)
        env.magnetic_last_step_buf = torch.full((n,), -1, device=device, dtype=torch.long)
        env.magnetic_packet_cache = {}
    if env_ids is None:
        env_ids = torch.arange(n, device=device)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)
    if env_ids.numel() == 0:
        return
    count = env_ids.numel()

    def uniform(bounds: tuple[float, float], shape: tuple[int, ...]) -> torch.Tensor:
        lo, hi = float(bounds[0]), float(bounds[1])
        return lo + (hi - lo) * torch.rand(shape, device=device)

    env.magnetic_sensor_gain_buf[env_ids] = uniform(
        sensor_gain_range, (count, 2, 15, 1)
    )
    env.magnetic_axis_gain_buf[env_ids] = uniform(
        axis_gain_range, (count, 2, 1, 3)
    )
    normal_nominal = torch.tensor([0.14, -0.10, 1.00], device=device)
    tangent_nominal = torch.tensor([1.00, 0.42, 0.12], device=device)
    tangent_sign = torch.where(
        torch.rand((count, 2, 1, 1), device=device) < 0.5,
        -1.0,
        1.0,
    )
    env.magnetic_normal_mix_buf[env_ids] = (
        normal_nominal.view(1, 1, 1, 3)
        + 0.08 * torch.randn((count, 2, 1, 3), device=device)
    )
    env.magnetic_tangent_mix_buf[env_ids] = (
        tangent_sign * tangent_nominal.view(1, 1, 1, 3)
        + 0.12 * torch.randn((count, 2, 1, 3), device=device)
    )
    env.magnetic_zero_residual_buf[env_ids] = zero_residual_std * torch.randn(
        (count, 2, 15, 3), device=device
    )
    env.magnetic_channel_keep_buf[env_ids] = (
        torch.rand((count, 2, 15, 1), device=device) >= dead_channel_prob
    ).float()
    env.magnetic_period_buf[env_ids] = uniform(period_range_s, (count, 2))
    valid = (
        torch.rand((count, 2), device=device) >= foot_dropout_prob
    ).float()
    episode_age = torch.where(
        valid > 0.5,
        0.08 * torch.rand((count, 2), device=device),
        0.25 + 0.35 * torch.rand((count, 2), device=device),
    )
    env.magnetic_episode_valid_lr_buf[env_ids] = valid
    env.magnetic_episode_age_lr_buf[env_ids] = episode_age
    env.magnetic_valid_lr_buf[env_ids] = valid
    env.magnetic_age_lr_buf[env_ids] = episode_age
    env.magnetic_last_step_buf[env_ids] = -1
    env.magnetic_packet_cache = {}


def _magnetic_packet(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> dict[str, torch.Tensor]:
    """Generate one coherent normalized dual-foot magnetic packet per step."""

    if not hasattr(env, "magnetic_sensor_gain_buf"):
        sample_magnetic_array_proxy(env)
    step = env.episode_length_buf.to(dtype=torch.long)
    changed = env.magnetic_last_step_buf != step
    ids = torch.nonzero(changed, as_tuple=False).squeeze(-1)
    if ids.numel() > 0:
        forces = _foot_forces_w(env, sensor_cfg)
        fn = torch.abs(forces[:, :, 2]) * DEFAULT_FORCE_SCALE
        ft = torch.linalg.norm(forces[:, :, :2], dim=-1) * DEFAULT_FORCE_SCALE
        profile = torch.tensor(
            [
                0.70, 0.76, 0.70,
                0.76, 0.82, 0.76,
                0.82, 0.88, 0.82,
                0.88, 0.94, 0.88,
                0.94, 1.00, 0.94,
            ],
            device=env.device,
        )
        profile = (profile / profile.mean()).view(1, 1, 15, 1)
        signal = profile * env.magnetic_sensor_gain_buf * env.magnetic_axis_gain_buf * (
            fn[:, :, None, None] * env.magnetic_normal_mix_buf
            + ft[:, :, None, None] * env.magnetic_tangent_mix_buf
        )
        magnetic = 5.0 * torch.tanh(signal / 5.0)
        magnetic = magnetic + env.magnetic_zero_residual_buf
        noise = (0.025 + 0.018 * torch.abs(magnetic)) * torch.randn_like(magnetic)
        magnetic = torch.clamp(
            (magnetic + noise) * env.magnetic_channel_keep_buf,
            -6.0,
            6.0,
        )
        # Structured sensor validity is dual-foot; retain independent fields in
        # the deploy schema so real left/right failures are represented exactly.
        if hasattr(env, "structured_foot_current_valid_buf"):
            common_valid = env.structured_foot_current_valid_buf[:, None]
            env.magnetic_valid_lr_buf = (
                env.magnetic_episode_valid_lr_buf * common_valid.expand(-1, 2)
            )
            common_age = env.structured_foot_current_age_buf[:, None]
            env.magnetic_age_lr_buf = torch.maximum(
                env.magnetic_episode_age_lr_buf, common_age.expand(-1, 2)
            )
        else:
            env.magnetic_valid_lr_buf = env.magnetic_episode_valid_lr_buf.clone()
            env.magnetic_age_lr_buf = env.magnetic_episode_age_lr_buf.clone()
        valid = env.magnetic_valid_lr_buf[:, :, None, None]
        magnetic = magnetic * valid
        env.magnetic_last_step_buf[ids] = step[ids]
        env.magnetic_packet_cache = {
            "magnetic": magnetic.reshape(env.num_envs, 90),
            "period": env.magnetic_period_buf,
            "valid": env.magnetic_valid_lr_buf,
            "age": env.magnetic_age_lr_buf,
        }
    return env.magnetic_packet_cache


def magnetic_array_proxy(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Normalized current frame, flattened left 15xXYZ then right 15xXYZ."""

    return _magnetic_packet(env, sensor_cfg)["magnetic"]


def magnetic_sample_period_lr(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    return _magnetic_packet(env, sensor_cfg)["period"]


def magnetic_sensor_valid_lr(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    return _magnetic_packet(env, sensor_cfg)["valid"]


def magnetic_sensor_age_lr(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    age_scale: float = 0.25,
) -> torch.Tensor:
    return torch.clamp(
        _magnetic_packet(env, sensor_cfg)["age"] / max(age_scale, 1.0e-6),
        0.0,
        1.0,
    )


def inject_foot_sensor_dropout(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
    dropout_prob: float = 0.05,
    stale_age: float = 0.3,
) -> None:
    """Domain-randomize sensor validity (call from event term).

    With probability ``dropout_prob``, mark sensor invalid and set age high.
    """
    n = env.num_envs
    device = env.device
    if not hasattr(env, "foot_sensor_valid_buf"):
        env.foot_sensor_valid_buf = torch.ones(n, device=device)
        env.foot_sensor_age_buf = torch.zeros(n, device=device)
    if env_ids is None:
        env_ids = torch.arange(n, device=device)
    if env_ids.numel() == 0:
        return
    drop = torch.rand(env_ids.shape[0], device=device) < dropout_prob
    env.foot_sensor_valid_buf[env_ids] = torch.where(
        drop, torch.zeros_like(env.foot_sensor_valid_buf[env_ids]), torch.ones_like(env.foot_sensor_valid_buf[env_ids])
    )
    env.foot_sensor_age_buf[env_ids] = torch.where(
        drop,
        torch.full_like(env.foot_sensor_age_buf[env_ids], stale_age),
        torch.zeros_like(env.foot_sensor_age_buf[env_ids]),
    )
