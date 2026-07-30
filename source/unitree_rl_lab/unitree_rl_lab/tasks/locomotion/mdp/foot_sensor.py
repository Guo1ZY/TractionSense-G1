# Copyright (c) 2022-2025, The Isaac Lab Project Developers / local foot-sensor extension.
# SPDX-License-Identifier: BSD-3-Clause
"""Foot-sensor observations aligned with zorn ``foot_sensor`` schema.

Simulation training uses Isaac Lab :class:`ContactSensor` on ankle-roll bodies
(vectorized, multi-env). The field semantics mirror the zorn ContactView /
ROS2 pipeline so that Real2Sim / Sim2Real can share the same observation
layout.

Zorn reference (host path configured through ``ZORN_FOOT_SENSOR_ROOT``):
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

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

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
