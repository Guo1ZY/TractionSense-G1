# Copyright (c) 2026 local Hall-foot extension.
# SPDX-License-Identifier: BSD-3-Clause
"""Privileged bookkeeping for a physical high--low--high friction course.

The functions in this module never create actor observations.  The friction
buffers are exact simulator labels used by rewards, the asymmetric critic and
evaluation only.  The deployable actor continues to receive Hall Bx/By/Bz,
packet health/timing and proprioception.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING
import torch

from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
from pxr import PhysxSchema, UsdPhysics

from .spatial_friction_state import (
    SPATIAL_HIGH_END,
    SPATIAL_HIGH_START,
    SPATIAL_LOW,
    advance_spatial_course_stage,
    spatial_course_success_mask,
    stratified_high_patch_reset_x,
    update_transition_retention_latch,
    update_low_capture_stability,
    update_low_capture_timing,
)
from unitree_rl_lab.traction.high_end_state_bank import (
    TRAINING_ROLE,
    load_high_end_state_bank,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _env_ids(
    env: ManagerBasedEnv, env_ids: torch.Tensor | slice | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return matching CPU and simulator-device environment indices."""
    if env_ids is None or isinstance(env_ids, slice):
        ids_cpu = torch.arange(env.scene.num_envs, device="cpu", dtype=torch.long)
    else:
        ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
    return ids_cpu, ids_cpu.to(device=env.device)


def reset_root_state_spatial_stratified(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    x_bands: tuple[tuple[float, float], ...],
    band_probabilities: tuple[float, ...],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    low_boundary_x: float = 0.0,
    minimum_high_margin: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset across several safe HighStart distances to mix rollout phases.

    Unlike random episode length, this always creates a physically valid
    HighStart state.  The LOW stage and its rewards remain disabled until a
    foot produces filtered contact with the real LOW collider.  ``x`` is
    intentionally excluded from ``pose_range`` so no second position sampler
    can move a reset into LOW.
    """

    if "x" in pose_range:
        raise ValueError("spatial stratified reset owns x; remove x from pose_range")
    _, ids = _env_ids(env, env_ids)
    if ids.numel() == 0:
        return

    asset = env.scene[asset_cfg.name]
    count = int(ids.numel())
    root_states = asset.data.default_root_state[ids].clone()
    local_x, band_index = stratified_high_patch_reset_x(
        torch.rand(count, device=asset.device),
        torch.rand(count, device=asset.device),
        x_bands=x_bands,
        band_probabilities=band_probabilities,
        low_boundary_x=float(low_boundary_x),
        minimum_high_margin=float(minimum_high_margin),
    )

    pose_keys = ("y", "z", "roll", "pitch", "yaw")
    pose_bounds = torch.tensor(
        [pose_range.get(key, (0.0, 0.0)) for key in pose_keys],
        device=asset.device,
        dtype=root_states.dtype,
    )
    pose_samples = math_utils.sample_uniform(
        pose_bounds[:, 0], pose_bounds[:, 1], (count, len(pose_keys)), device=asset.device
    )
    positions = root_states[:, 0:3] + env.scene.env_origins[ids]
    positions[:, 0] += local_x
    positions[:, 1] += pose_samples[:, 0]
    positions[:, 2] += pose_samples[:, 1]
    orientation_delta = math_utils.quat_from_euler_xyz(
        pose_samples[:, 2], pose_samples[:, 3], pose_samples[:, 4]
    )
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientation_delta)

    velocity_keys = ("x", "y", "z", "roll", "pitch", "yaw")
    velocity_bounds = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in velocity_keys],
        device=asset.device,
        dtype=root_states.dtype,
    )
    velocity_samples = math_utils.sample_uniform(
        velocity_bounds[:, 0],
        velocity_bounds[:, 1],
        (count, len(velocity_keys)),
        device=asset.device,
    )
    velocities = root_states[:, 7:13] + velocity_samples

    asset.write_root_pose_to_sim(
        torch.cat((positions, orientations), dim=-1), env_ids=ids
    )
    asset.write_root_velocity_to_sim(velocities, env_ids=ids)

    # Privileged diagnostics only. These names are intentionally absent from
    # FootTractionMagneticMotionObservationsCfg and the exported actor schema.
    if not hasattr(env, "spatial_reset_band_buf"):
        env.spatial_reset_band_buf = torch.full(
            (env.scene.num_envs,), -1, device=env.device, dtype=torch.long
        )
        env.spatial_reset_local_x_buf = torch.zeros(
            env.scene.num_envs, device=env.device, dtype=torch.float32
        )
    env.spatial_reset_band_buf[ids] = band_index.to(env.device)
    env.spatial_reset_local_x_buf[ids] = local_x.to(env.device)


def reset_root_state_high_end_perturbed(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    x_range: tuple[float, float],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    state_bank_path: str | None = None,
    state_bank_required_role: str = TRAINING_ROLE,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset directly inside the final high-friction patch at high speed."""

    if "x" in pose_range:
        raise ValueError("high-end reset owns x; remove x from pose_range")
    _, ids = _env_ids(env, env_ids)
    if ids.numel() == 0:
        return

    asset = env.scene[asset_cfg.name]
    count = int(ids.numel())
    if state_bank_path is not None:
        bank = getattr(env, "_high_end_recovery_state_bank", None)
        resolved_path = str(Path(state_bank_path).expanduser().resolve())
        if bank is None or bank.path != resolved_path:
            bank = load_high_end_state_bank(
                state_bank_path,
                device=asset.device,
                allowed_roles=(state_bank_required_role,),
            )
            env._high_end_recovery_state_bank = bank
        sample_count = bank.sample_count
        sample_ids = torch.randint(sample_count, (count,), device=asset.device)
        root_pose = bank.arrays["root_pose_local"][sample_ids].clone()
        root_pose[:, :3] += env.scene.env_origins[ids]
        root_velocity = bank.arrays["root_velocity"][sample_ids]
        asset.write_root_pose_to_sim(root_pose, env_ids=ids)
        asset.write_root_velocity_to_sim(root_velocity, env_ids=ids)
        asset.write_joint_state_to_sim(
            bank.arrays["joint_pos"][sample_ids],
            bank.arrays["joint_vel"][sample_ids],
            env_ids=ids,
        )
        if not hasattr(env, "_high_end_recovery_pending_sample_ids"):
            env._high_end_recovery_pending_sample_ids = torch.full(
                (env.scene.num_envs,), -1, device=env.device, dtype=torch.long
            )
        env._high_end_recovery_pending_sample_ids[ids] = sample_ids.to(env.device)
        return
    root_states = asset.data.default_root_state[ids].clone()
    local_x = torch.empty(count, device=asset.device, dtype=root_states.dtype)
    local_x.uniform_(float(x_range[0]), float(x_range[1]))

    pose_keys = ("y", "z", "roll", "pitch", "yaw")
    pose_bounds = torch.tensor(
        [pose_range.get(key, (0.0, 0.0)) for key in pose_keys],
        device=asset.device,
        dtype=root_states.dtype,
    )
    pose_samples = math_utils.sample_uniform(
        pose_bounds[:, 0], pose_bounds[:, 1], (count, len(pose_keys)), device=asset.device
    )
    positions = root_states[:, 0:3] + env.scene.env_origins[ids]
    positions[:, 0] += local_x
    positions[:, 1] += pose_samples[:, 0]
    positions[:, 2] += pose_samples[:, 1]
    orientation_delta = math_utils.quat_from_euler_xyz(
        pose_samples[:, 2], pose_samples[:, 3], pose_samples[:, 4]
    )
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientation_delta)

    velocity_keys = ("x", "y", "z", "roll", "pitch", "yaw")
    velocity_bounds = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in velocity_keys],
        device=asset.device,
        dtype=root_states.dtype,
    )
    velocity_samples = math_utils.sample_uniform(
        velocity_bounds[:, 0], velocity_bounds[:, 1], (count, len(velocity_keys)), device=asset.device
    )
    velocities = root_states[:, 7:13] + velocity_samples
    asset.write_root_pose_to_sim(
        torch.cat((positions, orientations), dim=-1), env_ids=ids
    )
    asset.write_root_velocity_to_sim(velocities, env_ids=ids)


def randomize_spatial_low_patch_mu(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    extreme_mu_range: tuple[float, float] = (0.10, 0.16),
    mild_mu_range: tuple[float, float] = (0.16, 0.28),
    extreme_fraction: float = 0.35,
    anchor_mu: float = 0.28,
    anchor_fraction: float = 0.0,
) -> None:
    """Assign one fixed LOW-patch mu per environment at startup.

    This is the R4 low-mu curriculum: 35% of environments get the extreme
    range and the rest get the mild range, so training covers the full
    0.10-0.28 fault distribution without every episode being dominated by the
    hardest surface.  An exact ``anchor_mu`` share keeps the nominal training
    point represented so broad curriculum coverage cannot erode it.  The
    authored USD collider material is rewritten per environment and the
    privileged mu buffer is published for the friction state machine.
    """

    for name, value in (
        ("extreme_mu_range", extreme_mu_range),
        ("mild_mu_range", mild_mu_range),
    ):
        if value[0] <= 0.0 or value[1] < value[0]:
            raise ValueError(f"{name} must be an increasing positive range")
    if not 0.0 <= extreme_fraction <= 1.0:
        raise ValueError("extreme_fraction must be in [0, 1]")
    if not 0.0 <= anchor_fraction <= 1.0 or extreme_fraction + anchor_fraction > 1.0:
        raise ValueError("anchor/extreme fractions must sum to at most 1")
    _, ids = _env_ids(env, env_ids)
    if ids.numel() == 0:
        return
    ids_cpu, _ = _env_ids(env, env_ids)
    stage = env.scene.stage
    env_paths = tuple(env.scene.env_prim_paths[index] for index in ids_cpu.tolist())
    count = int(ids.numel())
    device = env.device
    selector = torch.rand(count, device=device)
    anchor = selector < float(anchor_fraction)
    extreme = (
        (selector >= float(anchor_fraction))
        & (selector < float(anchor_fraction) + float(extreme_fraction))
    )
    mu = torch.empty(count, device=device)
    mu[extreme] = torch.rand(
        int(extreme.sum().item()), device=device
    ) * (extreme_mu_range[1] - extreme_mu_range[0]) + extreme_mu_range[0]
    mu[~extreme] = torch.rand(
        int((~extreme).sum().item()), device=device
    ) * (mild_mu_range[1] - mild_mu_range[0]) + mild_mu_range[0]
    mu[anchor] = float(anchor_mu)
    mu = mu.clamp(min=0.05, max=0.85)
    for env_path, value in zip(env_paths, mu.tolist()):
        material_prim = stage.GetPrimAtPath(
            f"{env_path}/FrictionLow/geometry/material"
        )
        if not material_prim.IsValid():
            raise RuntimeError(
                f"missing FrictionLow material prim at {env_path}"
            )
        material_api = UsdPhysics.MaterialAPI(material_prim)
        material_api.CreateStaticFrictionAttr().Set(float(value))
        material_api.CreateDynamicFrictionAttr().Set(float(value))
        physx_api = PhysxSchema.PhysxMaterialAPI(material_prim)
        if not physx_api.GetFrictionCombineModeAttr():
            physx_api.CreateFrictionCombineModeAttr("multiply")
    if not hasattr(env, "spatial_low_patch_mu_buf"):
        env.spatial_low_patch_mu_buf = torch.full(
            (env.scene.num_envs,),
            0.28,
            device=device,
            dtype=torch.float32,
        )
    env.spatial_low_patch_mu_buf[ids] = mu


def update_uniform_high_friction_buffer(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    ground_patch_mu: float = 0.90,
) -> None:
    """Publish the exact effective mu for an all-high physical course.

    The three static floor meshes all carry ``ground_patch_mu``.  Robot shape
    material randomization is represented by ``friction_material_scale_buf``
    and PhysX combines the two using ``multiply``.  This event mirrors that
    product into critic/reward diagnostics without exposing it to the 1864-D
    actor.  It deliberately has no LOW/HIGH state machine.
    """

    if not math.isfinite(float(ground_patch_mu)) or ground_patch_mu <= 0.0:
        raise ValueError("ground_patch_mu must be finite and positive")
    _, ids = _env_ids(env, env_ids)
    if ids.numel() == 0:
        return
    scale = getattr(env, "friction_material_scale_buf", None)
    if scale is None:
        raise RuntimeError(
            "uniform high-friction buffer requires physics material scale first"
        )
    if scale.ndim != 1 or scale.shape[0] != env.scene.num_envs:
        raise RuntimeError("friction_material_scale_buf must have shape [num_envs]")
    selected = scale.index_select(0, ids)
    if not torch.isfinite(selected).all() or bool((selected <= 0.0).any().item()):
        raise RuntimeError("friction material scale must be finite and positive")
    effective = selected * float(ground_patch_mu)

    def _ensure(name: str, value: torch.Tensor) -> torch.Tensor:
        target = getattr(env, name, None)
        if target is None:
            target = value.clone()
            setattr(env, name, target)
        elif target.ndim != 1 or target.shape[0] != env.scene.num_envs:
            raise RuntimeError(f"{name} must have shape [num_envs]")
        return target

    n = env.scene.num_envs
    device = env.device
    ground = _ensure(
        "ground_friction_mu_buf",
        torch.full((n,), float(ground_patch_mu), device=device),
    )
    effective_buf = _ensure(
        "effective_friction_mu_buf",
        torch.full((n,), float(ground_patch_mu), device=device),
    )
    regime = _ensure(
        "ground_friction_regime_buf",
        torch.full((n,), 2, device=device, dtype=torch.long),
    )
    ground[ids] = effective.to(device=ground.device, dtype=ground.dtype)
    effective_buf[ids] = effective.to(
        device=effective_buf.device, dtype=effective_buf.dtype
    )
    regime[ids] = 2


def sync_uniform_friction_course_stage(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    high_regime_indices: tuple[int, ...] = (2,),
) -> None:
    """Map the current uniform per-environment friction regime to the private stage.

    The slope/stairs task has no spatial high--low--high patch, so there is no
    contact-latched course state machine.  AnchoredPPO still requires the
    privileged ``spatial_course_stage_buf [num_envs]`` tensor; this event
    synthesizes it from the friction regime published by
    ``randomize_teacher_friction_with_buffer``.

    Regimes listed in ``high_regime_indices`` (by default the top stratum,
    e.g. mu in (0.75, 1.0] which contains the 0.8 reference surface) become
    ``SPATIAL_HIGH_START`` and therefore keep the frozen-Teacher anchor.  All
    lower-friction regimes become ``SPATIAL_LOW`` so the bounded stability
    residual and the frozen Hall capture adaptation stay free to deviate from
    the nominal fast gait on slippery terrain.  The stage label never enters
    the 1864-D actor observation.
    """
    _, ids = _env_ids(env, env_ids)
    if ids.numel() == 0:
        return
    n = env.scene.num_envs
    regime = getattr(env, "ground_friction_regime_buf", None)
    if regime is None or regime.ndim != 1 or regime.shape[0] != n:
        raise RuntimeError(
            "uniform stage sync requires ground_friction_regime_buf [num_envs] "
            "published by the friction randomization event first"
        )
    stage = getattr(env, "spatial_course_stage_buf", None)
    if stage is None:
        stage = torch.full(
            (n,), SPATIAL_HIGH_START, device=env.device, dtype=torch.long
        )
        setattr(env, "spatial_course_stage_buf", stage)
    elif (
        stage.ndim != 1
        or stage.shape[0] != n
        or stage.device != torch.device(env.device)
        or stage.is_floating_point()
        or stage.dtype == torch.bool
    ):
        raise RuntimeError(
            "spatial_course_stage_buf must be a [num_envs] long tensor on the env device"
        )
    high = torch.zeros(n, device=env.device, dtype=torch.bool)
    for index in high_regime_indices:
        high |= regime == int(index)
    stage[ids] = torch.where(high[ids], SPATIAL_HIGH_START, SPATIAL_LOW)


def push_spatial_high_grip_recovery_by_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Inject a recoverable asynchronous impulse only on a latched HIGH patch.

    This is a training-only state-distribution curriculum.  The physical
    patch/contact state selects which scheduled environments receive the
    perturbation, but it is never appended to actor observations.  LOW is
    deliberately excluded: its Hall-driven cadence/stride adaptation should
    not be confounded with an externally injected loss of balance.

    The velocity ranges are *increments* in world coordinates, matching the
    current Isaac Lab ``push_by_setting_velocity`` semantics.  Missing axes
    receive a zero increment.  A missing stage buffer is a configuration
    error rather than a silent all-stage fallback.
    """

    _, ids = _env_ids(env, env_ids)
    if ids.numel() == 0:
        return
    stage = getattr(env, "spatial_course_stage_buf", None)
    if stage is None:
        raise RuntimeError(
            "spatial_course_stage_buf must be initialized before recovery pushes"
        )
    if stage.ndim != 1 or stage.shape[0] != env.scene.num_envs:
        raise RuntimeError(
            "spatial_course_stage_buf must have shape [num_envs]"
        )
    if stage.dtype == torch.bool or stage.is_floating_point():
        raise TypeError("spatial_course_stage_buf must use an integer dtype")

    selected = ids[
        (stage[ids] == SPATIAL_HIGH_START) | (stage[ids] == SPATIAL_HIGH_END)
    ]
    if selected.numel() == 0:
        return

    asset = env.scene[asset_cfg.name]
    ranges = torch.as_tensor(
        [
            velocity_range.get(axis, (0.0, 0.0))
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        ],
        device=asset.device,
        dtype=asset.data.root_vel_w.dtype,
    )
    if ranges.shape != (6, 2) or not torch.isfinite(ranges).all():
        raise ValueError("velocity_range must contain finite (min, max) pairs")
    if bool((ranges[:, 0] > ranges[:, 1]).any().item()):
        raise ValueError("velocity_range minimum must not exceed maximum")

    velocity = asset.data.root_vel_w[selected].clone()
    velocity += math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], velocity.shape, device=asset.device
    )
    asset.write_root_velocity_to_sim(velocity, env_ids=selected)

    count = getattr(env, "spatial_high_grip_recovery_push_count_buf", None)
    if count is None:
        count = torch.zeros(
            env.scene.num_envs, device=env.device, dtype=torch.long
        )
        env.spatial_high_grip_recovery_push_count_buf = count
    count[selected] += 1


def randomize_coherent_material_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    scale_range: tuple[float, float],
    restitution_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    teacher_friction_ranges: tuple[tuple[float, float], ...] | None = None,
    regime_probabilities: tuple[float, ...] | None = None,
) -> None:
    """Assign one coherent rigid-material multiplier to each robot.

    The three course patches own the spatial friction profile.  Their PhysX
    materials use ``multiply`` combine mode, so a per-environment robot scale
    gives inexpensive domain randomization without changing or teleporting the
    ground.  Static and dynamic robot friction are deliberately identical.
    """
    # Several established G1 task ancestors annotate material events with
    # teacher-stratum metadata during post-init.  A spatial course does not
    # sample those strata; accepting and discarding the two explicit fields
    # avoids coupling commands to hidden material while satisfying the event
    # manager's signature validation.
    del teacher_friction_ranges, regime_probabilities
    lo, hi = map(float, scale_range)
    rest_lo, rest_hi = map(float, restitution_range)
    if lo <= 0.0 or hi < lo:
        raise ValueError(f"invalid scale_range={scale_range}")
    if rest_lo < 0.0 or rest_hi < rest_lo:
        raise ValueError(f"invalid restitution_range={restitution_range}")

    ids_cpu, ids_device = _env_ids(env, env_ids)
    if ids_cpu.numel() == 0:
        return

    asset = env.scene[asset_cfg.name]
    count = int(ids_cpu.numel())
    scale_cpu = torch.empty(count, device="cpu").uniform_(lo, hi)
    restitution_cpu = torch.empty(count, device="cpu").uniform_(rest_lo, rest_hi)
    samples = torch.stack((scale_cpu, scale_cpu, restitution_cpu), dim=-1)

    properties = asset.root_physx_view.get_material_properties()
    samples = samples[:, None, :].expand(-1, asset.root_physx_view.max_shapes, -1)
    properties[ids_cpu] = samples
    asset.root_physx_view.set_material_properties(properties, ids_cpu)

    if not hasattr(env, "friction_material_scale_buf"):
        env.friction_material_scale_buf = torch.ones(
            env.scene.num_envs, device=env.device, dtype=torch.float32
        )
    env.friction_material_scale_buf[ids_device] = scale_cpu.to(env.device)


def update_spatial_friction_buffer(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    low_patch_mu: float = 0.16,
    high_patch_mu: float = 0.90,
    contact_force_threshold: float = 5.0,
    control_dt: float = 0.02,
    capture_target_speed: float = 0.24,
    capture_speed_tolerance: float = 0.05,
    capture_stable_time_s: float = 0.12,
    capture_deadline_s: float = 0.90,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_hall_contact"),
    right_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_hall_contact"),
) -> None:
    """Track the physical patch contacted by each foot and latch H--L--H state.

    Filter order is ``HighStart, Low, HighEnd`` for each dedicated one-foot
    ContactSensor.  LOW becomes active only after measured low-patch contact,
    remains active through flight, and clears only after measured final-high
    contact.  The label is privileged and is never appended to the Hall actor
    observation.
    """
    if low_patch_mu <= 0.0 or high_patch_mu <= low_patch_mu:
        raise ValueError(
            "expected 0 < low_patch_mu < high_patch_mu, got "
            f"{low_patch_mu}, {high_patch_mu}"
        )
    if control_dt <= 0.0 or capture_stable_time_s <= 0.0:
        raise ValueError("control_dt and capture_stable_time_s must be positive")
    if capture_deadline_s <= 0.0:
        raise ValueError("capture_deadline_s must be positive")

    _, ids = _env_ids(env, env_ids)
    if ids.numel() == 0:
        return
    n = env.scene.num_envs
    device = env.device

    def _ensure(name: str, value: torch.Tensor) -> torch.Tensor:
        if not hasattr(env, name):
            setattr(env, name, value)
        return getattr(env, name)

    scale = _ensure(
        "friction_material_scale_buf",
        torch.ones(n, device=device, dtype=torch.float32),
    )
    ground_mu = _ensure(
        "ground_friction_mu_buf",
        torch.full((n,), high_patch_mu, device=device),
    )
    effective_mu = _ensure(
        "effective_friction_mu_buf",
        torch.full((n,), high_patch_mu, device=device),
    )
    regime = _ensure(
        "ground_friction_regime_buf",
        torch.full((n,), 2, device=device, dtype=torch.long),
    )
    low_contact_buf = _ensure(
        "spatial_low_contact_buf",
        torch.zeros(n, device=device, dtype=torch.bool),
    )
    course_stage = _ensure(
        "spatial_course_stage_buf",
        torch.full(
            (n,), SPATIAL_HIGH_START, device=device, dtype=torch.long
        ),
    )
    high_end_contact_buf = _ensure(
        "spatial_high_end_contact_buf",
        torch.zeros(n, device=device, dtype=torch.bool),
    )
    patch_force_buf = _ensure(
        "spatial_patch_contact_force_buf",
        torch.zeros((n, 2, 3), device=device, dtype=torch.float32),
    )
    foot_mu_buf = _ensure(
        "spatial_foot_friction_mu_buf",
        torch.full((n, 2), high_patch_mu, device=device),
    )
    switch_count = _ensure(
        "friction_switch_count_buf",
        torch.zeros(n, device=device, dtype=torch.long),
    )
    switch_step = _ensure(
        "friction_switch_step_buf",
        torch.full((n,), -1, device=device, dtype=torch.long),
    )
    switch_direction = _ensure(
        "friction_switch_direction_buf",
        torch.zeros(n, device=device, dtype=torch.int8),
    )
    previous_mu = _ensure(
        "friction_switch_previous_mu_buf",
        torch.full((n,), high_patch_mu, device=device),
    )
    target_mu = _ensure(
        "friction_switch_target_mu_buf",
        torch.full((n,), high_patch_mu, device=device),
    )
    low_entry_step = _ensure(
        "spatial_low_entry_step_buf",
        torch.full((n,), -1, device=device, dtype=torch.long),
    )
    low_entry_speed = _ensure(
        "spatial_low_entry_speed_buf",
        torch.zeros(n, device=device, dtype=torch.float32),
    )
    low_elapsed = _ensure(
        "spatial_low_elapsed_s_buf",
        torch.zeros(n, device=device, dtype=torch.float32),
    )
    stable_count = _ensure(
        "spatial_low_stable_count_buf",
        torch.zeros(n, device=device, dtype=torch.long),
    )
    capture_success = _ensure(
        "spatial_low_capture_success_buf",
        torch.zeros(n, device=device, dtype=torch.bool),
    )
    capture_new_success = _ensure(
        "spatial_low_capture_new_success_buf",
        torch.zeros(n, device=device, dtype=torch.bool),
    )
    capture_elapsed = _ensure(
        "spatial_low_capture_elapsed_s_buf",
        torch.full((n,), -1.0, device=device, dtype=torch.float32),
    )
    capture_timely = _ensure(
        "spatial_low_capture_timely_buf",
        torch.zeros(n, device=device, dtype=torch.bool),
    )
    transition_heading_error_buf = _ensure(
        "transition_heading_error_buf",
        torch.zeros(n, device=device, dtype=torch.float32),
    )
    transition_heading_reference = _ensure(
        "transition_heading_reference_xy",
        torch.zeros((n, 2), device=device, dtype=torch.float32),
    )
    transition_heading_initialized = _ensure(
        "transition_heading_initialized",
        torch.zeros(n, device=device, dtype=torch.bool),
    )
    low_entry_heading = _ensure(
        "spatial_low_entry_heading_buf",
        torch.full((n,), float("nan"), device=device, dtype=torch.float32),
    )
    high_end_elapsed = _ensure(
        "spatial_high_end_elapsed_s_buf",
        torch.zeros(n, device=device, dtype=torch.float32),
    )

    def _filtered_force(sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        sensor = env.scene.sensors[sensor_cfg.name]
        matrix = sensor.data.force_matrix_w
        if matrix is None:
            raise RuntimeError(
                f"filtered ContactSensor {sensor_cfg.name!r} has no force_matrix_w"
            )
        selected = matrix[ids]
        if selected.ndim != 4 or tuple(selected.shape[1:]) != (1, 3, 3):
            raise RuntimeError(
                f"filtered ContactSensor {sensor_cfg.name!r} must have shape "
                f"[num_envs, 1 foot, 3 patches, xyz], got {tuple(selected.shape)}"
            )
        # [selected env, one foot, three filters, xyz] -> [selected env, three]
        return torch.linalg.vector_norm(torch.nan_to_num(selected[:, 0]), dim=-1)

    force_by_foot_patch = torch.stack(
        (
            _filtered_force(left_contact_sensor_cfg),
            _filtered_force(right_contact_sensor_cfg),
        ),
        dim=1,
    )
    patch_contact = force_by_foot_patch >= float(contact_force_threshold)
    foot_low_contact = patch_contact[:, :, 1]
    foot_high_end_contact = patch_contact[:, :, 2]
    any_low_contact = foot_low_contact.any(dim=1)
    any_high_end_contact = foot_high_end_contact.any(dim=1)

    reset = env.episode_length_buf[ids] <= 1
    # A HighEnd bank reset deliberately restores episode_length_buf to two so
    # reference/history terms do not execute their ordinary reset branches.
    # Until the specialised environment has finished restoring the exact
    # actor/Hall context, however, ContactSensor buffers may still describe
    # the preceding rollout.  Treat those rows as reset here so a stale LOW
    # contact cannot relatch the privileged course state or friction label.
    # Ordinary spatial tasks never allocate this pending-sample buffer.
    pending_bank_rows = getattr(
        env, "_high_end_recovery_pending_sample_ids", None
    )
    if pending_bank_rows is not None:
        if (
            pending_bank_rows.ndim != 1
            or pending_bank_rows.shape[0] != env.scene.num_envs
            or pending_bank_rows.dtype == torch.bool
            or pending_bank_rows.is_floating_point()
        ):
            raise RuntimeError(
                "_high_end_recovery_pending_sample_ids must be an integer "
                "[num_envs] tensor"
            )
        reset = reset | (pending_bank_rows[ids] >= 0)
    old_stage = course_stage[ids].clone()
    new_stage = advance_spatial_course_stage(
        old_stage,
        any_low_contact,
        any_high_end_contact,
        reset,
    )
    is_low = new_stage == SPATIAL_LOW

    asset = env.scene[asset_cfg.name]
    forward_speed = torch.abs(
        torch.nan_to_num(
            asset.data.root_lin_vel_b[ids, 0],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    )
    quat = asset.data.root_quat_w[ids]
    heading_xy = torch.stack(
        (
            1.0 - 2.0 * (torch.square(quat[:, 2]) + torch.square(quat[:, 3])),
            2.0 * (quat[:, 1] * quat[:, 2] + quat[:, 0] * quat[:, 3]),
        ),
        dim=-1,
    )
    heading_xy = heading_xy / torch.linalg.vector_norm(
        heading_xy, dim=1, keepdim=True
    ).clamp(min=1.0e-6)
    needs_reference = reset | ~transition_heading_initialized[ids]
    if bool(needs_reference.any().item()):
        reference_ids = ids[needs_reference]
        transition_heading_reference[reference_ids] = heading_xy[needs_reference]
        transition_heading_initialized[reference_ids] = True
    reference = transition_heading_reference[ids]
    heading_cross = (
        reference[:, 0] * heading_xy[:, 1]
        - reference[:, 1] * heading_xy[:, 0]
    )
    heading_dot = torch.sum(reference * heading_xy, dim=-1).clamp(-1.0, 1.0)
    heading_error = torch.atan2(heading_cross, heading_dot).clamp(-1.0, 1.0)
    entry_heading_selected, high_end_elapsed_selected = (
        update_transition_retention_latch(
            old_stage,
            new_stage,
            heading_error,
            low_entry_heading[ids],
            high_end_elapsed[ids],
            reset,
            float(control_dt),
        )
    )
    entry_step_selected, entry_speed_selected, elapsed_selected = (
        update_low_capture_timing(
            old_stage,
            new_stage,
            env.episode_length_buf[ids],
            forward_speed,
            low_entry_step[ids],
            low_entry_speed[ids],
            low_elapsed[ids],
            reset,
            float(control_dt),
        )
    )
    required_stable_steps = max(
        1, int(math.ceil(float(capture_stable_time_s) / float(control_dt)))
    )
    (
        stable_count_selected,
        capture_success_selected,
        new_capture_selected,
        timely_selected,
    ) = update_low_capture_stability(
        new_stage,
        forward_speed,
        stable_count[ids],
        capture_success[ids],
        elapsed_selected,
        reset,
        float(capture_target_speed),
        float(capture_speed_tolerance),
        required_stable_steps,
        float(capture_deadline_s),
    )

    scale_selected = scale[ids]
    low_patch_mu_buf = getattr(env, "spatial_low_patch_mu_buf", None)
    if low_patch_mu_buf is not None and low_patch_mu_buf.ndim == 1:
        base_low_mu = low_patch_mu_buf[ids]
    else:
        base_low_mu = torch.full_like(scale_selected, float(low_patch_mu))
    low_mu = base_low_mu * scale_selected
    high_mu = torch.full_like(scale_selected, float(high_patch_mu)) * scale_selected
    selected_mu = torch.where(is_low, low_mu, high_mu)
    # During flight, retain the latched global state.  When a foot has a real
    # filtered contact, report that patch's material; low contact has priority
    # for a sole spanning the Low/HighEnd boundary.
    per_foot = selected_mu[:, None].expand(-1, 2).clone()
    any_high_foot_contact = patch_contact[:, :, 0] | foot_high_end_contact
    per_foot = torch.where(any_high_foot_contact, high_mu[:, None], per_foot)
    per_foot = torch.where(foot_low_contact, low_mu[:, None], per_foot)

    old_low = low_contact_buf[ids].clone()
    changed = (old_low != is_low) & ~reset
    old_mu = ground_mu[ids].clone()

    course_stage[ids] = new_stage
    high_end_contact_buf[ids] = any_high_end_contact & ~reset
    patch_force_buf[ids] = force_by_foot_patch
    foot_mu_buf[ids] = per_foot
    ground_mu[ids] = selected_mu
    effective_mu[ids] = selected_mu
    regime[ids] = torch.where(
        is_low,
        torch.zeros_like(regime[ids]),
        torch.full_like(regime[ids], 2),
    )
    low_contact_buf[ids] = is_low
    previous_mu[ids] = torch.where(changed, old_mu, previous_mu[ids])
    target_mu[ids] = selected_mu
    low_entry_step[ids] = entry_step_selected
    low_entry_speed[ids] = entry_speed_selected
    low_elapsed[ids] = elapsed_selected
    stable_count[ids] = stable_count_selected
    capture_success[ids] = capture_success_selected
    capture_new_success[ids] = new_capture_selected
    transition_heading_error_buf[ids] = heading_error
    low_entry_heading[ids] = entry_heading_selected
    high_end_elapsed[ids] = high_end_elapsed_selected
    capture_elapsed[ids] = torch.where(
        new_capture_selected,
        elapsed_selected,
        capture_elapsed[ids],
    )
    capture_timely[ids] = capture_timely[ids] | timely_selected

    if reset.any():
        reset_ids = ids[reset]
        switch_count[reset_ids] = 0
        switch_step[reset_ids] = -1
        switch_direction[reset_ids] = 0
        capture_elapsed[reset_ids] = -1.0
        capture_timely[reset_ids] = False
    if changed.any():
        changed_ids = ids[changed]
        switch_count[changed_ids] += 1
        switch_step[changed_ids] = env.episode_length_buf[changed_ids]
        switch_direction[changed_ids] = torch.where(
            is_low[changed],
            -torch.ones(changed_ids.numel(), device=device, dtype=torch.int8),
            torch.ones(changed_ids.numel(), device=device, dtype=torch.int8),
        )


def spatial_friction_course_success(
    env: ManagerBasedEnv,
    minimum_local_x: float = 2.60,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Truncate a completed H--L--H traversal before the finite floor ends.

    Success requires the latched state to prove a prior LOW contact, a current
    filtered contact with ``FrictionHighEnd``, and forward progress near the end
    of that patch.  This is a timeout-style curriculum boundary, not a failure,
    so it is excluded from ``mdp.is_terminated`` and its penalty.
    """

    stage = getattr(env, "spatial_course_stage_buf", None)
    high_end_contact = getattr(env, "spatial_high_end_contact_buf", None)
    if stage is None or high_end_contact is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    asset = env.scene[asset_cfg.name]
    root_local_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return spatial_course_success_mask(
        stage,
        high_end_contact,
        root_local_x,
        minimum_local_x,
    )


def push_spatial_transition_heading_recovery(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    vy_range: tuple[float, float] = (-0.12, 0.12),
    yaw_rate_range: tuple[float, float] = (-0.35, 0.35),
    high_end_window_s: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Inject velocity-based heading/lateral disturbances during LOW/HighEnd.

    Training-only transition curriculum: yaw-rate and lateral velocity
    increments that integrate into a sustained heading error the policy must
    actively correct.  Instantaneous world-yaw pose teleports are deliberately
    excluded: even ±0.03 rad mid-stance twists the supporting foot and made
    ~94% of training episodes end in bad orientation.  HIGH_START and late
    HIGH_END are exempt and actor observations are never touched.
    """

    for name, value in (
        ("vy_range", vy_range),
        ("yaw_rate_range", yaw_rate_range),
    ):
        if value[0] > value[1]:
            raise ValueError(f"{name} minimum must not exceed maximum")
    if high_end_window_s <= 0.0:
        raise ValueError("high_end_window_s must be positive")
    _, ids = _env_ids(env, env_ids)
    if ids.numel() == 0:
        return
    stage = getattr(env, "spatial_course_stage_buf", None)
    high_end_elapsed = getattr(env, "spatial_high_end_elapsed_s_buf", None)
    if stage is None or high_end_elapsed is None:
        raise RuntimeError(
            "spatial transition push requires course stage and high-end timing buffers"
        )
    in_window = (stage == SPATIAL_LOW) | (
        (stage == SPATIAL_HIGH_END)
        & (high_end_elapsed <= float(high_end_window_s))
    )
    selected = ids[in_window[ids]]
    if selected.numel() == 0:
        return

    asset = env.scene[asset_cfg.name]
    count = int(selected.numel())
    velocity = asset.data.root_vel_w[selected].clone()
    velocity[:, 1] += math_utils.sample_uniform(
        float(vy_range[0]),
        float(vy_range[1]),
        (count,),
        device=asset.device,
    )
    velocity[:, 5] += math_utils.sample_uniform(
        float(yaw_rate_range[0]),
        float(yaw_rate_range[1]),
        (count,),
        device=asset.device,
    )
    asset.write_root_velocity_to_sim(velocity, env_ids=selected)

    count_buf = getattr(env, "spatial_transition_push_count_buf", None)
    if count_buf is None:
        count_buf = torch.zeros(
            env.scene.num_envs, device=env.device, dtype=torch.long
        )
        env.spatial_transition_push_count_buf = count_buf
    count_buf[selected] += 1
