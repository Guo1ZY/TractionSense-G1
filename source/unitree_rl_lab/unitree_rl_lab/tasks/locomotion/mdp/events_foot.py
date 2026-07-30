# Copyright (c) 2025 local foot-sensor extension.
# SPDX-License-Identifier: BSD-3-Clause
"""Foot / friction events for Adaptive-V2 (stores privileged μ, sensor dropout)."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.events import randomize_rigid_body_material
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg


class randomize_friction_with_buffer(ManagerTermBase):
    """Randomize one coherent material per environment and store its exact μ.

    Isaac Lab's stock term samples a separate material bucket for every rigid
    shape.  That is useful as generic domain randomization, but it does not
    represent a floor with one friction coefficient.  The previous local
    wrapper also filled ``ground_friction_mu_buf`` with an *independent* random
    number after assigning materials, so the privileged μ did not match the
    physics.

    This term samples one bucket ID per environment, applies that material to
    every selected shape, and writes the very same static-friction value to the
    critic/teacher buffer.  The actor never receives this buffer.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # Reuse stock term (same cfg/params) for bucket creation + assignment.
        self._material_term = randomize_rigid_body_material(cfg, env)
        n = env.scene.num_envs
        device = env.device
        if not hasattr(env, "ground_friction_mu_buf"):
            env.ground_friction_mu_buf = torch.full((n,), 0.8, device=device)

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
    ):
        # Material assignment is CPU-side in PhysX, matching the stock term.
        if env_ids is None:
            env_ids_cpu = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids_cpu = env_ids.cpu()
        env_ids_t = env_ids_cpu.to(device=env.device)

        buckets = self._material_term.material_buckets
        bucket_count = int(buckets.shape[0])
        if bucket_count != int(num_buckets):
            raise ValueError(
                f"Configured num_buckets={num_buckets}, but material term owns {bucket_count} buckets."
            )

        # One μ per environment, expanded across all selected robot shapes.
        bucket_ids = torch.randint(0, bucket_count, (len(env_ids_cpu),), device="cpu")
        per_env_material = buckets[bucket_ids]  # (E, static/dynamic/restitution)
        total_num_shapes = self._material_term.asset.root_physx_view.max_shapes
        material_samples = per_env_material[:, None, :].expand(-1, total_num_shapes, -1)

        materials = self._material_term.asset.root_physx_view.get_material_properties()
        if self._material_term.num_shapes_per_body is not None:
            for body_id in self._material_term.asset_cfg.body_ids:
                start_idx = sum(self._material_term.num_shapes_per_body[:body_id])
                end_idx = start_idx + self._material_term.num_shapes_per_body[body_id]
                materials[env_ids_cpu, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
        else:
            materials[env_ids_cpu] = material_samples

        self._material_term.asset.root_physx_view.set_material_properties(materials, env_ids_cpu)

        # Exact teacher label: the same static μ that was assigned to physics.
        lo, hi = map(float, static_friction_range)
        mu = per_env_material[:, 0].to(device=env.device)
        env.ground_friction_mu_buf[env_ids_t] = torch.clamp(mu, lo, hi)


class randomize_teacher_friction_with_buffer(randomize_friction_with_buffer):
    """Assign coherent static/dynamic friction from balanced teacher strata.

    The generic Isaac material randomizer samples static and dynamic friction
    independently.  That is useful for broad robustness, but it makes a
    privileged scalar teacher label ambiguous.  This term builds explicit
    low/medium/high buckets with ``static == dynamic == effective_mu`` and
    samples the strata with configurable probabilities.  Three strata remain
    the default; targeted fine-tunes may split the high-grip interval into a
    shoulder and an extreme tail without changing the exact-mu actor schema.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        ranges = cfg.params.get(
            "teacher_friction_ranges", ((0.05, 0.25), (0.25, 0.75), (0.75, 1.20))
        )
        probabilities = cfg.params.get("regime_probabilities", (0.25, 0.50, 0.25))
        restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))
        count = int(cfg.params.get("num_buckets", 64))
        if len(ranges) != len(probabilities) or len(ranges) < 2:
            raise ValueError(
                "teacher_friction_ranges and regime_probabilities must have the same length >= 2"
            )
        if count < len(ranges):
            raise ValueError(
                f"num_buckets={count} is too small for {len(ranges)} teacher strata"
            )
        weights = torch.tensor(probabilities, dtype=torch.float64, device="cpu")
        if torch.any(weights < 0) or float(weights.sum()) <= 0.0:
            raise ValueError(f"invalid regime_probabilities={probabilities}")
        weights /= weights.sum()
        # Reserve one bucket per stratum, then distribute the remainder by the
        # requested runtime weights.  For the default 25/50/25 and 64 buckets
        # this remains exactly 16/32/16.
        remaining = count - len(ranges)
        fractional = weights * remaining
        counts_t = torch.floor(fractional).to(dtype=torch.long) + 1
        for index in torch.argsort(fractional - torch.floor(fractional), descending=True):
            if int(counts_t.sum()) >= count:
                break
            counts_t[int(index)] += 1
        counts = tuple(int(value) for value in counts_t.tolist())
        bucket_regimes = []
        offset = 0
        for regime, (value_range, regime_count) in enumerate(zip(ranges, counts, strict=True)):
            lo, hi = map(float, value_range)
            mu = torch.empty(regime_count, device="cpu").uniform_(lo, hi)
            self._material_term.material_buckets[offset : offset + regime_count, 0] = mu
            self._material_term.material_buckets[offset : offset + regime_count, 1] = mu
            restitution = torch.empty(regime_count, device="cpu").uniform_(
                float(restitution_range[0]), float(restitution_range[1])
            )
            self._material_term.material_buckets[offset : offset + regime_count, 2] = restitution
            bucket_regimes.extend([regime] * regime_count)
            offset += regime_count
        self._bucket_regimes = torch.tensor(bucket_regimes, dtype=torch.long, device="cpu")
        self._regime_count = len(ranges)
        if not hasattr(env, "effective_friction_mu_buf"):
            env.effective_friction_mu_buf = torch.full(
                (env.scene.num_envs,), 0.5, device=env.device
            )
        if not hasattr(env, "ground_friction_regime_buf"):
            env.ground_friction_regime_buf = torch.ones(
                env.scene.num_envs, dtype=torch.long, device=env.device
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
        teacher_friction_ranges: tuple[tuple[float, float], ...] = (
            (0.05, 0.25),
            (0.25, 0.75),
            (0.75, 1.20),
        ),
        regime_probabilities: tuple[float, ...] = (0.25, 0.50, 0.25),
    ):
        del static_friction_range, dynamic_friction_range, restitution_range
        del asset_cfg, make_consistent, teacher_friction_ranges
        if env_ids is None:
            env_ids_cpu = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids_cpu = env_ids.cpu()
        env_ids_t = env_ids_cpu.to(device=env.device)
        probabilities = torch.tensor(regime_probabilities, dtype=torch.float32, device="cpu")
        probabilities = probabilities / probabilities.sum()
        regimes = torch.multinomial(probabilities, len(env_ids_cpu), replacement=True)
        bucket_ids = torch.empty(len(env_ids_cpu), dtype=torch.long, device="cpu")
        if len(regime_probabilities) != self._regime_count:
            raise ValueError(
                f"expected {self._regime_count} regime probabilities, got {len(regime_probabilities)}"
            )
        for regime in range(self._regime_count):
            selected = regimes == regime
            candidates = torch.nonzero(self._bucket_regimes == regime, as_tuple=False).flatten()
            if torch.any(selected):
                choice = torch.randint(0, candidates.numel(), (int(selected.sum()),), device="cpu")
                bucket_ids[selected] = candidates[choice]

        buckets = self._material_term.material_buckets
        if int(buckets.shape[0]) != int(num_buckets):
            raise ValueError(
                f"Configured num_buckets={num_buckets}, but material term owns {buckets.shape[0]} buckets."
            )
        per_env_material = buckets[bucket_ids]
        total_num_shapes = self._material_term.asset.root_physx_view.max_shapes
        material_samples = per_env_material[:, None, :].expand(-1, total_num_shapes, -1)
        materials = self._material_term.asset.root_physx_view.get_material_properties()
        if self._material_term.num_shapes_per_body is not None:
            for body_id in self._material_term.asset_cfg.body_ids:
                start_idx = sum(self._material_term.num_shapes_per_body[:body_id])
                end_idx = start_idx + self._material_term.num_shapes_per_body[body_id]
                materials[env_ids_cpu, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
        else:
            materials[env_ids_cpu] = material_samples
        self._material_term.asset.root_physx_view.set_material_properties(materials, env_ids_cpu)

        mu = per_env_material[:, 1].to(device=env.device)
        env.ground_friction_mu_buf[env_ids_t] = mu
        env.effective_friction_mu_buf[env_ids_t] = mu
        env.ground_friction_regime_buf[env_ids_t] = regimes.to(device=env.device)


class two_surface_friction_with_buffer(ManagerTermBase):
    """Initialize or alternate between calibrated low/high-friction surfaces.

    The material is still attached to the robot collision shapes, matching the
    existing traction tasks and the terrain's ``multiply`` combine mode.  In
    contrast to episode-only domain randomization, the interval instance of
    this term flips every selected environment to the opposite regime while
    keeping the velocity command untouched.

    Runtime buffers are intentionally public so evaluation code can measure
    the causal response to a switch:

    - ``ground_friction_mu_buf``: exact effective friction used by physics;
    - ``ground_friction_regime_buf``: 0 for low, 2 for high;
    - ``friction_switch_count_buf``: number of in-episode switches;
    - ``friction_switch_step_buf``: policy step of the latest switch;
    - ``friction_switch_direction_buf``: -1 high->low, +1 low->high.

    The deployable actor never observes these buffers.  They are privileged
    labels for the Oracle Teacher, rewards, logging, and Student distillation.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # The stock term resolves the articulation/body selection and exposes
        # shape bookkeeping.  We assign continuous per-environment values
        # directly instead of drawing an unrelated stock material bucket.
        self._material_term = randomize_rigid_body_material(cfg, env)
        n = env.scene.num_envs
        device = env.device
        if not hasattr(env, "ground_friction_mu_buf"):
            env.ground_friction_mu_buf = torch.full((n,), 0.8, device=device)
        if not hasattr(env, "effective_friction_mu_buf"):
            env.effective_friction_mu_buf = torch.full((n,), 0.8, device=device)
        if not hasattr(env, "ground_friction_regime_buf"):
            env.ground_friction_regime_buf = torch.full(
                (n,), 2, dtype=torch.long, device=device
            )
        if not hasattr(env, "friction_switch_is_high_buf"):
            env.friction_switch_is_high_buf = torch.ones(
                n, dtype=torch.bool, device=device
            )
        if not hasattr(env, "friction_switch_count_buf"):
            env.friction_switch_count_buf = torch.zeros(
                n, dtype=torch.long, device=device
            )
        if not hasattr(env, "friction_switch_step_buf"):
            env.friction_switch_step_buf = torch.full(
                (n,), -1, dtype=torch.long, device=device
            )
        if not hasattr(env, "friction_switch_previous_mu_buf"):
            env.friction_switch_previous_mu_buf = torch.full(
                (n,), 0.8, device=device
            )
        if not hasattr(env, "friction_switch_target_mu_buf"):
            env.friction_switch_target_mu_buf = torch.full(
                (n,), 0.8, device=device
            )
        if not hasattr(env, "friction_switch_direction_buf"):
            env.friction_switch_direction_buf = torch.zeros(
                n, dtype=torch.int8, device=device
            )

    @staticmethod
    def _validate_range(name: str, value_range: tuple[float, float]) -> tuple[float, float]:
        low, high = map(float, value_range)
        if low <= 0.0 or high < low:
            raise ValueError(f"invalid {name}={value_range}")
        return low, high

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        low_friction_range: tuple[float, float] = (0.08, 0.20),
        high_friction_range: tuple[float, float] = (0.80, 1.20),
        initial_high_probability: float = 0.5,
        flip_existing: bool = False,
        make_consistent: bool = True,
        teacher_friction_ranges: tuple[tuple[float, float], ...] | None = None,
        regime_probabilities: tuple[float, ...] | None = None,
    ) -> None:
        del static_friction_range, dynamic_friction_range, num_buckets
        del asset_cfg, make_consistent
        # Some inherited G1 robust configs rewrite these generic teacher
        # fields in ``__post_init__``.  The two-surface term owns explicit
        # low/high ranges, so the inherited values are intentionally ignored.
        del teacher_friction_ranges, regime_probabilities
        low_min, low_max = self._validate_range(
            "low_friction_range", low_friction_range
        )
        high_min, high_max = self._validate_range(
            "high_friction_range", high_friction_range
        )
        if low_max >= high_min:
            raise ValueError(
                "low_friction_range must end below high_friction_range: "
                f"{low_friction_range} vs {high_friction_range}"
            )
        if not 0.0 <= float(initial_high_probability) <= 1.0:
            raise ValueError(
                "initial_high_probability must be in [0, 1], got "
                f"{initial_high_probability}"
            )
        restitution_min, restitution_max = map(float, restitution_range)
        if restitution_min < 0.0 or restitution_max < restitution_min:
            raise ValueError(f"invalid restitution_range={restitution_range}")

        if env_ids is None or isinstance(env_ids, slice):
            env_ids_cpu = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
        if env_ids_cpu.numel() == 0:
            return
        env_ids_t = env_ids_cpu.to(device=env.device)
        count = int(env_ids_cpu.numel())

        if flip_existing:
            previous_is_high = env.friction_switch_is_high_buf[env_ids_t]
            is_high = ~previous_is_high
        else:
            previous_is_high = env.friction_switch_is_high_buf[env_ids_t].clone()
            is_high = (
                torch.rand(count, device=env.device)
                < float(initial_high_probability)
            )

        low_mu = torch.empty(count, device=env.device).uniform_(low_min, low_max)
        high_mu = torch.empty(count, device=env.device).uniform_(
            high_min, high_max
        )
        mu = torch.where(is_high, high_mu, low_mu)
        restitution = torch.empty(count, device=env.device).uniform_(
            restitution_min, restitution_max
        )
        per_env_material = torch.stack((mu, mu, restitution), dim=-1).cpu()

        total_num_shapes = self._material_term.asset.root_physx_view.max_shapes
        material_samples = per_env_material[:, None, :].expand(
            -1, total_num_shapes, -1
        )
        materials = self._material_term.asset.root_physx_view.get_material_properties()
        if self._material_term.num_shapes_per_body is not None:
            for body_id in self._material_term.asset_cfg.body_ids:
                start_idx = sum(
                    self._material_term.num_shapes_per_body[:body_id]
                )
                end_idx = (
                    start_idx
                    + self._material_term.num_shapes_per_body[body_id]
                )
                materials[env_ids_cpu, start_idx:end_idx] = material_samples[
                    :, start_idx:end_idx
                ]
        else:
            materials[env_ids_cpu] = material_samples
        self._material_term.asset.root_physx_view.set_material_properties(
            materials, env_ids_cpu
        )

        previous_mu = env.ground_friction_mu_buf[env_ids_t].clone()
        env.ground_friction_mu_buf[env_ids_t] = mu
        env.effective_friction_mu_buf[env_ids_t] = mu
        env.ground_friction_regime_buf[env_ids_t] = torch.where(
            is_high,
            torch.full_like(
                env.ground_friction_regime_buf[env_ids_t], 2
            ),
            torch.zeros_like(env.ground_friction_regime_buf[env_ids_t]),
        )
        env.friction_switch_is_high_buf[env_ids_t] = is_high
        env.friction_switch_previous_mu_buf[env_ids_t] = (
            previous_mu if flip_existing else mu
        )
        env.friction_switch_target_mu_buf[env_ids_t] = mu

        if flip_existing:
            env.friction_switch_count_buf[env_ids_t] += 1
            if hasattr(env, "episode_length_buf"):
                env.friction_switch_step_buf[env_ids_t] = (
                    env.episode_length_buf[env_ids_t]
                )
            env.friction_switch_direction_buf[env_ids_t] = torch.where(
                is_high,
                torch.ones(count, dtype=torch.int8, device=env.device),
                -torch.ones(count, dtype=torch.int8, device=env.device),
            )
        else:
            env.friction_switch_count_buf[env_ids_t] = 0
            env.friction_switch_step_buf[env_ids_t] = -1
            env.friction_switch_direction_buf[env_ids_t] = 0


def randomize_motor_effort_limits(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    strength_range: tuple[float, float] = (0.85, 1.15),
) -> None:
    """Scale the PhysX joint-effort limits by one motor-strength factor per env.

    G1 uses implicit actuators, so changing only a Python-side torque tensor
    does not affect physics.  This event writes the scaled limits through the
    articulation PhysX view and retains the original limits as an immutable
    reference for repeatable calls.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids_t = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids_t = env_ids.to(device=env.device, dtype=torch.long)

    cache_name = f"_robust_default_effort_limits_{asset_cfg.name}"
    if not hasattr(env, cache_name):
        setattr(env, cache_name, asset.data.joint_effort_limits.detach().clone())
    default_limits = getattr(env, cache_name)

    low, high = map(float, strength_range)
    strength = torch.empty((len(env_ids_t), 1), device=env.device).uniform_(low, high)
    limits = default_limits[env_ids_t] * strength
    asset.write_joint_effort_limit_to_sim(limits, env_ids=env_ids_t)

    # Keep implicit-actuator bookkeeping consistent with the PhysX limits.
    for actuator in asset.actuators.values():
        joint_ids = actuator.joint_indices
        selected_limits = limits if isinstance(joint_ids, slice) else limits[:, joint_ids]
        actuator.effort_limit_sim[env_ids_t] = selected_limits
        if actuator.is_implicit_model:
            actuator.effort_limit[env_ids_t] = selected_limits

    if not hasattr(env, "motor_strength_scale_buf"):
        env.motor_strength_scale_buf = torch.ones(env.scene.num_envs, device=env.device)
    env.motor_strength_scale_buf[env_ids_t] = strength[:, 0]


def randomize_foot_sensor_dropout(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    dropout_prob: float = 0.05,
    stale_age: float = 0.3,
) -> None:
    """Episode-level foot sensor dropout DR for validity/age channels."""
    from unitree_rl_lab.tasks.locomotion.mdp import foot_sensor as foot_mdp

    foot_mdp.inject_foot_sensor_dropout(
        env, env_ids=env_ids, dropout_prob=dropout_prob, stale_age=stale_age
    )


def reset_foot_sensor_valid(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Ensure validity buffers exist and start valid after reset (before dropout)."""
    n = env.num_envs
    device = env.device
    if not hasattr(env, "foot_sensor_valid_buf"):
        env.foot_sensor_valid_buf = torch.ones(n, device=device)
        env.foot_sensor_age_buf = torch.zeros(n, device=device)
    if env_ids is None:
        env_ids = torch.arange(n, device=device)
    env.foot_sensor_valid_buf[env_ids] = 1.0
    env.foot_sensor_age_buf[env_ids] = 0.0


def randomize_magnetic_array_proxy(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    sensor_gain_range: tuple[float, float] = (0.72, 1.28),
    axis_gain_range: tuple[float, float] = (0.75, 1.25),
    zero_residual_std: float = 0.06,
    dead_channel_prob: float = 0.015,
    foot_dropout_prob: float = 0.02,
    period_range_s: tuple[float, float] = (0.018, 0.048),
) -> None:
    """Reset-domain event forwarding to the shared magnetic proxy sampler."""

    from .foot_sensor import sample_magnetic_array_proxy

    sample_magnetic_array_proxy(
        env,
        env_ids=env_ids,
        sensor_gain_range=sensor_gain_range,
        axis_gain_range=axis_gain_range,
        zero_residual_std=zero_residual_std,
        dead_channel_prob=dead_channel_prob,
        foot_dropout_prob=foot_dropout_prob,
        period_range_s=period_range_s,
    )


def randomize_structured_foot_sensor(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    gain_range: tuple[float, float] = (0.80, 1.20),
    normal_bias_range: tuple[float, float] = (-0.05, 0.05),
    tangent_bias_range: tuple[float, float] = (-0.03, 0.03),
    lowpass_alpha_range: tuple[float, float] = (0.25, 1.00),
    delay_steps_range: tuple[int, int] = (0, 3),
    sample_dropout_prob_range: tuple[float, float] = (0.0, 0.05),
    burst_dropout_prob_range: tuple[float, float] = (0.0, 0.02),
    burst_length_range: tuple[int, int] = (2, 10),
) -> None:
    """Episode-correlated gain/bias/filter/delay plus temporal dropouts."""
    from unitree_rl_lab.tasks.locomotion.mdp import foot_sensor as foot_mdp

    foot_mdp.sample_structured_foot_sensor_noise(
        env,
        env_ids=env_ids,
        gain_range=gain_range,
        normal_bias_range=normal_bias_range,
        tangent_bias_range=tangent_bias_range,
        lowpass_alpha_range=lowpass_alpha_range,
        delay_steps_range=delay_steps_range,
        sample_dropout_prob_range=sample_dropout_prob_range,
        burst_dropout_prob_range=burst_dropout_prob_range,
        burst_length_range=burst_length_range,
    )
