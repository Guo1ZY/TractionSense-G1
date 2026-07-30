# Copyright (c) 2022-2025, The Isaac Lab Project Developers / local turn extension.
# SPDX-License-Identifier: BSD-3-Clause
"""Velocity command generators with curriculum limit_ranges + optional pure-spin sampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformLevelVelocityCommand(UniformVelocityCommand):
    """Uniform SE(2) velocity command with optional pure in-place yaw (spin) envs.

    Extra behaviour over :class:`UniformVelocityCommand`:

    * ``rel_spin_envs``: fraction of envs get ``vx=vy=0`` but non-zero ``wz``
      (right-stick in-place turn). Mutually exclusive with standing.
    * ``min_spin_ang_vel``: optional floor on ``|wz|`` for spin envs so they
      practice real turning, not near-zero noise.
    * ``limit_ranges``: used by lin/ang curriculum (same as before).

    With ``rel_spin_envs=0`` behaviour matches stock UniformVelocityCommand.
    """

    cfg: "UniformLevelVelocityCommandCfg"

    def __init__(self, cfg: "UniformLevelVelocityCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.is_spin_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def __str__(self) -> str:
        msg = super().__str__()
        msg += f"\n\tSpin-in-place probability: {self.cfg.rel_spin_envs}"
        if self.cfg.min_spin_ang_vel > 0.0:
            msg += f"\n\tMin |wz| for spin: {self.cfg.min_spin_ang_vel}"
        return msg

    def _resample_command(self, env_ids: Sequence[int]):
        # Sample vx, vy, wz (+ standing / heading flags) as usual.
        super()._resample_command(env_ids)

        n = len(env_ids)
        if n == 0:
            return

        r = torch.empty(n, device=self.device)
        # Spin among non-standing envs. rel_spin_envs is absolute probability over all envs.
        spin_roll = r.uniform_(0.0, 1.0) <= self.cfg.rel_spin_envs
        self.is_spin_env[env_ids] = spin_roll & ~self.is_standing_env[env_ids]

        # Optional: ensure spin envs have meaningful |wz|.
        min_wz = float(self.cfg.min_spin_ang_vel)
        if min_wz > 0.0:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            spin_local = self.is_spin_env[env_ids_t]
            if torch.any(spin_local):
                global_ids = env_ids_t[spin_local]
                lo, hi = self.cfg.ranges.ang_vel_z
                lo_f, hi_f = float(lo), float(hi)
                max_mag = max(abs(lo_f), abs(hi_f))
                if max_mag > min_wz:
                    n_spin = global_ids.numel()
                    mag = torch.empty(n_spin, device=self.device).uniform_(min_wz, max_mag)
                    sign = torch.where(
                        torch.empty(n_spin, device=self.device).uniform_(0.0, 1.0) < 0.5,
                        -torch.ones(n_spin, device=self.device),
                        torch.ones(n_spin, device=self.device),
                    )
                    wz = torch.clamp(mag * sign, min=lo_f, max=hi_f)
                    self.vel_command_b[global_ids, 2] = wz

    def _update_command(self):
        """Standing → zero all; spin → zero lin, keep yaw; heading → as parent."""
        # Heading → ang vel (same as parent).
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[env_ids] - self.robot.data.heading_w[env_ids]
            )
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )

        # Pure in-place yaw: zero linear, keep angular.
        spin_ids = self.is_spin_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[spin_ids, 0] = 0.0
        self.vel_command_b[spin_ids, 1] = 0.0

        # Standing still: zero everything (overrides spin if both set — shouldn't happen).
        standing_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_ids, :] = 0.0


class TractionAdaptiveVelocityCommand(UniformLevelVelocityCommand):
    """Mostly-normal forward commands with a small high-speed probe mixture.

    The regular distribution is capped at 1.0 m/s.  A configurable fraction
    of non-standing environments is deliberately assigned a command in
    ``high_speed_range`` (normally 1.0--1.5 m/s).  Because the probe sampling
    is independent of the randomized friction, the policy experiences both:

    * high μ + high command, where it should accelerate; and
    * low μ + high command, where it should ignore the unsafe request and slow.

    This avoids making 1.5 m/s the default training regime while keeping large
    deployment commands in-distribution.
    """

    cfg: "TractionAdaptiveVelocityCommandCfg"

    def __init__(self, cfg: "TractionAdaptiveVelocityCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.is_high_speed_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def __str__(self) -> str:
        msg = super().__str__()
        msg += f"\n\tHigh-speed probe probability: {self.cfg.high_speed_fraction}"
        msg += f"\n\tHigh-speed probe range: {self.cfg.high_speed_range}"
        return msg

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return
        if not hasattr(self, "is_high_speed_env"):
            self.is_high_speed_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        fraction = float(self.cfg.high_speed_fraction)
        probe = torch.rand(env_ids_t.numel(), device=self.device) < fraction
        probe &= ~self.is_standing_env[env_ids_t]
        self.is_high_speed_env[env_ids_t] = probe

        if torch.any(probe):
            probe_ids = env_ids_t[probe]
            lo, hi = map(float, self.cfg.high_speed_range)
            self.vel_command_b[probe_ids, 0].uniform_(lo, hi)
            # This task is intentionally straight; do not let a parent range
            # or stale command inject lateral/yaw components into probe slots.
            self.vel_command_b[probe_ids, 1:] = 0.0


class TractionTeacherVelocityCommand(UniformLevelVelocityCommand):
    """Balanced commands coupled to the teacher friction stratum.

    The teacher event assigns 25% low-friction, 50% medium-friction and 25%
    high-friction environments.  Low/high strata always receive a high-speed
    request, while 60% of the medium stratum receives an ordinary forward
    request.  The remaining medium environments practice stop, low-speed and
    reverse commands.  Across a large vectorized batch this realizes the
    requested 25/25/30/20 joint distribution without exposing a modified
    command to the policy.
    """

    cfg: "TractionTeacherVelocityCommandCfg"

    def __init__(self, cfg: "TractionTeacherVelocityCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.is_high_speed_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @staticmethod
    def _uniform(count: int, value_range: tuple[float, float], device: str) -> torch.Tensor:
        lo, hi = map(float, value_range)
        return torch.empty(count, device=device).uniform_(lo, hi)

    def _resample_command(self, env_ids: Sequence[int]):
        # Let the parent maintain heading/standing bookkeeping, then replace
        # every planar command with the balanced teacher distribution.
        super()._resample_command(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return

        env = self._env
        if hasattr(env, "ground_friction_regime_buf"):
            regime = env.ground_friction_regime_buf[env_ids_t]
        else:
            mu = getattr(env, "effective_friction_mu_buf", None)
            if mu is None:
                mu = getattr(env, "ground_friction_mu_buf", None)
            if mu is None:
                mu = torch.full((self.num_envs,), 0.5, device=self.device)
            value = mu[env_ids_t]
            regime = torch.where(value <= 0.25, 0, torch.where(value >= 0.75, 2, 1))

        self.vel_command_b[env_ids_t, :] = 0.0
        self.is_standing_env[env_ids_t] = False
        self.is_high_speed_env[env_ids_t] = False

        high_request = torch.zeros_like(regime, dtype=torch.bool)
        for regime_id in self.cfg.high_speed_regimes:
            high_request |= regime == int(regime_id)
        high_ids = env_ids_t[high_request]
        if high_ids.numel() > 0:
            self.vel_command_b[high_ids, 0] = self._uniform(
                high_ids.numel(), self.cfg.high_speed_range, self.device
            )
            self.is_high_speed_env[high_ids] = True

        mid_ids = env_ids_t[~high_request]
        if mid_ids.numel() == 0:
            return
        normal = torch.rand(mid_ids.numel(), device=self.device) < float(self.cfg.mid_normal_fraction)
        normal_ids = mid_ids[normal]
        if normal_ids.numel() > 0:
            self.vel_command_b[normal_ids, 0] = self._uniform(
                normal_ids.numel(), self.cfg.normal_speed_range, self.device
            )

        special_ids = mid_ids[~normal]
        if special_ids.numel() == 0:
            return
        roll = torch.rand(special_ids.numel(), device=self.device)
        stop_limit = float(self.cfg.special_stop_fraction)
        low_limit = stop_limit + 0.5 * (1.0 - stop_limit)
        stop = roll < stop_limit
        low = (roll >= stop_limit) & (roll < low_limit)
        reverse = roll >= low_limit
        self.is_standing_env[special_ids[stop]] = True
        low_ids = special_ids[low]
        reverse_ids = special_ids[reverse]
        if low_ids.numel() > 0:
            self.vel_command_b[low_ids, 0] = self._uniform(
                low_ids.numel(), self.cfg.low_speed_range, self.device
            )
        if reverse_ids.numel() > 0:
            self.vel_command_b[reverse_ids, 0] = self._uniform(
                reverse_ids.numel(), self.cfg.reverse_speed_range, self.device
            )


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """Uniform velocity command + curriculum ``limit_ranges`` + optional spin-in-place."""

    class_type: type = UniformLevelVelocityCommand

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    """Max command envelope reached by lin/ang curriculum."""

    rel_spin_envs: float = 0.0
    """Probability of pure in-place yaw envs (vx=vy=0, wz sampled). Defaults to 0 (off).

    Mutually exclusive with standing. Use ~0.25–0.35 for right-stick turn fine-tunes.
    """

    min_spin_ang_vel: float = 0.0
    """If >0, spin envs resample ``|wz|`` in ``[min_spin_ang_vel, max(|range|)]``.

    Avoids near-zero yaw that wastes the pure-turn slot. Typical: 0.15–0.25.
    """


@configclass
class TractionAdaptiveVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    """Configuration for a default-1.0 / occasional-1.5 command mixture."""

    class_type: type = TractionAdaptiveVelocityCommand

    high_speed_fraction: float = 0.15
    """Fraction of non-standing samples drawn from ``high_speed_range``."""

    high_speed_range: tuple[float, float] = (1.0, 1.5)
    """Forward-only stress range used to teach traction-dependent saturation."""


@configclass
class TractionTeacherVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    """Joint friction/command distribution for the privileged teacher."""

    class_type: type = TractionTeacherVelocityCommand

    high_speed_range: tuple[float, float] = (1.0, 1.5)
    normal_speed_range: tuple[float, float] = (0.30, 1.0)
    low_speed_range: tuple[float, float] = (0.05, 0.30)
    reverse_speed_range: tuple[float, float] = (-0.30, -0.05)
    mid_normal_fraction: float = 0.60
    special_stop_fraction: float = 0.40
    high_speed_regimes: tuple[int, ...] = (0, 2)
    """Friction-stratum indices that always receive high-speed requests."""
