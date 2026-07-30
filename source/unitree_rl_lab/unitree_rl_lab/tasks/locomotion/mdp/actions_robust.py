# Copyright (c) 2026 local Sim2Sim extension.
# SPDX-License-Identifier: BSD-3-Clause
"""Action terms used by the Sim2Sim-robust traction teacher."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer


class RandomDelayedJointPositionAction(JointPositionAction):
    """Joint-position action with an episode-randomized control-step delay.

    The delay is applied once per policy/control step, before the usual action
    scale and default-position offset.  This matches the latency seen by a
    deployed 50-Hz policy more closely than delaying every PhysX sub-step.
    """

    cfg: "RandomDelayedJointPositionActionCfg"

    def __init__(self, cfg: "RandomDelayedJointPositionActionCfg", env):
        if cfg.min_delay < 0 or cfg.max_delay < cfg.min_delay:
            raise ValueError(
                f"Invalid action delay range [{cfg.min_delay}, {cfg.max_delay}]"
            )
        super().__init__(cfg, env)
        self._delay_buffer = DelayBuffer(cfg.max_delay, self.num_envs, self.device)
        self._sample_delay(None)

    def _sample_delay(self, env_ids: Sequence[int] | slice | None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            batch_ids = None if env_ids is None else env_ids
            count = self.num_envs
        else:
            batch_ids = env_ids
            count = len(env_ids)
        if self.cfg.delay_probabilities is None:
            lags = torch.randint(
                low=self.cfg.min_delay,
                high=self.cfg.max_delay + 1,
                size=(count,),
                # DelayBuffer stores torch.int internally in this Isaac Lab build.
                dtype=torch.int,
                device=self.device,
            )
        else:
            probabilities = torch.as_tensor(
                self.cfg.delay_probabilities, dtype=torch.float32, device=self.device
            )
            expected = self.cfg.max_delay - self.cfg.min_delay + 1
            if probabilities.numel() != expected or torch.any(probabilities < 0):
                raise ValueError(
                    "delay_probabilities must contain one non-negative value "
                    f"for each lag in [{self.cfg.min_delay}, {self.cfg.max_delay}]"
                )
            probabilities = probabilities / probabilities.sum().clamp(min=1.0e-8)
            lags = torch.multinomial(probabilities, count, replacement=True)
            lags = (lags + self.cfg.min_delay).to(dtype=torch.int)
        self._delay_buffer.set_time_lag(lags, batch_ids=batch_ids)

    def process_actions(self, actions: torch.Tensor):
        # Keep raw_actions equal to the policy output so last_action preserves
        # the deployment observation semantics; only the applied target lags.
        self._raw_actions[:] = actions
        delayed_actions = self._delay_buffer.compute(actions)
        if self.cfg.raw_clip is not None:
            delayed_actions = torch.clamp(
                delayed_actions,
                min=float(self.cfg.raw_clip[0]),
                max=float(self.cfg.raw_clip[1]),
            )
        self._processed_actions = delayed_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        batch_ids = slice(None) if env_ids is None else env_ids
        self._sample_delay(batch_ids)
        self._delay_buffer.reset(batch_ids)


@configclass
class RandomDelayedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for control-step joint-position latency randomization."""

    class_type: type = RandomDelayedJointPositionAction

    min_delay: int = 0
    """Minimum action delay in policy/control steps."""

    max_delay: int = 2
    """Maximum action delay in policy/control steps."""

    raw_clip: tuple[float, float] | None = None
    """Optional policy-action clamp applied before scale and joint offset."""

    delay_probabilities: tuple[float, ...] | None = None
    """Optional categorical weights ordered from ``min_delay`` to ``max_delay``."""
