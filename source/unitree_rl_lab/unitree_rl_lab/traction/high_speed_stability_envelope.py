"""Deployable high-speed stability command envelope.

The state machine consumes only the 1864-D actor observation and the upstream
command that would otherwise be applied.  It is independent from Isaac Sim and
can therefore be replayed offline or ported to the robot runtime.  Its default
v1 output is *only attenuating*: lateral/yaw commands are preserved and the
magnitude of the requested forward command can never increase.  A separately
opt-in straight-heading correction may add a bounded yaw command after WARN;
it does not alter the forward-command attenuation invariant.

The first version treats observation column 1863 as heading error relative to
the reset heading.  That is valid for straight walking.  When the five-frame
mean yaw command is non-zero, heading thresholds are disabled; roll/pitch
angular-rate, tilt, and action emergency checks remain active.  A later turning
version must replace the reset-heading signal with error relative to an
integrated commanded heading before heading thresholds may be enabled while
turning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


NORMAL = 0
WARN = 1
LIMIT = 2
EMERGENCY = 3
STABILITY_STATE_NAMES = {
    NORMAL: "NORMAL",
    WARN: "WARN",
    LIMIT: "LIMIT",
    EMERGENCY: "EMERGENCY",
}

REASON_HEADING_WARN = 1 << 0
REASON_HEADING_LIMIT_045 = 1 << 1
REASON_HEADING_LIMIT_048 = 1 << 2
REASON_HEADING_OMEGA = 1 << 3
REASON_TURNING_OMEGA = 1 << 4
REASON_TILT = 1 << 5
REASON_ACTION_NORM = 1 << 6
REASON_ACTION_SATURATION = 1 << 7
STABILITY_REASON_NAMES = {
    REASON_HEADING_WARN: "heading_warn_persistent",
    REASON_HEADING_LIMIT_045: "heading_limit_045_persistent",
    REASON_HEADING_LIMIT_048: "heading_limit_048_persistent",
    REASON_HEADING_OMEGA: "heading_and_roll_pitch_rate_emergency",
    REASON_TURNING_OMEGA: "turning_roll_pitch_rate_emergency",
    REASON_TILT: "projected_gravity_tilt_emergency",
    REASON_ACTION_NORM: "action_norm_emergency",
    REASON_ACTION_SATURATION: "action_saturation_emergency",
}

POLICY_OBSERVATION_DIM = 1864
COMMAND_X_INDICES = (30, 33, 36, 39, 42)
COMMAND_YAW_INDICES = (32, 35, 38, 41, 44)
LATEST_BASE_ANG_VEL_SLICE = slice(12, 15)
LATEST_PROJECTED_GRAVITY_SLICE = slice(27, 30)
LAST_ACTION_HISTORY_SLICE = slice(335, 480)
ACTION_DIM = 29
HISTORY_FRAMES = 5
BASE_ANG_VEL_OBSERVATION_SCALE = 0.2
RELATIVE_HEADING_INDEX = 1863


@dataclass(frozen=True)
class HighSpeedStabilityEnvelopeCfg:
    """Thresholds for :class:`HighSpeedStabilityEnvelope`."""

    high_speed_command_threshold: float = 0.65
    turning_yaw_command_threshold: float = 0.05
    warn_heading_threshold: float = 0.40
    warn_persistence_steps: int = 5
    warn_speed_cap: float = 0.55
    limit_heading_threshold: float = 0.45
    limit_persistence_steps: int = 5
    hard_limit_heading_threshold: float = 0.48
    hard_limit_persistence_steps: int = 3
    limit_speed_cap: float = 0.40
    emergency_omega_xy_threshold: float = 1.20
    emergency_tilt_threshold: float = 0.18
    emergency_action_norm_threshold: float = 4.0
    emergency_action_component_threshold: float = 2.5
    emergency_action_component_count: int = 2
    emergency_speed_cap: float = 0.25
    recovery_heading_threshold: float = 0.30
    recovery_tilt_threshold: float = 0.10
    recovery_omega_xy_threshold: float = 0.80
    recovery_persistence_steps: int = 10
    enable_heading_correction: bool = False
    heading_correction_gain: float = 0.80
    heading_correction_abs_cap: float = 0.40
    heading_correction_integral_gain: float = 0.0
    heading_correction_integral_abs_cap: float = 0.20
    heading_correction_integral_decay: float = 0.995
    heading_correction_activate_always: bool = False

    def validate(self) -> None:
        scalar_values = {
            name: value
            for name, value in vars(self).items()
            if not name.endswith("_steps") and name != "emergency_action_component_count"
        }
        if not all(
            torch.isfinite(torch.tensor(float(value))).item()
            for value in scalar_values.values()
        ):
            raise ValueError("stability-envelope scalar parameters must be finite")
        if self.high_speed_command_threshold < 0.0:
            raise ValueError("high_speed_command_threshold must be non-negative")
        if self.turning_yaw_command_threshold < 0.0:
            raise ValueError("turning_yaw_command_threshold must be non-negative")
        if not (
            0.0 <= self.recovery_heading_threshold
            < self.warn_heading_threshold
            < self.limit_heading_threshold
            < self.hard_limit_heading_threshold
        ):
            raise ValueError("heading thresholds must satisfy recovery < warn < limit < hard limit")
        if not (
            0.0 <= self.recovery_tilt_threshold < self.emergency_tilt_threshold
        ):
            raise ValueError("tilt thresholds must satisfy recovery < emergency")
        if not (
            0.0
            <= self.recovery_omega_xy_threshold
            < self.emergency_omega_xy_threshold
        ):
            raise ValueError("angular-rate thresholds must satisfy recovery < emergency")
        for name in ("warn_speed_cap", "limit_speed_cap", "emergency_speed_cap"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not (
            self.emergency_speed_cap <= self.limit_speed_cap <= self.warn_speed_cap
        ):
            raise ValueError("speed caps must satisfy emergency <= limit <= warn")
        if self.emergency_action_norm_threshold <= 0.0:
            raise ValueError("emergency_action_norm_threshold must be positive")
        if self.emergency_action_component_threshold <= 0.0:
            raise ValueError("emergency_action_component_threshold must be positive")
        if not 1 <= self.emergency_action_component_count <= ACTION_DIM:
            raise ValueError("emergency_action_component_count is out of range")
        for name in (
            "warn_persistence_steps",
            "limit_persistence_steps",
            "hard_limit_persistence_steps",
            "recovery_persistence_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.heading_correction_gain < 0.0:
            raise ValueError("heading_correction_gain must be non-negative")
        if self.heading_correction_abs_cap <= 0.0:
            raise ValueError("heading_correction_abs_cap must be positive")
        if self.heading_correction_integral_gain < 0.0:
            raise ValueError("heading_correction_integral_gain must be non-negative")
        if self.heading_correction_integral_abs_cap <= 0.0:
            raise ValueError("heading_correction_integral_abs_cap must be positive")
        if not 0.0 <= self.heading_correction_integral_decay <= 1.0:
            raise ValueError("heading_correction_integral_decay must be in [0, 1]")


@dataclass(frozen=True)
class HighSpeedStabilityEnvelopeOutput:
    """One batched, observation-causal state-machine update."""

    upstream_command: torch.Tensor
    effective_command: torch.Tensor
    state: torch.Tensor
    reason_mask: torch.Tensor
    intervened: torch.Tensor
    command_mean: torch.Tensor
    heading_command_mean: torch.Tensor
    heading_enabled: torch.Tensor
    heading_error: torch.Tensor
    heading_error_abs: torch.Tensor
    heading_correction_active: torch.Tensor
    heading_correction_yaw: torch.Tensor
    omega_xy: torch.Tensor
    tilt: torch.Tensor
    previous_action_norm: torch.Tensor
    current_action_norm: torch.Tensor
    action_slew_norm: torch.Tensor
    action_saturation_count: torch.Tensor
    warn_count: torch.Tensor
    limit_count: torch.Tensor
    hard_limit_count: torch.Tensor
    recovery_count: torch.Tensor


class HighSpeedStabilityEnvelope:
    """Batched NORMAL/WARN/LIMIT/EMERGENCY forward-command envelope.

    Entering a more conservative state takes effect as soon as that state's
    trigger and configured persistence are satisfied.  A state is never relaxed
    merely because its trigger disappeared: all recovery observables must stay
    inside their stricter thresholds for ``recovery_persistence_steps``.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        cfg: HighSpeedStabilityEnvelopeCfg | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.cfg = cfg or HighSpeedStabilityEnvelopeCfg()
        self.cfg.validate()
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.state = torch.full(
            (self.num_envs,), NORMAL, dtype=torch.long, device=self.device
        )
        self.warn_count = torch.zeros_like(self.state)
        self.limit_count = torch.zeros_like(self.state)
        self.hard_limit_count = torch.zeros_like(self.state)
        self.recovery_count = torch.zeros_like(self.state)
        # The actor observation contains a five-sample command history.  Keep
        # the matching causal history of yaw that *this envelope* injected so
        # operator turn intent can be recovered without feeding our own
        # correction back into the turn detector.  A single previous sample is
        # insufficient: after one corrected step its contribution remains in
        # the observation mean for HISTORY_FRAMES updates.
        self._heading_correction_history = torch.zeros(
            (self.num_envs, HISTORY_FRAMES), device=self.device
        )
        self._heading_integral = torch.zeros(
            self.num_envs, device=self.device
        )

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        """Reset all environments or selected environment indices."""

        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            if ids.ndim != 1:
                raise ValueError("env_ids must be one-dimensional")
            if ids.numel() and ((ids < 0).any() or (ids >= self.num_envs).any()):
                raise IndexError("env_ids contains an out-of-range index")
        self.state[ids] = NORMAL
        self.warn_count[ids] = 0
        self.limit_count[ids] = 0
        self.hard_limit_count[ids] = 0
        self.recovery_count[ids] = 0
        self._heading_correction_history[ids] = 0.0
        self._heading_integral[ids] = 0.0

    def update(
        self,
        policy_observation: torch.Tensor,
        upstream_command: torch.Tensor,
    ) -> HighSpeedStabilityEnvelopeOutput:
        """Update from actor-visible history and return an attenuated command.

        The newest ``last_action`` history sample is the action currently held
        by the controller at observation time; the preceding sample provides
        the causal slew reference.  This avoids a second policy inference and
        remains compatible with stateful policy wrappers.
        """

        observation = torch.as_tensor(policy_observation, device=self.device)
        if observation.shape != (self.num_envs, POLICY_OBSERVATION_DIM):
            raise ValueError(
                "policy_observation must have shape "
                f"{(self.num_envs, POLICY_OBSERVATION_DIM)}, got {tuple(observation.shape)}"
            )
        if not observation.is_floating_point():
            observation = observation.float()
        if not torch.isfinite(observation).all():
            raise FloatingPointError("policy_observation must be finite")

        command = torch.as_tensor(
            upstream_command, device=self.device, dtype=observation.dtype
        )
        if command.shape != (self.num_envs, 3):
            raise ValueError(
                "upstream_command must have shape "
                f"{(self.num_envs, 3)}, got {tuple(command.shape)}"
            )
        if not torch.isfinite(command).all():
            raise FloatingPointError("upstream_command must be finite")

        command_x_indices = torch.tensor(
            COMMAND_X_INDICES, dtype=torch.long, device=self.device
        )
        command_yaw_indices = torch.tensor(
            COMMAND_YAW_INDICES, dtype=torch.long, device=self.device
        )
        command_mean = torch.stack(
            (
                observation.index_select(1, command_x_indices).mean(dim=1),
                observation[:, [31, 34, 37, 40, 43]].mean(dim=1),
                observation.index_select(1, command_yaw_indices).mean(dim=1),
            ),
            dim=1,
        )
        # The evaluator synchronizes all five history frames to the command
        # applied after this update.  Keep persistent heading supervision alive
        # after WARN has capped that history by also considering the still-high
        # upstream request.  If an upstream health envelope has already reduced
        # the request, its more conservative value naturally disables this
        # high-speed-only branch.
        high_speed = torch.maximum(
            command_mean[:, 0].abs(), command[:, 0].abs()
        ) >= self.cfg.high_speed_command_threshold
        # The observation manager advances one command-history sample per
        # control step.  Subtract the mean of the matching five causal samples
        # injected by this envelope.  Subtracting only the last full correction
        # would leave +/-correction/5 in the raw history on the next step and
        # produce a one-frame-on/five-frames-off square wave.  The current
        # upstream command is checked independently so a newly requested turn
        # becomes transparent immediately rather than waiting for history.
        heading_command_mean = command_mean[:, 2] - (
            self._heading_correction_history.mean(dim=1)
        )
        heading_enabled = (
            heading_command_mean.abs() <= self.cfg.turning_yaw_command_threshold
        ) & (
            command[:, 2].abs() <= self.cfg.turning_yaw_command_threshold
        )
        heading_error = observation[:, RELATIVE_HEADING_INDEX]
        heading_error_abs = heading_error.abs()
        angular_velocity = (
            observation[:, LATEST_BASE_ANG_VEL_SLICE]
            / BASE_ANG_VEL_OBSERVATION_SCALE
        )
        omega_xy = torch.linalg.vector_norm(angular_velocity[:, :2], dim=1)
        projected_gravity = observation[:, LATEST_PROJECTED_GRAVITY_SLICE]
        tilt = torch.linalg.vector_norm(projected_gravity[:, :2], dim=1)
        action_history = observation[:, LAST_ACTION_HISTORY_SLICE].reshape(
            self.num_envs, HISTORY_FRAMES, ACTION_DIM
        )
        previous_action = action_history[:, -2]
        current_action = action_history[:, -1]
        previous_action_norm = torch.linalg.vector_norm(previous_action, dim=1)
        current_action_norm = torch.linalg.vector_norm(current_action, dim=1)
        action_slew_norm = torch.linalg.vector_norm(
            current_action - previous_action, dim=1
        )
        action_saturation_count = (
            current_action.abs() >= self.cfg.emergency_action_component_threshold
        ).sum(dim=1)

        heading_guard = high_speed & heading_enabled
        warn_condition = heading_guard & (
            heading_error_abs > self.cfg.warn_heading_threshold
        )
        limit_condition = heading_guard & (
            heading_error_abs > self.cfg.limit_heading_threshold
        )
        hard_limit_condition = heading_guard & (
            heading_error_abs > self.cfg.hard_limit_heading_threshold
        )
        self.warn_count = torch.where(
            warn_condition, self.warn_count + 1, torch.zeros_like(self.warn_count)
        )
        self.limit_count = torch.where(
            limit_condition, self.limit_count + 1, torch.zeros_like(self.limit_count)
        )
        self.hard_limit_count = torch.where(
            hard_limit_condition,
            self.hard_limit_count + 1,
            torch.zeros_like(self.hard_limit_count),
        )
        warn_ready = self.warn_count >= self.cfg.warn_persistence_steps
        limit_ready = self.limit_count >= self.cfg.limit_persistence_steps
        hard_limit_ready = (
            self.hard_limit_count >= self.cfg.hard_limit_persistence_steps
        )

        heading_omega_emergency = (
            heading_enabled
            & (heading_error_abs > self.cfg.hard_limit_heading_threshold)
            & (omega_xy > self.cfg.emergency_omega_xy_threshold)
        )
        # Reset-relative heading is invalid during a commanded turn.  Roll and
        # pitch rate remain valid, so retain the angular-rate emergency without
        # consulting the heading channel in that case.
        turning_omega_emergency = (~heading_enabled) & (
            omega_xy > self.cfg.emergency_omega_xy_threshold
        )
        tilt_emergency = tilt > self.cfg.emergency_tilt_threshold
        action_norm_emergency = (
            current_action_norm > self.cfg.emergency_action_norm_threshold
        )
        action_saturation_emergency = (
            action_saturation_count >= self.cfg.emergency_action_component_count
        )
        emergency = (
            heading_omega_emergency
            | turning_omega_emergency
            | tilt_emergency
            | action_norm_emergency
            | action_saturation_emergency
        )

        reason_mask = torch.zeros_like(self.state)
        reason_mask |= warn_ready.to(torch.long) * REASON_HEADING_WARN
        reason_mask |= limit_ready.to(torch.long) * REASON_HEADING_LIMIT_045
        reason_mask |= hard_limit_ready.to(torch.long) * REASON_HEADING_LIMIT_048
        reason_mask |= (
            heading_omega_emergency.to(torch.long) * REASON_HEADING_OMEGA
        )
        reason_mask |= (
            turning_omega_emergency.to(torch.long) * REASON_TURNING_OMEGA
        )
        reason_mask |= tilt_emergency.to(torch.long) * REASON_TILT
        reason_mask |= action_norm_emergency.to(torch.long) * REASON_ACTION_NORM
        reason_mask |= (
            action_saturation_emergency.to(torch.long)
            * REASON_ACTION_SATURATION
        )

        desired_state = torch.full_like(self.state, NORMAL)
        desired_state = torch.where(warn_ready, WARN, desired_state)
        desired_state = torch.where(
            limit_ready | hard_limit_ready, LIMIT, desired_state
        )
        desired_state = torch.where(emergency, EMERGENCY, desired_state)
        worsening = desired_state > self.state
        self.state = torch.where(worsening, desired_state, self.state)

        # A temporarily disabled reset-relative heading channel must not be
        # interpreted as evidence of recovery.  Once the envelope has entered
        # a conservative state, actual heading, tilt and angular rate all have
        # to satisfy the recovery envelope for the full hysteresis interval.
        heading_recovered = (
            heading_error_abs < self.cfg.recovery_heading_threshold
        )
        recovery_condition = (
            heading_recovered
            & (tilt < self.cfg.recovery_tilt_threshold)
            & (omega_xy < self.cfg.recovery_omega_xy_threshold)
            & ~emergency
        )
        self.recovery_count = torch.where(
            (self.state != NORMAL) & recovery_condition,
            self.recovery_count + 1,
            torch.zeros_like(self.recovery_count),
        )
        recovered = (
            (self.state != NORMAL)
            & (self.recovery_count >= self.cfg.recovery_persistence_steps)
        )
        self.state = torch.where(recovered, torch.full_like(self.state, NORMAL), self.state)
        self.recovery_count = torch.where(
            recovered, torch.zeros_like(self.recovery_count), self.recovery_count
        )

        cap = torch.full(
            (self.num_envs,), float("inf"), dtype=command.dtype, device=self.device
        )
        cap = torch.where(
            self.state == WARN, torch.full_like(cap, self.cfg.warn_speed_cap), cap
        )
        cap = torch.where(
            self.state == LIMIT, torch.full_like(cap, self.cfg.limit_speed_cap), cap
        )
        cap = torch.where(
            self.state == EMERGENCY,
            torch.full_like(cap, self.cfg.emergency_speed_cap),
            cap,
        )
        effective = command.clone()
        effective[:, 0] = command[:, 0].sign() * torch.minimum(
            command[:, 0].abs(), cap
        )
        heading_correction_active = self.cfg.enable_heading_correction & (
            heading_enabled
            & (
                (self.state >= WARN)
                | bool(self.cfg.heading_correction_activate_always)
            )
        )
        finite_heading_error = torch.nan_to_num(
            heading_error, nan=0.0, posinf=0.0, neginf=0.0
        )
        self._heading_integral = (
            self._heading_integral * float(self.cfg.heading_correction_integral_decay)
            + torch.where(
                heading_correction_active,
                finite_heading_error.detach(),
                torch.zeros_like(finite_heading_error),
            )
        )
        integral_scale = max(
            float(self.cfg.heading_correction_integral_gain), 1.0e-9
        )
        integral_clipped = torch.clamp(
            self._heading_integral,
            -float(self.cfg.heading_correction_integral_abs_cap) / integral_scale,
            float(self.cfg.heading_correction_integral_abs_cap) / integral_scale,
        )
        correction = (
            float(self.cfg.heading_correction_gain) * finite_heading_error
            + float(self.cfg.heading_correction_integral_gain) * integral_clipped
        )
        corrected_yaw = torch.clamp(
            command[:, 2] - correction,
            -self.cfg.heading_correction_abs_cap,
            self.cfg.heading_correction_abs_cap,
        )
        effective[:, 2] = torch.where(
            heading_correction_active, corrected_yaw, command[:, 2]
        )
        heading_correction_yaw = effective[:, 2] - command[:, 2]
        self._heading_correction_history = torch.roll(
            self._heading_correction_history, shifts=-1, dims=1
        )
        self._heading_correction_history[:, -1] = heading_correction_yaw
        intervened = (effective - command).abs().amax(dim=1) > 1.0e-7

        return HighSpeedStabilityEnvelopeOutput(
            upstream_command=command.clone(),
            effective_command=effective,
            state=self.state.clone(),
            reason_mask=reason_mask,
            intervened=intervened,
            command_mean=command_mean,
            heading_command_mean=heading_command_mean,
            heading_enabled=heading_enabled,
            heading_error=heading_error,
            heading_error_abs=heading_error_abs,
            heading_correction_active=heading_correction_active,
            heading_correction_yaw=heading_correction_yaw,
            omega_xy=omega_xy,
            tilt=tilt,
            previous_action_norm=previous_action_norm,
            current_action_norm=current_action_norm,
            action_slew_norm=action_slew_norm,
            action_saturation_count=action_saturation_count,
            warn_count=self.warn_count.clone(),
            limit_count=self.limit_count.clone(),
            hard_limit_count=self.hard_limit_count.clone(),
            recovery_count=self.recovery_count.clone(),
        )


def summarize_high_speed_stability_trace(
    *,
    upstream_command: torch.Tensor,
    effective_command: torch.Tensor,
    state: torch.Tensor,
    reason_mask: torch.Tensor,
    intervened: torch.Tensor,
    heading_enabled: torch.Tensor,
    heading_command_mean: torch.Tensor,
    heading_error: torch.Tensor,
    heading_error_abs: torch.Tensor,
    heading_correction_active: torch.Tensor,
    heading_correction_yaw: torch.Tensor,
    omega_xy: torch.Tensor,
    tilt: torch.Tensor,
) -> dict[str, object]:
    """Build a JSON-compatible summary from aligned envelope samples."""

    upstream = torch.as_tensor(upstream_command, dtype=torch.float64, device="cpu")
    effective = torch.as_tensor(effective_command, dtype=torch.float64, device="cpu")
    states = torch.as_tensor(state, dtype=torch.long, device="cpu").reshape(-1)
    reasons = torch.as_tensor(reason_mask, dtype=torch.long, device="cpu").reshape(-1)
    intervened_b = torch.as_tensor(intervened, dtype=torch.bool, device="cpu").reshape(-1)
    heading_enabled_b = torch.as_tensor(
        heading_enabled, dtype=torch.bool, device="cpu"
    ).reshape(-1)
    heading_command = torch.as_tensor(
        heading_command_mean, dtype=torch.float64, device="cpu"
    ).reshape(-1)
    signed_heading = torch.as_tensor(
        heading_error, dtype=torch.float64, device="cpu"
    ).reshape(-1)
    heading = torch.as_tensor(
        heading_error_abs, dtype=torch.float64, device="cpu"
    ).reshape(-1)
    correction_active = torch.as_tensor(
        heading_correction_active, dtype=torch.bool, device="cpu"
    ).reshape(-1)
    correction_yaw = torch.as_tensor(
        heading_correction_yaw, dtype=torch.float64, device="cpu"
    ).reshape(-1)
    omega = torch.as_tensor(omega_xy, dtype=torch.float64, device="cpu").reshape(-1)
    gravity_tilt = torch.as_tensor(tilt, dtype=torch.float64, device="cpu").reshape(-1)
    count = int(states.numel())
    if upstream.shape != (count, 3) or effective.shape != (count, 3):
        raise ValueError("command traces must be aligned [samples,3]")
    for name, value in (
        ("reason_mask", reasons),
        ("intervened", intervened_b),
        ("heading_enabled", heading_enabled_b),
        ("heading_command_mean", heading_command),
        ("heading_error", signed_heading),
        ("heading_error_abs", heading),
        ("heading_correction_active", correction_active),
        ("heading_correction_yaw", correction_yaw),
        ("omega_xy", omega),
        ("tilt", gravity_tilt),
    ):
        if value.shape != (count,):
            raise ValueError(f"{name} trace must be aligned [samples]")
    if count and not torch.isin(
        states, torch.tensor(tuple(STABILITY_STATE_NAMES), dtype=torch.long)
    ).all():
        raise ValueError("state trace contains an unknown stability state")
    if (reasons < 0).any():
        raise ValueError("reason masks must be non-negative")
    if not all(
        torch.isfinite(value).all()
        for value in (
            upstream,
            effective,
            signed_heading,
            heading,
            heading_command,
            correction_yaw,
            omega,
            gravity_tilt,
        )
    ):
        raise ValueError("stability trace contains a non-finite value")

    def _stats(values: torch.Tensor) -> dict[str, float | None]:
        if values.numel() == 0:
            return {"mean": None, "min": None, "max": None}
        return {
            "mean": float(values.mean().item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
        }

    by_state: dict[str, object] = {}
    for value, name in STABILITY_STATE_NAMES.items():
        mask = states == value
        by_state[name] = {
            "samples": int(mask.sum().item()),
            "fraction": float(mask.float().mean().item()) if count else 0.0,
            "upstream_vx_m_s": _stats(upstream[mask, 0]),
            "effective_vx_m_s": _stats(effective[mask, 0]),
        }
    reason_counts = {
        name: int(((reasons & bit) != 0).sum().item())
        for bit, name in STABILITY_REASON_NAMES.items()
    }
    reduction = (upstream[:, 0].abs() - effective[:, 0].abs()).clamp_min(0.0)
    return {
        "definition": "deployable-high-speed-stability-envelope-v1",
        "sample_count": count,
        "state_encoding": {
            name: value for value, name in STABILITY_STATE_NAMES.items()
        },
        "reason_bits": {
            name: bit for bit, name in STABILITY_REASON_NAMES.items()
        },
        "intervention_fraction": (
            float(intervened_b.float().mean().item()) if count else 0.0
        ),
        "heading_threshold_enabled_fraction": (
            float(heading_enabled_b.float().mean().item()) if count else 0.0
        ),
        "heading_correction_active_fraction": (
            float(correction_active.float().mean().item()) if count else 0.0
        ),
        "upstream_vx_m_s": _stats(upstream[:, 0]),
        "effective_vx_m_s": _stats(effective[:, 0]),
        "absolute_vx_reduction_m_s": _stats(reduction),
        "relative_heading_rad": _stats(signed_heading),
        "inferred_heading_command_mean_rad_s": _stats(heading_command),
        "heading_error_abs_rad": _stats(heading),
        "heading_correction_yaw_rad_s": _stats(correction_yaw),
        "roll_pitch_rate_rad_s": _stats(omega),
        "projected_gravity_tilt": _stats(gravity_tilt),
        "by_state": by_state,
        "reason_sample_counts": reason_counts,
    }


__all__ = [
    "ACTION_DIM",
    "COMMAND_X_INDICES",
    "COMMAND_YAW_INDICES",
    "EMERGENCY",
    "HighSpeedStabilityEnvelope",
    "HighSpeedStabilityEnvelopeCfg",
    "HighSpeedStabilityEnvelopeOutput",
    "LIMIT",
    "NORMAL",
    "POLICY_OBSERVATION_DIM",
    "STABILITY_REASON_NAMES",
    "STABILITY_STATE_NAMES",
    "WARN",
    "summarize_high_speed_stability_trace",
]
