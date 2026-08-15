"""Health-only command envelope for dual-foot Hall locomotion.

The envelope is deliberately simulator agnostic.  It consumes only packet
health that is available on the robot (valid flags, packet age and finite
checks) and never consumes contact, force, terrain, friction or other
privileged simulation state.  The same state machine can therefore be used by
Isaac evaluation, offline replay and a later robot runtime adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


FAIL_STOP = 0
SINGLE_FOOT = 1
HEALTHY = 2
HEALTH_STATE_NAMES = {
    FAIL_STOP: "FAIL_STOP",
    SINGLE_FOOT: "SINGLE_FOOT",
    HEALTHY: "HEALTHY",
}

# Match the deployed foot bridge watchdog.  The hardened simulation can report
# a 50 ms period with five buffered delay steps, which is the same 250 ms
# worst-case age represented by this threshold.
DEFAULT_MAX_PACKET_AGE_S = 0.25

# Five three-axis command frames in the frozen 1864-D locomotion observation.
# Keeping the tuple here avoids importing an actor implementation into this
# safety-only module.  A contract test guards the schema value.
DEFAULT_COMMAND_X_INDICES = (30, 33, 36, 39, 42)


@dataclass(frozen=True)
class HealthEnvelopeCfg:
    """Configuration for :class:`HealthEnvelope`.

    ``linear_accel_rate`` and ``linear_decel_rate`` are applied independently
    to all three command axes.  A one-foot fallback retains only bounded
    forward/backward motion; lateral and yaw commands are suppressed.  A
    two-foot failure targets an all-zero command.
    """

    single_foot_speed_cap: float = 0.25
    max_packet_age_s: float = DEFAULT_MAX_PACKET_AGE_S
    linear_accel_rate: float = 0.30
    linear_decel_rate: float = 2.00
    recovery_hold_s: float = 0.50

    def validate(self) -> None:
        values = {
            "single_foot_speed_cap": self.single_foot_speed_cap,
            "max_packet_age_s": self.max_packet_age_s,
            "linear_accel_rate": self.linear_accel_rate,
            "linear_decel_rate": self.linear_decel_rate,
            "recovery_hold_s": self.recovery_hold_s,
        }
        if not all(torch.isfinite(torch.tensor(value)).item() for value in values.values()):
            raise ValueError("health-envelope parameters must be finite")
        if self.single_foot_speed_cap < 0.0:
            raise ValueError("single_foot_speed_cap must be non-negative")
        if self.max_packet_age_s <= 0.0:
            raise ValueError("max_packet_age_s must be positive")
        if self.linear_accel_rate <= 0.0 or self.linear_decel_rate <= 0.0:
            raise ValueError("linear acceleration and deceleration rates must be positive")
        if self.recovery_hold_s < 0.0:
            raise ValueError("recovery_hold_s must be non-negative")


@dataclass(frozen=True)
class HealthEnvelopeOutput:
    """One batched update of the health envelope."""

    requested_command: torch.Tensor
    target_command: torch.Tensor
    effective_command: torch.Tensor
    state: torch.Tensor
    valid: torch.Tensor
    packet_age_s: torch.Tensor
    finite: torch.Tensor
    foot_healthy: torch.Tensor
    recovery_timer_s: torch.Tensor
    intervened: torch.Tensor


class HealthEnvelope:
    """Per-environment dual-foot health fallback with recovery hysteresis.

    Health degradation takes effect immediately at the state-machine level.
    The output command then approaches the safer target with the configured
    deceleration rate.  Recovery to a less conservative state requires stable
    evidence for ``recovery_hold_s`` and accelerates only afterwards.
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        device: str | torch.device,
        cfg: HealthEnvelopeCfg | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not torch.isfinite(torch.tensor(dt)).item() or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.cfg = cfg or HealthEnvelopeCfg()
        self.cfg.validate()
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.device = torch.device(device)
        self.state = torch.full(
            (self.num_envs,), FAIL_STOP, dtype=torch.long, device=self.device
        )
        self._recovery_candidate = self.state.clone()
        self.recovery_timer_s = torch.zeros(self.num_envs, device=self.device)
        self.effective_command = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self._initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        """Reset all environments or the selected environment indices."""

        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            if ids.ndim != 1:
                raise ValueError("env_ids must be one-dimensional")
            if ids.numel() and ((ids < 0).any() or (ids >= self.num_envs).any()):
                raise IndexError("env_ids contains an out-of-range index")
        self.state[ids] = FAIL_STOP
        self._recovery_candidate[ids] = FAIL_STOP
        self.recovery_timer_s[ids] = 0.0
        self.effective_command[ids] = 0.0
        self._initialized[ids] = False

    def update(
        self,
        requested_command: torch.Tensor,
        valid: torch.Tensor,
        age_s: torch.Tensor,
        finite: torch.Tensor,
    ) -> HealthEnvelopeOutput:
        """Update the envelope from deployable health signals.

        Args:
            requested_command: Requested ``[vx, vy, yaw_rate]``, shape ``[N,3]``.
            valid: Per-foot packet validity, shape ``[N,2]``.
            age_s: Per-foot packet age in seconds, shape ``[N,2]``.
            finite: Per-foot finite-data check, shape ``[N,2]``.
        """

        requested = self._command_tensor(requested_command)
        valid_b = self._health_tensor(valid, "valid")
        finite_b = self._health_tensor(finite, "finite")
        age = torch.as_tensor(age_s, device=self.device)
        if age.shape != (self.num_envs, 2):
            raise ValueError(
                f"age_s must have shape {(self.num_envs, 2)}, got {tuple(age.shape)}"
            )
        if not age.is_floating_point():
            age = age.to(dtype=requested.dtype)
        else:
            age = age.to(dtype=requested.dtype)

        requested_is_finite = torch.isfinite(requested).all(dim=1)
        safe_requested = torch.nan_to_num(requested, nan=0.0, posinf=0.0, neginf=0.0)
        age_is_finite = torch.isfinite(age)
        # Preserve a finite diagnostic value for malformed ages; ``finite`` and
        # ``foot_healthy`` retain the reason that the packet was rejected.
        safe_age = torch.where(
            age_is_finite,
            age,
            torch.full_like(age, self.cfg.max_packet_age_s + self.dt),
        )
        foot_healthy = (
            valid_b
            & finite_b
            & age_is_finite
            & (safe_age >= 0.0)
            & (safe_age <= self.cfg.max_packet_age_s)
        )
        healthy_count = foot_healthy.sum(dim=1)
        desired_state = torch.where(
            healthy_count == 2,
            torch.full_like(healthy_count, HEALTHY),
            torch.where(
                healthy_count == 1,
                torch.full_like(healthy_count, SINGLE_FOOT),
                torch.full_like(healthy_count, FAIL_STOP),
            ),
        )
        # An invalid command source is itself a fail-stop condition.  It does
        # not falsely mark Hall packets unhealthy in diagnostics.
        desired_state = torch.where(
            requested_is_finite,
            desired_state,
            torch.full_like(desired_state, FAIL_STOP),
        )

        # Hall auto-zero legitimately reports both feet invalid for its first
        # few samples after reset.  Keep the robot stopped but do not start a
        # recovery timer before the health link has ever been established.
        # The first usable one- or two-foot packet initializes directly to the
        # corresponding conservative target.  Hysteresis applies to every
        # degradation/recovery after that first acquisition.
        previously_initialized = self._initialized.clone()
        startup_ready = (healthy_count >= 1) & requested_is_finite
        newly_initialized = ~previously_initialized & startup_ready
        if newly_initialized.any():
            self.state[newly_initialized] = desired_state[newly_initialized]
            self._recovery_candidate[newly_initialized] = desired_state[newly_initialized]
            self.recovery_timer_s[newly_initialized] = 0.0
            self._initialized[newly_initialized] = True

        active = previously_initialized
        worsening = active & (desired_state < self.state)
        if worsening.any():
            self.state[worsening] = desired_state[worsening]
            self._recovery_candidate[worsening] = desired_state[worsening]
            self.recovery_timer_s[worsening] = 0.0

        stable = active & (desired_state == self.state)
        if stable.any():
            self._recovery_candidate[stable] = self.state[stable]
            self.recovery_timer_s[stable] = 0.0

        recovering = active & (desired_state > self.state)
        changed_candidate = recovering & (self._recovery_candidate != desired_state)
        if changed_candidate.any():
            self._recovery_candidate[changed_candidate] = desired_state[changed_candidate]
            self.recovery_timer_s[changed_candidate] = 0.0
        self.recovery_timer_s[recovering] += self.dt
        recovered = recovering & (
            self.recovery_timer_s + 1.0e-9 >= self.cfg.recovery_hold_s
        )
        if recovered.any():
            self.state[recovered] = desired_state[recovered]
            self._recovery_candidate[recovered] = desired_state[recovered]
            self.recovery_timer_s[recovered] = 0.0

        target = safe_requested.clone()
        single = self.state == SINGLE_FOOT
        target[single, 0] = target[single, 0].clamp(
            -self.cfg.single_foot_speed_cap,
            self.cfg.single_foot_speed_cap,
        )
        target[single, 1:] = 0.0
        target[self.state == FAIL_STOP] = 0.0

        # The first valid update establishes the command without an artificial
        # startup ramp.  Subsequent changes use asymmetric slew limits.
        if newly_initialized.any():
            self.effective_command[newly_initialized] = target[newly_initialized]
        slew_ids = ~newly_initialized
        if slew_ids.any():
            current = self.effective_command[slew_ids]
            desired = target[slew_ids]
            same_direction = current * desired >= 0.0
            increasing_magnitude = desired.abs() > current.abs()
            accelerating = same_direction & increasing_magnitude
            rates = torch.where(
                accelerating,
                torch.full_like(current, self.cfg.linear_accel_rate),
                torch.full_like(current, self.cfg.linear_decel_rate),
            )
            maximum_delta = rates * self.dt
            delta = (desired - current).clamp(-maximum_delta, maximum_delta)
            slewed = current + delta
            # Avoid leaving tiny floating-point residual commands at a stop or
            # cap after the slew has effectively reached its target.
            slewed = torch.where(
                (desired - slewed).abs() <= 1.0e-7,
                desired,
                slewed,
            )
            self.effective_command[slew_ids] = slewed

        effective = self.effective_command.clone()
        intervened = (
            (effective - safe_requested).abs().amax(dim=1) > 1.0e-7
        ) | ~requested_is_finite
        return HealthEnvelopeOutput(
            requested_command=safe_requested.clone(),
            target_command=target,
            effective_command=effective,
            state=self.state.clone(),
            valid=valid_b.clone(),
            packet_age_s=safe_age,
            finite=finite_b.clone(),
            foot_healthy=foot_healthy,
            recovery_timer_s=self.recovery_timer_s.clone(),
            intervened=intervened,
        )

    def _command_tensor(self, value: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self.device)
        if tensor.shape != (self.num_envs, 3):
            raise ValueError(
                "requested_command must have shape "
                f"{(self.num_envs, 3)}, got {tuple(tensor.shape)}"
            )
        if not tensor.is_floating_point():
            tensor = tensor.float()
        return tensor

    def _health_tensor(self, value: torch.Tensor, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self.device)
        if tensor.shape != (self.num_envs, 2):
            raise ValueError(
                f"{name} must have shape {(self.num_envs, 2)}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype == torch.bool:
            return tensor
        # Numeric health flags use the same >0.5 convention as actor validity;
        # NaN/Inf never become truthy through a raw bool cast.
        return torch.isfinite(tensor) & (tensor > 0.5)


def rewrite_command_history(
    policy_observation: torch.Tensor,
    effective_command: torch.Tensor,
    command_x_indices: Sequence[int] = DEFAULT_COMMAND_X_INDICES,
) -> torch.Tensor:
    """Clone an actor observation and rewrite every command-history frame."""

    if policy_observation.ndim != 2:
        raise ValueError("policy_observation must be rank two")
    command = torch.as_tensor(
        effective_command,
        device=policy_observation.device,
        dtype=policy_observation.dtype,
    )
    if command.shape != (policy_observation.shape[0], 3):
        raise ValueError(
            "effective_command must have shape "
            f"{(policy_observation.shape[0], 3)}, got {tuple(command.shape)}"
        )
    indices = tuple(int(index) for index in command_x_indices)
    if not indices:
        raise ValueError("command_x_indices must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError("command_x_indices must be unique")
    if min(indices) < 0 or max(indices) + 2 >= policy_observation.shape[1]:
        raise IndexError("command history lies outside the policy observation")
    rewritten = policy_observation.clone()
    for index in indices:
        rewritten[:, index : index + 3] = command
    return rewritten


def summarize_health_envelope_trace(
    *,
    requested_command: torch.Tensor,
    effective_command: torch.Tensor,
    state: torch.Tensor,
    valid: torch.Tensor,
    age_s: torch.Tensor,
    finite: torch.Tensor,
    foot_healthy: torch.Tensor,
    intervened: torch.Tensor,
) -> dict[str, object]:
    """Build a JSON-compatible aggregate from aligned envelope samples."""

    requested = torch.as_tensor(requested_command, dtype=torch.float64, device="cpu")
    effective = torch.as_tensor(effective_command, dtype=torch.float64, device="cpu")
    states = torch.as_tensor(state, dtype=torch.long, device="cpu").reshape(-1)
    valid_b = torch.as_tensor(valid, dtype=torch.bool, device="cpu")
    ages = torch.as_tensor(age_s, dtype=torch.float64, device="cpu")
    finite_b = torch.as_tensor(finite, dtype=torch.bool, device="cpu")
    healthy_b = torch.as_tensor(foot_healthy, dtype=torch.bool, device="cpu")
    intervened_b = torch.as_tensor(intervened, dtype=torch.bool, device="cpu").reshape(-1)
    count = int(states.numel())
    expected_command_shape = (count, 3)
    expected_foot_shape = (count, 2)
    if requested.shape != expected_command_shape or effective.shape != expected_command_shape:
        raise ValueError("requested/effective command traces must be aligned [samples,3]")
    if any(value.shape != expected_foot_shape for value in (valid_b, ages, finite_b, healthy_b)):
        raise ValueError("valid/age/finite/healthy traces must be aligned [samples,2]")
    if intervened_b.shape != (count,):
        raise ValueError("intervened trace must be aligned [samples]")
    if not torch.isfinite(requested).all() or not torch.isfinite(effective).all():
        raise ValueError("command traces must be finite")
    if not torch.isfinite(ages).all():
        raise ValueError("age trace must be finite")
    if count and not torch.isin(states, torch.tensor(tuple(HEALTH_STATE_NAMES))).all():
        raise ValueError("state trace contains an unknown health state")

    def _stats(values: torch.Tensor) -> dict[str, float | None]:
        if values.numel() == 0:
            return {"mean": None, "min": None, "max": None}
        return {
            "mean": float(values.mean().item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
        }

    by_state: dict[str, object] = {}
    for value, name in HEALTH_STATE_NAMES.items():
        mask = states == value
        samples = int(mask.sum().item())
        by_state[name] = {
            "samples": samples,
            "fraction": samples / count if count else 0.0,
            "requested_vx_m_s": _stats(requested[mask, 0]),
            "effective_vx_m_s": _stats(effective[mask, 0]),
        }
    healthy_count = healthy_b.sum(dim=1) if count else torch.empty(0, dtype=torch.long)
    reduction = (requested[:, 0].abs() - effective[:, 0].abs()).clamp_min(0.0)
    return {
        "definition": "hall-health-command-envelope-v1",
        "sample_count": count,
        "state_encoding": {name: value for value, name in HEALTH_STATE_NAMES.items()},
        "intervention_fraction": (
            float(intervened_b.float().mean().item()) if count else 0.0
        ),
        "requested_vx_m_s": _stats(requested[:, 0]),
        "effective_vx_m_s": _stats(effective[:, 0]),
        "absolute_vx_reduction_m_s": _stats(reduction),
        "by_state": by_state,
        "health_patterns": {
            "both_healthy_samples": int((healthy_count == 2).sum().item()),
            "single_healthy_samples": int((healthy_count == 1).sum().item()),
            "neither_healthy_samples": int((healthy_count == 0).sum().item()),
            "invalid_flag_foot_samples": int((~valid_b).sum().item()),
            "nonfinite_foot_samples": int((~finite_b).sum().item()),
        },
        "packet_age_s": _stats(ages.reshape(-1)),
    }


# Explicit alias for call sites that prefer the subsystem-qualified name.
HallHealthEnvelope = HealthEnvelope


__all__ = [
    "DEFAULT_COMMAND_X_INDICES",
    "DEFAULT_MAX_PACKET_AGE_S",
    "FAIL_STOP",
    "HEALTHY",
    "HEALTH_STATE_NAMES",
    "HallHealthEnvelope",
    "HealthEnvelope",
    "HealthEnvelopeCfg",
    "HealthEnvelopeOutput",
    "SINGLE_FOOT",
    "rewrite_command_history",
    "summarize_health_envelope_trace",
]
