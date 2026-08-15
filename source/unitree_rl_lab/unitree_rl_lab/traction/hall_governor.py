"""Causal Hall-risk command governor shared by simulation validation tools.

The governor consumes a learned probability of low usable traction.  That
probability is inferred directly from Hall/proprioceptive history; it is not a
normal/tangential force estimate and no magnetic-to-force inverse appears in
this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


UNKNOWN = 0
LOW = 1
HIGH = 2


@dataclass
class HallTractionGovernorCfg:
    low_speed_limit: float = 0.10
    high_speed_limit: float = 0.90
    critical_speed_limit: float = 0.0
    low_lateral_limit: float = 0.05
    high_lateral_limit: float = 0.35
    low_yaw_limit: float = 0.15
    high_yaw_limit: float = 0.80
    probability_low_enter: float = 0.65
    probability_high_enter: float = 0.35
    probability_critical_enter: float = 0.85
    critical_hold_s: float = 0.04
    probability_ema_alpha: float = 0.20
    state_reference_ema_alpha: float = 0.01
    # A Hall-risk head sees a transient zero-load/first-contact distribution
    # immediately after the bounded launch probe.  Do not compare that
    # zero-load reference directly with the first normal walking cycles:
    # briefly adapt the per-foot reference at a faster rate, then enable
    # relative-rise braking.  This is still entirely causal Hall/proprio
    # processing -- it is not a force, friction, or slip estimate.
    reference_settle_s: float = 0.60
    reference_settle_alpha: float = 0.25
    # Optional one-step Hall-only pre-brake.  A prospective-risk rise can
    # arrive a fraction of a gait cycle before the ordinary LOW hold is
    # confirmed.  When enabled, it limits the commanded envelope immediately
    # but deliberately leaves the traction state unchanged until the normal
    # hysteresis evidence is complete.
    prebrake_probability: float | None = None
    prebrake_relative_rise: float | None = None
    prebrake_speed_limit: float | None = None
    relative_low_rise: float = 0.12
    relative_high_drop: float = 0.12
    relative_low_min_probability: float | None = None
    allow_absolute_high_clear: bool = False
    low_hold_s: float = 0.20
    high_hold_s: float = 1.00
    min_detection_command: float = 0.20
    startup_command_threshold: float = 0.02
    unknown_warmup_s: float = 0.20
    probe_duration_s: float = 0.45
    probe_speed_limit: float = 0.25
    initial_probe_ignore_critical: bool = False
    allow_critical_reprobe: bool = False
    critical_reprobe_s: float = 2.50
    low_reprobe_s: float = 2.50
    probe_relative_clear_drop: float = 0.08
    crawl_pulse_s: float = 0.45
    crawl_min_hold_s: float = 0.25
    launch_accel_rate: float = 1.00
    linear_accel_rate: float = 0.30
    linear_decel_rate: float = 1.00
    yaw_accel_rate: float = 0.80
    yaw_decel_rate: float = 2.00

    def validate(self) -> None:
        limits = (
            self.low_speed_limit,
            self.high_speed_limit,
            self.critical_speed_limit,
            self.low_lateral_limit,
            self.high_lateral_limit,
            self.low_yaw_limit,
            self.high_yaw_limit,
        )
        if any(value < 0.0 for value in limits):
            raise ValueError("command limits must be non-negative")
        if self.high_speed_limit < self.low_speed_limit:
            raise ValueError("high_speed_limit must be >= low_speed_limit")
        if self.high_lateral_limit < self.low_lateral_limit:
            raise ValueError("high_lateral_limit must be >= low_lateral_limit")
        if self.high_yaw_limit < self.low_yaw_limit:
            raise ValueError("high_yaw_limit must be >= low_yaw_limit")
        if not 0.0 <= self.probability_high_enter <= self.probability_low_enter <= 1.0:
            raise ValueError("risk thresholds must satisfy 0 <= high <= low <= 1")
        if not self.probability_low_enter <= self.probability_critical_enter <= 1.0:
            raise ValueError("critical threshold must be >= low-enter threshold")
        if not 0.0 < self.probability_ema_alpha <= 1.0:
            raise ValueError("probability_ema_alpha must be in (0,1]")
        if not 0.0 < self.state_reference_ema_alpha <= 1.0:
            raise ValueError("state_reference_ema_alpha must be in (0,1]")
        if self.reference_settle_s < 0.0:
            raise ValueError("reference_settle_s must be non-negative")
        if not 0.0 < self.reference_settle_alpha <= 1.0:
            raise ValueError("reference_settle_alpha must be in (0,1]")
        prebrake_values = (
            self.prebrake_probability,
            self.prebrake_relative_rise,
        )
        if any(value is None for value in prebrake_values) and any(
            value is not None for value in prebrake_values
        ):
            raise ValueError(
                "prebrake_probability and prebrake_relative_rise must be set together"
            )
        if self.prebrake_probability is not None and not 0.0 <= self.prebrake_probability <= 1.0:
            raise ValueError("prebrake_probability must be in [0,1]")
        if self.prebrake_relative_rise is not None and not 0.0 <= self.prebrake_relative_rise <= 1.0:
            raise ValueError("prebrake_relative_rise must be in [0,1]")
        if self.prebrake_speed_limit is not None and self.prebrake_speed_limit < 0.0:
            raise ValueError("prebrake_speed_limit must be non-negative")
        if not 0.0 <= self.relative_low_rise <= 1.0:
            raise ValueError("relative_low_rise must be in [0,1]")
        if not 0.0 <= self.relative_high_drop <= 1.0:
            raise ValueError("relative_high_drop must be in [0,1]")
        if (
            self.relative_low_min_probability is not None
            and not 0.0 <= self.relative_low_min_probability <= 1.0
        ):
            raise ValueError("relative_low_min_probability must be in [0,1]")
        if not 0.0 <= self.probe_relative_clear_drop <= 1.0:
            raise ValueError("probe_relative_clear_drop must be in [0,1]")
        if self.low_hold_s < 0.0 or self.high_hold_s < 0.0 or self.critical_hold_s < 0.0:
            raise ValueError("hold times must be non-negative")
        if min(
            self.min_detection_command,
            self.startup_command_threshold,
            self.unknown_warmup_s,
            self.probe_duration_s,
            self.probe_speed_limit,
            self.low_reprobe_s,
            self.crawl_pulse_s,
            self.crawl_min_hold_s,
            self.launch_accel_rate,
            self.critical_reprobe_s,
        ) < 0.0:
            raise ValueError("probe settings must be non-negative")


class HallTractionGovernor:
    """Per-environment hysteretic command limiter with asymmetric slew."""

    def __init__(
        self,
        num_envs: int,
        dt: float,
        device: str | torch.device,
        cfg: HallTractionGovernorCfg | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.cfg = cfg or HallTractionGovernorCfg()
        self.cfg.validate()
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.device = torch.device(device)
        self.state = torch.full(
            (num_envs,), UNKNOWN, dtype=torch.int64, device=self.device
        )
        self.low_evidence_s = torch.zeros(num_envs, device=self.device)
        self.high_evidence_s = torch.zeros(num_envs, device=self.device)
        self.critical_evidence_s = torch.zeros(num_envs, device=self.device)
        self.output_command = torch.zeros((num_envs, 3), device=self.device)
        self.unknown_time_s = torch.zeros(num_envs, device=self.device)
        self.low_state_time_s = torch.zeros(num_envs, device=self.device)
        self.probe_time_s = torch.zeros(num_envs, device=self.device)
        self.probe_probability_sum = torch.zeros(num_envs, device=self.device)
        self.probe_probability_count = torch.zeros(num_envs, device=self.device)
        self.probe_start_probability = torch.ones(num_envs, device=self.device)
        self.probing = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.critical_reprobe_active = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )
        self.crawl_cycle_time_s = torch.zeros(num_envs, device=self.device)
        self.probability_ema = torch.ones(num_envs, device=self.device)
        self.probability_ema_initialized = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )
        self.state_probability_reference = torch.ones(num_envs, device=self.device)
        self.state_reference_initialized = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )
        self.state_settle_time_s = torch.zeros(num_envs, device=self.device)
        self.prebrake_active = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.state[env_ids] = UNKNOWN
        self.low_evidence_s[env_ids] = 0.0
        self.high_evidence_s[env_ids] = 0.0
        self.critical_evidence_s[env_ids] = 0.0
        self.output_command[env_ids] = 0.0
        self.unknown_time_s[env_ids] = 0.0
        self.low_state_time_s[env_ids] = 0.0
        self.probe_time_s[env_ids] = 0.0
        self.probe_probability_sum[env_ids] = 0.0
        self.probe_probability_count[env_ids] = 0.0
        self.probe_start_probability[env_ids] = 1.0
        self.probing[env_ids] = False
        self.critical_reprobe_active[env_ids] = False
        self.crawl_cycle_time_s[env_ids] = 0.0
        self.probability_ema[env_ids] = 1.0
        self.probability_ema_initialized[env_ids] = False
        self.state_probability_reference[env_ids] = 1.0
        self.state_reference_initialized[env_ids] = False
        self.state_settle_time_s[env_ids] = 0.0
        self.prebrake_active[env_ids] = False

    def update(
        self,
        requested_command: torch.Tensor,
        low_traction_probability: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if requested_command.shape != (self.num_envs, 3):
            raise ValueError(
                "requested_command must be "
                f"[{self.num_envs},3], got {tuple(requested_command.shape)}"
            )
        probability_raw = low_traction_probability.reshape(self.num_envs).to(
            device=self.device, dtype=torch.float32
        )
        finite = torch.isfinite(probability_raw)
        probability = probability_raw
        probability = torch.nan_to_num(
            probability, nan=1.0, posinf=1.0, neginf=1.0
        ).clamp(0.0, 1.0)
        if valid is not None:
            valid = valid.reshape(self.num_envs).to(self.device).bool()
            finite &= valid
        probability = torch.where(finite, probability, torch.ones_like(probability))
        # At standstill the Hall layer has not yet been mechanically excited,
        # so a learned slip-risk score can be out of distribution for both a
        # safe and an unsafe surface.  An explicitly enabled bounded initial
        # probe lets a *valid* stream acquire that excitation before critical
        # action.  Invalid/stale packets never use this exception and retain
        # immediate fail-safe authority.
        protected_probe = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if self.cfg.initial_probe_ignore_critical:
            protected_probe |= finite & ((self.state == UNKNOWN) | self.probing)
        if self.cfg.allow_critical_reprobe:
            protected_probe |= finite & self.critical_reprobe_active
        critical_sample = (
            probability >= self.cfg.probability_critical_enter
        ) & ~protected_probe
        self.critical_evidence_s = torch.where(
            finite & critical_sample,
            self.critical_evidence_s + self.dt,
            torch.zeros_like(self.critical_evidence_s),
        )
        critical_confirmed = (
            self.critical_evidence_s + 1.0e-6 >= self.cfg.critical_hold_s
        ) & finite
        alpha = self.cfg.probability_ema_alpha
        filtered = torch.where(
            self.probability_ema_initialized,
            (1.0 - alpha) * self.probability_ema + alpha * probability,
            probability,
        )
        self.probability_ema = torch.where(
            finite, filtered, torch.ones_like(filtered)
        )
        self.probability_ema_initialized |= finite
        probability = self.probability_ema

        requested = requested_command.to(self.device)
        requested_motion = (
            torch.abs(requested).amax(dim=1)
            >= self.cfg.startup_command_threshold
        )
        invalid = ~finite
        if torch.any(invalid):
            self.state[invalid] = LOW
            self.probing[invalid] = False
            self.critical_reprobe_active[invalid] = False
            self.unknown_time_s[invalid] = 0.0
            self.low_state_time_s[invalid] = 0.0
            self.prebrake_active[invalid] = False

        # Probe samples form their own guarded experiment.  Do not carry
        # ordinary LOW/HIGH hold evidence through that experiment.
        prior_state = self.state.clone()
        relative_low_min_probability = (
            self.cfg.probability_high_enter
            if self.cfg.relative_low_min_probability is None
            else self.cfg.relative_low_min_probability
        )
        relative_low = (
            (self.state == HIGH)
            & self.state_reference_initialized
            & (
                self.state_settle_time_s + 1.0e-6
                >= self.cfg.reference_settle_s
            )
            & requested_motion
            & (
                probability - self.state_probability_reference
                >= self.cfg.relative_low_rise
            )
            & (probability > relative_low_min_probability)
        )
        relative_high = (
            (self.state == LOW)
            & self.state_reference_initialized
            & requested_motion
            & (
                self.state_probability_reference - probability
                >= self.cfg.relative_high_drop
            )
            & (probability < self.cfg.probability_low_enter)
        )
        # Absolute calibration is used to leave UNKNOWN and as a critical
        # safety override.  Once a bounded probe has established LOW/HIGH,
        # ordinary threshold crossings are not allowed to erase that
        # time-aggregated decision; only sustained change from the state's
        # own reference can switch it.  This is essential for real Hall soles,
        # whose absolute baselines differ across channels and power cycles.
        # UNKNOWN must finish its full active-sensing window.  Ordinary
        # single-frame thresholds are too gait-phase dependent to pre-empt
        # that aggregation.  Only critical evidence may abort early.
        low_absolute = critical_confirmed
        # Most magnetic-foot deployments should use the relative-clear path:
        # channel baselines can shift after power cycles, so an absolute score
        # alone is not generally enough evidence to release speed.  A
        # prospective-slip head can instead be explicitly calibrated on
        # causal slip/fall outcomes and validated under the deployment
        # randomization suite.  In that narrowly opt-in case, a sustained low
        # probability is positive evidence that a prior LOW state recovered.
        # The normal high-hold, packet-validity and critical checks below are
        # still mandatory.
        high_absolute = (
            self.cfg.allow_absolute_high_clear
            & (self.state == LOW)
            & (probability <= self.cfg.probability_high_enter)
        )
        low = (low_absolute | relative_low) & ~self.probing
        high = (high_absolute | relative_high) & ~self.probing
        self.low_evidence_s = torch.where(
            low, self.low_evidence_s + self.dt, torch.zeros_like(self.low_evidence_s)
        )
        self.high_evidence_s = torch.where(
            high,
            self.high_evidence_s + self.dt,
            torch.zeros_like(self.high_evidence_s),
        )
        low_confirmed = (
            self.low_evidence_s + 1.0e-6 >= self.cfg.low_hold_s
        ) & finite
        low_confirmed |= critical_confirmed
        self.state = torch.where(
            low_confirmed,
            torch.full_like(self.state, LOW),
            self.state,
        )
        high_confirmed_from_low = (
            (self.state == LOW)
            & (self.high_evidence_s + 1.0e-6 >= self.cfg.high_hold_s)
            & finite
            & ~self.probing
        )
        self.state = torch.where(
            high_confirmed_from_low,
            torch.full_like(self.state, HIGH),
            self.state,
        )
        self.low_state_time_s = torch.where(
            high_confirmed_from_low,
            torch.zeros_like(self.low_state_time_s),
            self.low_state_time_s,
        )
        changed_state = (self.state != prior_state) & finite
        if torch.any(changed_state):
            self.state_probability_reference[changed_state] = probability[changed_state]
            self.state_reference_initialized[changed_state] = True
            self.state_settle_time_s[changed_state] = 0.0

        # High traction is only accepted after an intentional bounded probe.
        # At the conservative crawl speed, low/high surfaces need not be
        # observable and a static classifier shortcut would be unsafe.
        unknown_waiting = (self.state == UNKNOWN) & requested_motion & ~self.probing
        self.unknown_time_s = torch.where(
            unknown_waiting,
            self.unknown_time_s + self.dt,
            torch.where(
                self.state == UNKNOWN,
                self.unknown_time_s,
                torch.zeros_like(self.unknown_time_s),
            ),
        )
        # Keep a bounded re-probe clock running in LOW even when the learned
        # Hall risk remains moderately above ``probability_low_enter``.  A
        # conservative low-speed gait may not excite the TPU enough for the
        # risk estimate to fall by itself after the robot returns to a
        # high-friction surface.  Only critical risk, invalid Hall data, or a
        # zero motion request suppresses the clock; the probe itself remains
        # speed limited and is aborted immediately by critical evidence.
        critical_reprobe_waiting = (
            self.cfg.allow_critical_reprobe
            & (self.state == LOW)
            & requested_motion
            & ~self.probing
            & finite
            & (probability >= self.cfg.probability_critical_enter)
        )
        low_waiting = (
            (self.state == LOW)
            & requested_motion
            & ~self.probing
            & finite
            & (
                (probability < self.cfg.probability_critical_enter)
                | critical_reprobe_waiting
            )
        )
        self.low_state_time_s = torch.where(
            low_waiting,
            self.low_state_time_s + self.dt,
            torch.zeros_like(self.low_state_time_s),
        )
        ordinary_probe_start = (
            ((self.state == UNKNOWN) & (self.unknown_time_s >= self.cfg.unknown_warmup_s))
            | ((self.state == LOW) & (self.low_state_time_s >= self.cfg.low_reprobe_s))
        )
        critical_probe_start = (
            self.cfg.allow_critical_reprobe
            & (self.state == LOW)
            & (self.low_state_time_s >= self.cfg.critical_reprobe_s)
            & (probability >= self.cfg.probability_critical_enter)
        )
        start_probe = (
            (ordinary_probe_start & (probability < self.cfg.probability_critical_enter))
            | critical_probe_start
        ) & requested_motion & finite & ~self.probing
        if torch.any(start_probe):
            self.probing[start_probe] = True
            self.critical_reprobe_active[start_probe] = critical_probe_start[start_probe]
            self.probe_time_s[start_probe] = 0.0
            self.probe_probability_sum[start_probe] = 0.0
            self.probe_probability_count[start_probe] = 0.0
            self.probe_start_probability[start_probe] = probability[start_probe]
            self.low_evidence_s[start_probe] = 0.0
            self.high_evidence_s[start_probe] = 0.0
            self.unknown_time_s[start_probe] = 0.0
            self.low_state_time_s[start_probe] = 0.0

        # A critical Hall risk or invalid packet aborts a launch immediately.
        abort_probe = self.probing & (
            ~finite
            | (critical_confirmed & ~self.critical_reprobe_active)
        )
        if torch.any(abort_probe):
            self.probing[abort_probe] = False
            self.critical_reprobe_active[abort_probe] = False
            self.state[abort_probe] = LOW
            self.probe_time_s[abort_probe] = 0.0
        active_probe = self.probing & finite
        self.probe_time_s = torch.where(
            active_probe,
            self.probe_time_s + self.dt,
            self.probe_time_s,
        )
        collect_probe = active_probe & (
            self.probe_time_s >= 0.5 * self.cfg.probe_duration_s
        )
        self.probe_probability_sum += torch.where(
            collect_probe, probability, torch.zeros_like(probability)
        )
        self.probe_probability_count += collect_probe.to(torch.float32)
        finish_probe = active_probe & (
            self.probe_time_s + 1.0e-6 >= self.cfg.probe_duration_s
        )
        if torch.any(finish_probe):
            mean_probability = self.probe_probability_sum / self.probe_probability_count.clamp_min(1.0)
            relative_clear = (
                self.probe_start_probability - mean_probability
                >= self.cfg.probe_relative_clear_drop
            )
            high_after_probe = finish_probe & (
                (mean_probability <= self.cfg.probability_high_enter)
                | relative_clear
            )
            self.state[finish_probe] = LOW
            self.state[high_after_probe] = HIGH
            self.state_probability_reference[finish_probe] = mean_probability[
                finish_probe
            ]
            self.state_reference_initialized[finish_probe] = True
            self.probing[finish_probe] = False
            self.critical_reprobe_active[finish_probe] = False
            self.probe_time_s[finish_probe] = 0.0
            self.low_state_time_s[finish_probe] = 0.0
            self.low_evidence_s[finish_probe] = 0.0
            self.high_evidence_s[finish_probe] = 0.0

        # Track a slowly moving, per-environment reference only while the
        # traction state is settled.  A true surface transition changes the
        # risk faster than this reference, so relative evidence survives
        # sensor-to-sensor baseline offsets without converting Hall data to
        # normal/tangential force.  State transitions reset the reference
        # above, preventing stale evidence from immediately flipping back.
        update_reference = (
            finite
            & self.state_reference_initialized
            & (self.state != UNKNOWN)
            & ~self.probing
        )
        # After entering HIGH, learn the normal walking score before using it
        # as the reference for a relative-risk brake.  Without this window a
        # quiet initial probe followed by the first loaded stance looks like a
        # false low-traction jump, which unnecessarily limits high-friction
        # walking.  LOW reference tracking remains deliberately slow.
        settling_high = (
            finite
            & (self.state == HIGH)
            & ~self.probing
            & (self.state_settle_time_s < self.cfg.reference_settle_s)
        )
        reference_alpha = torch.where(
            settling_high,
            probability.new_full(
                probability.shape, self.cfg.reference_settle_alpha
            ),
            probability.new_full(
                probability.shape, self.cfg.state_reference_ema_alpha
            ),
        )
        reference_update = (
            (1.0 - reference_alpha) * self.state_probability_reference
            + reference_alpha * probability
        )
        self.state_probability_reference = torch.where(
            update_reference,
            reference_update,
            self.state_probability_reference,
        )
        settled_high = finite & (self.state == HIGH) & ~self.probing
        self.state_settle_time_s = torch.where(
            settled_high,
            self.state_settle_time_s + self.dt,
            torch.zeros_like(self.state_settle_time_s),
        )

        high_state = self.state == HIGH
        limits_low = torch.tensor(
            [
                self.cfg.low_speed_limit,
                self.cfg.low_lateral_limit,
                self.cfg.low_yaw_limit,
            ],
            device=self.device,
        )
        limits_high = torch.tensor(
            [
                self.cfg.high_speed_limit,
                self.cfg.high_lateral_limit,
                self.cfg.high_yaw_limit,
            ],
            device=self.device,
        )
        limits = torch.where(high_state[:, None], limits_high, limits_low)
        # A structural Hall/proprio risk jump may precede the normal LOW
        # evidence hold by one or two policy frames.  Apply only an envelope
        # brake here; state changes, critical handling and later release stay
        # governed by the ordinary hysteretic logic above.  The gate requires
        # both an absolute calibrated score and a rise from this foot's own
        # settled walking reference, so it cannot become a static friction
        # proxy or a zero-load startup shortcut.
        prebrake_enabled = self.cfg.prebrake_probability is not None
        if prebrake_enabled:
            assert self.cfg.prebrake_relative_rise is not None
            prebrake = (
                finite
                & high_state
                & self.state_reference_initialized
                & (
                    self.state_settle_time_s + 1.0e-6
                    >= self.cfg.reference_settle_s
                )
                & requested_motion
                & (probability >= self.cfg.prebrake_probability)
                & (
                    probability - self.state_probability_reference
                    >= self.cfg.prebrake_relative_rise
                )
            )
            self.prebrake_active = prebrake
            prebrake_limits = limits_low.clone()
            prebrake_limits[0] = (
                self.cfg.low_speed_limit
                if self.cfg.prebrake_speed_limit is None
                else self.cfg.prebrake_speed_limit
            )
            limits = torch.where(prebrake[:, None], prebrake_limits, limits)
        else:
            self.prebrake_active.zero_()
        critical_limits = limits_low.clone()
        critical_limits[0] = self.cfg.critical_speed_limit
        critical_limits[1:] = 0.0
        critical = critical_confirmed | ~finite
        # A finite, explicitly enabled critical re-probe is a bounded active
        # sensing maneuver, not a release of the critical state.  Its result
        # decides HIGH versus LOW below; invalid data always remains fail-stop.
        critical &= ~(self.probing & self.critical_reprobe_active)
        limits = torch.where(critical[:, None], critical_limits, limits)
        if torch.any(self.probing):
            probe_limits = limits_low.clone()
            probe_limits[0] = self.cfg.probe_speed_limit
            limits = torch.where(self.probing[:, None], probe_limits, limits)
        target = requested.clamp(-limits, limits)

        # The legacy walking actor has a low-command dead zone.  During the
        # single bounded probe, lift a non-zero forward request to the minimum
        # launch speed.  Zero commands stay exactly zero.
        probe_forward = self.probing & (
            torch.abs(requested[:, 0]) >= self.cfg.startup_command_threshold
        )
        target[:, 0] = torch.where(
            probe_forward,
            torch.sign(requested[:, 0]) * self.cfg.probe_speed_limit,
            target[:, 0],
        )

        # Realize commands below the actor's launch threshold by short
        # micro-step pulses after the first traction decision.  LOW is included
        # because weak TPU excitation is otherwise self-locking: the actor does
        # not step, so Hall/proprio history never gains enough information to
        # clear a conservative state.  Duty cycle preserves the requested
        # long-term mean; zero, critical risk and invalid Hall remain zero.
        crawl = (
            (self.state != UNKNOWN)
            & ~self.probing
            & ~critical
            & (torch.abs(requested[:, 0]) >= self.cfg.startup_command_threshold)
            & (torch.abs(requested[:, 0]) < self.cfg.min_detection_command)
        )
        requested_abs = torch.abs(requested[:, 0]).clamp_min(
            self.cfg.startup_command_threshold
        )
        crawl_period = torch.maximum(
            self.cfg.crawl_pulse_s
            * self.cfg.probe_speed_limit
            / requested_abs,
            requested_abs.new_full(
                requested_abs.shape,
                self.cfg.crawl_pulse_s + self.cfg.crawl_min_hold_s,
            ),
        )
        self.crawl_cycle_time_s = torch.where(
            crawl,
            torch.remainder(self.crawl_cycle_time_s + self.dt, crawl_period),
            torch.zeros_like(self.crawl_cycle_time_s),
        )
        crawl_active = crawl & (
            self.crawl_cycle_time_s <= self.cfg.crawl_pulse_s
        )
        target[:, 0] = torch.where(
            crawl_active,
            torch.sign(requested[:, 0]) * self.cfg.probe_speed_limit,
            torch.where(crawl, torch.zeros_like(target[:, 0]), target[:, 0]),
        )
        # Critical risk has final authority over both probes and crawl pulses.
        target = torch.where(critical[:, None], torch.zeros_like(target), target)

        accel = torch.tensor(
            [
                self.cfg.linear_accel_rate,
                self.cfg.linear_accel_rate,
                self.cfg.yaw_accel_rate,
            ],
            device=self.device,
        )
        decel = torch.tensor(
            [
                self.cfg.linear_decel_rate,
                self.cfg.linear_decel_rate,
                self.cfg.yaw_decel_rate,
            ],
            device=self.device,
        )
        accelerating = (
            torch.sign(target) == torch.sign(self.output_command)
        ) & (torch.abs(target) > torch.abs(self.output_command))
        rate = torch.where(accelerating, accel, decel)
        launching = self.probing | crawl_active
        rate[:, 0] = torch.where(
            launching,
            torch.maximum(
                rate[:, 0],
                rate[:, 0].new_full(
                    rate[:, 0].shape, self.cfg.launch_accel_rate
                ),
            ),
            rate[:, 0],
        )
        delta = (target - self.output_command).clamp(-rate * self.dt, rate * self.dt)
        self.output_command = self.output_command + delta
        return self.output_command.clone(), self.state.clone()
