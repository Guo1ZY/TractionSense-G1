"""Risk-gated Hall recovery residual for the magnetic-foot locomotion policy.

The normal Hall locomotion Student and the independent Hall risk estimator are
frozen.  Only a small bounded residual is learned from privileged Teacher
actions offline.  Deployment input remains Hall Bx/By/Bz history plus
proprioception; contact force and friction truth never cross this interface.
"""

from __future__ import annotations

import torch
from torch import nn

from .layout_magnetic_student import (
    ACTION_DIM,
    ACTION_OUTPUT_LIMIT,
    LayoutMagneticStudent,
    normalize_trailing_feature_mode,
)


class HallRecoveryPolicy(nn.Module):
    """Add a confidence- and risk-gated correction to a frozen Hall policy."""

    def __init__(
        self,
        base_policy: LayoutMagneticStudent,
        risk_estimator: nn.Module,
        correction_limit: float = 0.25,
        risk_gate_start: float = 0.35,
        risk_gate_full: float = 0.75,
    ) -> None:
        super().__init__()
        if correction_limit <= 0.0:
            raise ValueError("correction_limit must be positive")
        if not 0.0 <= risk_gate_start < risk_gate_full <= 1.0:
            raise ValueError("risk gate must satisfy 0 <= start < full <= 1")
        base_trailing_mode = normalize_trailing_feature_mode(
            base_policy.trailing_feature_mode
        )
        risk_trailing_mode = normalize_trailing_feature_mode(
            str(getattr(risk_estimator, "trailing_feature_mode", "sensor_age"))
        )
        if base_trailing_mode != risk_trailing_mode:
            raise ValueError(
                "base policy and risk estimator disagree on channels 1862:1864: "
                f"{base_trailing_mode!r} != {risk_trailing_mode!r}"
            )
        self.base_policy = base_policy
        self.risk_estimator = risk_estimator
        feature_dim = int(getattr(risk_estimator, "feature_dim", 64))
        self.recovery_head = nn.Sequential(
            nn.Linear(feature_dim, 96),
            nn.ELU(),
            nn.Linear(96, ACTION_DIM),
        )
        self.register_buffer(
            "correction_limit",
            torch.tensor(float(correction_limit), dtype=torch.float32),
        )
        self.register_buffer(
            "risk_gate_start",
            torch.tensor(float(risk_gate_start), dtype=torch.float32),
        )
        self.register_buffer(
            "risk_gate_full",
            torch.tensor(float(risk_gate_full), dtype=torch.float32),
        )
        nn.init.zeros_(self.recovery_head[-1].weight)
        nn.init.zeros_(self.recovery_head[-1].bias)
        self.freeze_upstream()

    def freeze_upstream(self) -> None:
        """Keep the already validated gait and risk networks immutable."""
        for module in (self.base_policy, self.risk_estimator):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "HallRecoveryPolicy":
        super().train(mode)
        # No BatchNorm/dropout is currently used, but enforcing eval here
        # makes the frozen-upstream contract robust to future architecture edits.
        self.base_policy.eval()
        self.risk_estimator.eval()
        return self

    def recovery_outputs(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base_action = self.base_policy(observation)
        latent, health = self.risk_estimator.features(observation)
        risk = self.risk_estimator(observation)
        confidence = self.risk_estimator.physical_confidence(
            health,
            getattr(self.risk_estimator, "trailing_feature_mode", "sensor_age"),
        )
        gate = torch.clamp(
            (risk - self.risk_gate_start)
            / (self.risk_gate_full - self.risk_gate_start),
            0.0,
            1.0,
        )
        # Invalid/stale Hall means no learned action authority.  The separate
        # command governor still treats it as maximum risk and slows down.
        gate = confidence * gate
        raw_correction = self.correction_limit * torch.tanh(
            self.recovery_head(latent)
        )
        correction = gate * raw_correction
        action = torch.clamp(
            base_action + correction,
            -ACTION_OUTPUT_LIMIT,
            ACTION_OUTPUT_LIMIT,
        )
        return action, base_action, risk, correction

    def all_outputs(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action, _, _, correction = self.recovery_outputs(observation)
        _, estimated_mu, slip_probability, confidence, base_residual = (
            self.base_policy.all_outputs(observation)
        )
        return (
            action,
            estimated_mu,
            slip_probability,
            confidence,
            base_residual + correction,
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.recovery_outputs(observation)[0]
