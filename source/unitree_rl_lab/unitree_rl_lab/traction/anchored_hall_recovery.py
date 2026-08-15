"""Bounded Hall-conditioned recovery residual around the audited G1 actor.

This module is intentionally narrower than a replacement locomotion policy:
the original 480-D proprioceptive actor remains the action baseline.  A small
residual can be learned from Hall ``Bx/By/Bz`` histories and proprioception,
but it has authority only when a *deployable* Hall risk estimate is high and
both feet have healthy packets.  There is no Hall-to-force, Hall-to-contact,
or Hall-to-friction inverse anywhere in this path.

The design makes the safety contract explicit:

* nominal/high-traction state: exact original actor action;
* missing, stale, NaN Hall stream: exact original actor action;
* confirmed risk with healthy Hall: bounded residual only.

The command governor remains responsible for command limiting.  This module
only supplies a smoother low-traction posture/step correction when that
governor asks for it.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .hall_risk_estimator import build_hall_risk_estimator
from .layout_magnetic_student import ACTION_DIM, INPUT_DIM
from .networks import LegacyLocomotionActor


class AnchoredHallRecoveryPolicy(nn.Module):
    """Original G1 actor plus a strictly bounded, healthy-Hall residual."""

    def __init__(
        self,
        baseline_actor: LegacyLocomotionActor,
        risk_estimator: nn.Module,
        *,
        correction_limit: float = 0.12,
        risk_gate_start: float = 0.45,
        risk_gate_full: float = 0.75,
    ) -> None:
        super().__init__()
        if correction_limit <= 0.0:
            raise ValueError("correction_limit must be positive")
        if not 0.0 <= risk_gate_start < risk_gate_full <= 1.0:
            raise ValueError("risk gate must satisfy 0 <= start < full <= 1")
        self.baseline_actor = baseline_actor
        self.risk_estimator = risk_estimator
        feature_dim = int(getattr(risk_estimator, "feature_dim", 64))
        self.recovery_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ELU(),
            nn.Linear(128, ACTION_DIM),
        )
        self.register_buffer(
            "correction_limit", torch.tensor(float(correction_limit), dtype=torch.float32)
        )
        self.register_buffer(
            "risk_gate_start", torch.tensor(float(risk_gate_start), dtype=torch.float32)
        )
        self.register_buffer(
            "risk_gate_full", torch.tensor(float(risk_gate_full), dtype=torch.float32)
        )
        # Start as an exact original-actor policy.  Training is the only way
        # a correction can gain authority.
        nn.init.zeros_(self.recovery_head[-1].weight)
        nn.init.zeros_(self.recovery_head[-1].bias)
        self.freeze_upstream()

    def freeze_upstream(self) -> None:
        """Keep the audited actor and risk estimator immutable during fitting."""

        for module in (self.baseline_actor, self.risk_estimator):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "AnchoredHallRecoveryPolicy":
        super().train(mode)
        self.baseline_actor.eval()
        self.risk_estimator.eval()
        return self

    def recovery_outputs(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``action, original_action, risk, bounded_correction``.

        ``risk`` is a Hall/proprioceptive risk probability.  It is not a force
        estimate and is used only to gate the learned residual.
        """

        if observation.ndim != 2 or observation.shape[1] != INPUT_DIM:
            raise ValueError(
                f"expected [B,{INPUT_DIM}], got {tuple(observation.shape)}"
            )
        # The original actor is robust to non-finite deployable packet fields
        # because it consumes only the proprioceptive 480-D prefix.  Replace
        # non-finite prefix values deterministically so the exported ONNX
        # recovery graph always remains finite under malformed packets.
        baseline_input = torch.nan_to_num(observation[:, :480])
        base_action = self.baseline_actor(baseline_input)
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
        # A failed Hall stream has zero residual authority.  The external
        # governor is independently conservative in that same condition.
        gate = confidence * gate
        raw_correction = self.correction_limit * torch.tanh(self.recovery_head(latent))
        correction = gate * raw_correction
        return base_action + correction, base_action, risk, correction

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.recovery_outputs(observation)[0]


def load_baseline_actor(checkpoint_path: Path) -> LegacyLocomotionActor:
    """Load the audited 480-D original actor from ``model_49999.pt``."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("actor_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"{checkpoint_path}: missing actor_state_dict")
    actor = LegacyLocomotionActor(480)
    actor.load_state_dict(
        {key: value for key, value in state.items() if key.startswith("mlp.")},
        strict=True,
    )
    return actor.eval()


def build_anchored_hall_recovery(payload: dict[str, object]) -> AnchoredHallRecoveryPolicy:
    """Reconstruct an anchored recovery checkpoint for replay/inspection."""

    if payload.get("policy_type") != "anchored_hall_recovery_policy":
        raise ValueError("not an anchored Hall recovery checkpoint")
    risk_payload = payload.get("risk_payload")
    if not isinstance(risk_payload, dict):
        raise ValueError("anchored recovery checkpoint has no embedded risk payload")
    baseline_state = payload.get("baseline_actor_state")
    if not isinstance(baseline_state, dict):
        raise ValueError("anchored recovery checkpoint has no baseline actor state")
    actor = LegacyLocomotionActor(480)
    actor.load_state_dict(baseline_state, strict=True)
    risk = build_hall_risk_estimator(risk_payload)
    policy = AnchoredHallRecoveryPolicy(
        actor,
        risk,
        correction_limit=float(payload.get("correction_limit", 0.12)),
        risk_gate_start=float(payload.get("risk_gate_start", 0.45)),
        risk_gate_full=float(payload.get("risk_gate_full", 0.75)),
    )
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("anchored recovery checkpoint has no model state")
    policy.load_state_dict(state, strict=True)
    return policy
