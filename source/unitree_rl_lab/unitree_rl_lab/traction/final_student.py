"""Final deployable Hall-force + motor-effort traction Student.

The privileged Teacher is deliberately absent from this module.  The Student
accepts only signals available on the real G1: the audited 480-D proprioceptive
history, 15 frames of motor-estimated effort, calibrated dual-foot three-axis
force, and per-foot sensor health metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from torch import nn

from .networks import LegacyLocomotionActor
from .schema import FORCE_FRAME, FORCE_ORDER, G1_29DOF_JOINT_ORDER


BASE_PROPRIO_DIM = 480
HISTORY_FRAMES = 15
JOINT_EFFORT_DIM = 29
FOOT_FORCE_DIM = 6
HEALTH_DIM = 6
FINAL_STUDENT_INPUT_DIM = (
    BASE_PROPRIO_DIM
    + HISTORY_FRAMES * JOINT_EFFORT_DIM
    + HISTORY_FRAMES * FOOT_FORCE_DIM
    + HEALTH_DIM
)
ACTION_DIM = 29

BASE_PROPRIO_SLICE = slice(0, BASE_PROPRIO_DIM)
JOINT_EFFORT_SLICE = slice(
    BASE_PROPRIO_SLICE.stop,
    BASE_PROPRIO_SLICE.stop + HISTORY_FRAMES * JOINT_EFFORT_DIM,
)
FOOT_FORCE_SLICE = slice(
    JOINT_EFFORT_SLICE.stop,
    JOINT_EFFORT_SLICE.stop + HISTORY_FRAMES * FOOT_FORCE_DIM,
)
FOOT_VALID_SLICE = slice(FOOT_FORCE_SLICE.stop, FOOT_FORCE_SLICE.stop + 2)
FOOT_AGE_SLICE = slice(FOOT_VALID_SLICE.stop, FOOT_VALID_SLICE.stop + 2)
FOOT_PERIOD_SLICE = slice(FOOT_AGE_SLICE.stop, FOOT_AGE_SLICE.stop + 2)

assert FINAL_STUDENT_INPUT_DIM == 1011
assert FOOT_PERIOD_SLICE.stop == FINAL_STUDENT_INPUT_DIM


@dataclass(frozen=True)
class FinalStudentSchema:
    schema_version: str = "g1_hall_force_tau_student_v1"
    input_dimension: int = FINAL_STUDENT_INPUT_DIM
    action_dimension: int = ACTION_DIM
    history_frames: int = HISTORY_FRAMES
    policy_rate_hz: float = 50.0
    force_frame: str = FORCE_FRAME
    force_order: tuple[str, ...] = FORCE_ORDER
    force_scale: str = "F_local_N / (robot_mass_kg * 9.81)"
    force_clip: tuple[float, float] = (-2.0, 2.0)
    joint_effort_scale: float = 0.02
    joint_effort_clip_nm: tuple[float, float] = (-100.0, 100.0)
    joint_order: tuple[str, ...] = G1_29DOF_JOINT_ORDER
    flatten_order: str = (
        "legacy_480_term_major, tau_15x29_time_major, "
        "force_15x6_time_major, valid_lr, age_s_lr, period_s_lr"
    )

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["slices"] = {
            "legacy_proprio": [BASE_PROPRIO_SLICE.start, BASE_PROPRIO_SLICE.stop],
            "joint_effort_history": [JOINT_EFFORT_SLICE.start, JOINT_EFFORT_SLICE.stop],
            "foot_force_history": [FOOT_FORCE_SLICE.start, FOOT_FORCE_SLICE.stop],
            "foot_valid_lr": [FOOT_VALID_SLICE.start, FOOT_VALID_SLICE.stop],
            "foot_age_s_lr": [FOOT_AGE_SLICE.start, FOOT_AGE_SLICE.stop],
            "foot_period_s_lr": [FOOT_PERIOD_SLICE.start, FOOT_PERIOD_SLICE.stop],
        }
        result["privileged_fields_forbidden"] = [
            "ground_friction_mu",
            "ideal_contact_force",
            "terrain_label",
            "future_friction",
            "simulator_contact_or_slip_truth",
        ]
        return result

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


FINAL_STUDENT_SCHEMA = FinalStudentSchema()


class FinalHallForceStudent(nn.Module):
    """Baseline-preserving temporal Student with bounded action adaptation."""

    def __init__(
        self,
        *,
        signal_mean: torch.Tensor | None = None,
        signal_scale: torch.Tensor | None = None,
        residual_limit: float = 1.0,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if residual_limit <= 0.0:
            raise ValueError("residual_limit must be positive")
        self.baseline_actor = LegacyLocomotionActor(BASE_PROPRIO_DIM)
        self.signal_encoder = nn.GRU(
            JOINT_EFFORT_DIM + FOOT_FORCE_DIM,
            hidden_dim,
            batch_first=True,
        )
        # The newest health tuple is kept outside the recurrent stream so a
        # stale packet causes an immediate, causal confidence drop.
        self.fusion = nn.Sequential(
            nn.Linear(128 + hidden_dim + HEALTH_DIM, 192),
            nn.ELU(),
            nn.Linear(192, 128),
            nn.ELU(),
        )
        self.action_residual = nn.Linear(128, ACTION_DIM)
        self.friction_head = nn.Linear(128, 1)
        self.slip_head = nn.Linear(128, 2)
        self.learned_confidence_head = nn.Linear(128, 1)
        self.register_buffer(
            "signal_mean",
            torch.zeros(JOINT_EFFORT_DIM + FOOT_FORCE_DIM)
            if signal_mean is None
            else torch.as_tensor(signal_mean, dtype=torch.float32).clone(),
        )
        self.register_buffer(
            "signal_scale",
            torch.ones(JOINT_EFFORT_DIM + FOOT_FORCE_DIM)
            if signal_scale is None
            else torch.as_tensor(signal_scale, dtype=torch.float32).clone(),
        )
        self.register_buffer(
            "residual_limit",
            torch.tensor(float(residual_limit), dtype=torch.float32),
        )
        nn.init.zeros_(self.action_residual.weight)
        nn.init.zeros_(self.action_residual.bias)

    def load_baseline_checkpoint(self, checkpoint: dict[str, object]) -> None:
        state = checkpoint.get("actor_state_dict")
        if not isinstance(state, dict):
            raise ValueError("baseline checkpoint has no actor_state_dict")
        actor_state = {
            key: value for key, value in state.items() if key.startswith("mlp.")
        }
        self.baseline_actor.load_state_dict(actor_state, strict=True)

    def _split(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[-1] != FINAL_STUDENT_INPUT_DIM:
            raise ValueError(
                "Final Student observation must be "
                f"[batch,{FINAL_STUDENT_INPUT_DIM}], got {tuple(observation.shape)}"
            )
        observation = torch.nan_to_num(observation)
        baseline = observation[:, BASE_PROPRIO_SLICE]
        torque = observation[:, JOINT_EFFORT_SLICE].reshape(
            -1, HISTORY_FRAMES, JOINT_EFFORT_DIM
        )
        force = observation[:, FOOT_FORCE_SLICE].reshape(
            -1, HISTORY_FRAMES, FOOT_FORCE_DIM
        )
        health = observation[:, FOOT_VALID_SLICE.start : FOOT_PERIOD_SLICE.stop]
        signals = torch.cat((torque, force), dim=-1)
        signals = (signals - self.signal_mean) / self.signal_scale.clamp_min(1.0e-4)
        return baseline, signals, health

    def forward(
        self, observation: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        baseline, signals, health = self._split(observation)
        baseline_latent = self.baseline_actor.mlp[:6](baseline)
        baseline_action = self.baseline_actor.mlp[6](baseline_latent)
        _, hidden = self.signal_encoder(signals)
        latent = self.fusion(torch.cat((baseline_latent, hidden[-1], health), dim=-1))
        residual = self.residual_limit * torch.tanh(self.action_residual(latent))
        action = baseline_action + residual
        estimated_mu = 1.30 * torch.sigmoid(self.friction_head(latent))
        slip_probability = torch.sigmoid(self.slip_head(latent))
        valid = health[:, 0:2].clamp(0.0, 1.0)
        age = health[:, 2:4].clamp_min(0.0)
        physical_confidence = valid.mean(dim=-1, keepdim=True) * torch.exp(
            -age.max(dim=-1, keepdim=True).values / 0.10
        )
        sensor_confidence = (
            torch.sigmoid(self.learned_confidence_head(latent)) * physical_confidence
        )
        return (
            action,
            estimated_mu,
            slip_probability,
            sensor_confidence,
            residual,
        )

