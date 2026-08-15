"""RSL-RL model adapters for canonical traction observation groups."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel

from .networks import (
    PrivilegedTractionEncoder,
    PrivilegedTractionEncoderCfg,
    TemporalStudentEncoderCfg,
    TemporalTactileProprioceptiveStudentEncoder,
    temporal_history_to_legacy_proprio,
)
from .schema import PRIVILEGED_TRACTION_SCHEMA, TEMPORAL_STUDENT_FRAME_SCHEMA


TEACHER_FRAME_DIM = 96 + 3 + PRIVILEGED_TRACTION_SCHEMA.flat_dimension
TEACHER_HISTORY_FRAMES = 5
TEACHER_FLAT_DIM = TEACHER_FRAME_DIM * TEACHER_HISTORY_FRAMES


def _drop_deprecated_model_kwargs(kwargs: dict) -> None:
    for name in (
        "stochastic",
        "init_noise_std",
        "noise_std_type",
        "state_dependent_std",
        "actor_obs_normalization",
        "critic_obs_normalization",
    ):
        kwargs.pop(name, None)


def teacher_history_to_legacy_observation(
    history: torch.Tensor,
    *,
    include_base_linear_velocity: bool,
) -> torch.Tensor:
    """Recover the audited term-major baseline observation from Teacher history.

    Each canonical Teacher frame is ``96-D proprio + 3-D adjusted command +
    135-D privilege``.  Isaac's single-term history is time-major.  The
    pretrained locomotion checkpoint instead consumes a five-frame term-major
    vector, so this conversion is required for an exact warm start rather than
    a prefix assumption.
    """
    if (
        history.ndim < 3
        or history.shape[-2] != TEACHER_HISTORY_FRAMES
        or history.shape[-1] != TEACHER_FRAME_DIM
    ):
        raise ValueError(
            "Teacher history must end in "
            f"[{TEACHER_HISTORY_FRAMES},{TEACHER_FRAME_DIM}], "
            f"got {tuple(history.shape)}"
        )
    proprio = history[..., :96]
    actor = torch.cat(
        (
            proprio[..., 0:3].flatten(-2),
            proprio[..., 3:6].flatten(-2),
            proprio[..., 6:9].flatten(-2),
            proprio[..., 9:38].flatten(-2),
            proprio[..., 38:67].flatten(-2),
            proprio[..., 67:96].flatten(-2),
        ),
        dim=-1,
    )
    if not include_base_linear_velocity:
        return actor
    base_velocity_slice = PRIVILEGED_TRACTION_SCHEMA.term_slice(
        "base_linear_velocity"
    )
    privileged_offset = 99
    base_linear_velocity = history[
        ...,
        privileged_offset + base_velocity_slice.start :
        privileged_offset + base_velocity_slice.stop,
    ].flatten(-2)
    return torch.cat((base_linear_velocity, actor), dim=-1)


class TractionTeacherRslModel(MLPModel):
    """Baseline-preserving PPO Teacher with privileged traction residual input."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        latent_dim: int = 16,
        **kwargs,
    ) -> None:
        _drop_deprecated_model_kwargs(kwargs)
        self.traction_latent_dim = latent_dim
        self.legacy_observation_dim = 480
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        if self.obs_dim != TEACHER_FLAT_DIM:
            raise ValueError(
                f"Teacher RSL observation has {self.obs_dim} elements, "
                f"expected {TEACHER_FLAT_DIM}"
            )
        self.traction_encoder = PrivilegedTractionEncoder(
            PrivilegedTractionEncoderCfg(
                input_dim=PRIVILEGED_TRACTION_SCHEMA.flat_dimension,
                latent_dim=latent_dim,
            )
        )
        self.latest_traction_latent: torch.Tensor | None = None

    def _get_latent_dim(self) -> int:
        return self.legacy_observation_dim + self.traction_latent_dim

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        raw = super().get_latent(obs, masks, hidden_state)
        history = raw.reshape(
            *raw.shape[:-1], TEACHER_HISTORY_FRAMES, TEACHER_FRAME_DIM
        )
        baseline = teacher_history_to_legacy_observation(
            history,
            include_base_linear_velocity=False,
        )
        privileged = history[..., -1, 99:]
        traction_latent = self.traction_encoder(privileged)
        self.latest_traction_latent = traction_latent
        return torch.cat((baseline, traction_latent), dim=-1)

    def as_jit(self) -> nn.Module:
        return _TorchTeacherModel(self, include_base_linear_velocity=False)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        del verbose
        return _TorchTeacherModel(self, include_base_linear_velocity=False)


class TractionTeacherCriticRslModel(MLPModel):
    """Baseline-preserving value model using critic history plus Teacher latent."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        latent_dim: int = 16,
        **kwargs,
    ) -> None:
        _drop_deprecated_model_kwargs(kwargs)
        self.traction_latent_dim = latent_dim
        self.legacy_observation_dim = 495
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        if self.obs_dim != TEACHER_FLAT_DIM:
            raise ValueError(
                f"Teacher critic observation has {self.obs_dim} elements, "
                f"expected {TEACHER_FLAT_DIM}"
            )
        self.traction_encoder = PrivilegedTractionEncoder(
            PrivilegedTractionEncoderCfg(
                input_dim=PRIVILEGED_TRACTION_SCHEMA.flat_dimension,
                latent_dim=latent_dim,
            )
        )
        self.latest_traction_latent: torch.Tensor | None = None

    def _get_latent_dim(self) -> int:
        return self.legacy_observation_dim + self.traction_latent_dim

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        raw = super().get_latent(obs, masks, hidden_state)
        history = raw.reshape(
            *raw.shape[:-1], TEACHER_HISTORY_FRAMES, TEACHER_FRAME_DIM
        )
        baseline = teacher_history_to_legacy_observation(
            history,
            include_base_linear_velocity=True,
        )
        privileged = history[..., -1, 99:]
        traction_latent = self.traction_encoder(privileged)
        self.latest_traction_latent = traction_latent
        return torch.cat((baseline, traction_latent), dim=-1)

    def as_jit(self) -> nn.Module:
        return _TorchTeacherModel(self, include_base_linear_velocity=True)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        del verbose
        return _TorchTeacherModel(self, include_base_linear_velocity=True)


class CanonicalStudentRslModel(MLPModel):
    """PPO-trainable fixed-window Student actor.

    The RSL model consumes exactly one flattened 15×106 observation group. Its
    MLP head sees the audited newest-five-frame 480-D proprioception, the
    learned 16-D traction latent, and the latest 3-D raw command.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        latent_dim: int = 16,
        temporal_variant: str = "gru",
        **kwargs,
    ) -> None:
        _drop_deprecated_model_kwargs(kwargs)
        self.traction_latent_dim = latent_dim
        self.student_head_input_dim = 480 + latent_dim + 3
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        if self.obs_dim != TEMPORAL_STUDENT_FRAME_SCHEMA.flat_dimension:
            raise ValueError(
                f"Student RSL observation has {self.obs_dim} elements, expected "
                f"{TEMPORAL_STUDENT_FRAME_SCHEMA.flat_dimension}"
            )
        self.student_encoder = TemporalTactileProprioceptiveStudentEncoder(
            TemporalStudentEncoderCfg(
                latent_dim=latent_dim,
                variant=temporal_variant,
            )
        )
        self.latest_slip_probability: torch.Tensor | None = None
        self.latest_traction_score: torch.Tensor | None = None
        self.latest_sensor_confidence: torch.Tensor | None = None
        self.latest_traction_latent: torch.Tensor | None = None

    def _get_latent_dim(self) -> int:
        return self.student_head_input_dim

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        flat = super().get_latent(obs, masks, hidden_state)
        leading_shape = flat.shape[:-1]
        history = flat.reshape(
            -1,
            TEMPORAL_STUDENT_FRAME_SCHEMA.history_frames,
            TEMPORAL_STUDENT_FRAME_SCHEMA.frame_dimension,
        )
        encoded = self.student_encoder(history)
        baseline = temporal_history_to_legacy_proprio(history)
        latest_command = history[:, -1, 93:96]
        self.latest_traction_latent = encoded.latent
        self.latest_slip_probability = encoded.slip_probability
        self.latest_traction_score = encoded.traction_score
        self.latest_sensor_confidence = encoded.sensor_confidence
        latent = torch.cat((baseline, encoded.latent, latest_command), dim=-1)
        return latent.reshape(*leading_shape, latent.shape[-1])

    def as_jit(self) -> nn.Module:
        return _TorchStudentRslModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        del verbose
        return _TorchStudentRslModel(self)


class _TorchTeacherModel(nn.Module):
    def __init__(
        self,
        model: TractionTeacherRslModel | TractionTeacherCriticRslModel,
        *,
        include_base_linear_velocity: bool,
    ) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder = copy.deepcopy(model.traction_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        self.include_base_linear_velocity = include_base_linear_velocity
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else nn.Identity()
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        observation = self.obs_normalizer(observation)
        history = observation.reshape(
            *observation.shape[:-1], TEACHER_HISTORY_FRAMES, TEACHER_FRAME_DIM
        )
        baseline = teacher_history_to_legacy_observation(
            history,
            include_base_linear_velocity=self.include_base_linear_velocity,
        )
        latent = self.encoder(history[..., -1, 99:])
        output = self.mlp(torch.cat((baseline, latent), dim=-1))
        return self.deterministic_output(output)


class _TorchStudentRslModel(nn.Module):
    def __init__(self, model: CanonicalStudentRslModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder = copy.deepcopy(model.student_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else nn.Identity()
        )

    def forward(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        observation = self.obs_normalizer(observation)
        history = observation.reshape(
            -1,
            TEMPORAL_STUDENT_FRAME_SCHEMA.history_frames,
            TEMPORAL_STUDENT_FRAME_SCHEMA.frame_dimension,
        )
        encoded = self.encoder(history)
        baseline = temporal_history_to_legacy_proprio(history)
        latest_command = history[:, -1, 93:96]
        action = self.deterministic_output(
            self.mlp(torch.cat((baseline, encoded.latent, latest_command), dim=-1))
        )
        return (
            action,
            encoded.slip_probability,
            encoded.traction_score,
            encoded.sensor_confidence,
        )
