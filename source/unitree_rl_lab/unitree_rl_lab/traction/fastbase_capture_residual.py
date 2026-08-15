"""Hall-gated capture residual on top of the frozen speedboost112 actor.

The fast actor is immutable.  Only the bounded residual and its Hall/proprio
gate are trainable, which prevents low-grip training from erasing the proven
high-grip gait.  Exact friction/contact labels may supervise the gate during
training, but are deliberately absent from :meth:`forward` and the exported
1864-D deployment graph.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel

from .frozen_speedboost_teacher import (
    INPUT_DIM,
    OUTPUT_DIM,
    TRAILING_SLICE,
    VALID_SLICE,
    FrozenSpeedBoostTeacher,
    load_frozen_speedboost_teacher,
)


TeacherTrailingMode = Literal["passthrough", "assume_fresh"]


# Canonical term-major slices in the audited 1864-D Motion observation.  The
# first 480 values are five causal frames of proprioception.  The final two
# values are deployable lateral velocity and reset-relative heading feedback;
# no force, contact, material or simulator-only stage enters this branch.
LATEST_BASE_ANG_VEL_SLICE = slice(12, 15)
LATEST_PROJECTED_GRAVITY_SLICE = slice(27, 30)
COMMAND_YAW_INDICES = (32, 35, 38, 41, 44)
MOTION_FEEDBACK_SLICE = slice(1862, 1864)
BASE_ANG_VEL_OBSERVATION_SCALE = 0.2


class RslActorMean(nn.Module):
    """Deterministic mean network used by native 1864-D RSL checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, OUTPUT_DIM),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != INPUT_DIM:
            raise ValueError(f"expected [batch,{INPUT_DIM}], got {tuple(observation.shape)}")
        return self.mlp(observation)

    def load_rsl_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        actor_state = {name: value for name, value in state_dict.items() if name.startswith("mlp.")}
        self.load_state_dict(actor_state, strict=True)


class FastBaseHallCaptureResidual(nn.Module):
    """Frozen fast base plus a learned, smooth and bounded capture correction."""

    def __init__(
        self,
        teacher: FrozenSpeedBoostTeacher,
        *,
        residual_limit: float = 1.25,
        gate_power: float = 1.5,
        gate_logit_scale: float = 1.0,
        gate_logit_bias: float = 0.0,
        teacher_trailing_mode: TeacherTrailingMode = "passthrough",
        structured_features: bool = True,
    ) -> None:
        super().__init__()
        if residual_limit <= 0.0:
            raise ValueError("residual_limit must be positive")
        if gate_power <= 0.0:
            raise ValueError("gate_power must be positive")
        if not math.isfinite(gate_logit_scale) or gate_logit_scale <= 0.0:
            raise ValueError("gate_logit_scale must be finite and positive")
        if not math.isfinite(gate_logit_bias):
            raise ValueError("gate_logit_bias must be finite")
        if teacher_trailing_mode not in ("passthrough", "assume_fresh"):
            raise ValueError(f"unsupported teacher_trailing_mode={teacher_trailing_mode!r}")
        self.teacher = teacher.freeze()
        self.residual_limit = float(residual_limit)
        self.gate_power = float(gate_power)
        # Calibration changes the deployed action even though it is not a
        # learned gate weight.  It therefore belongs in state_dict: otherwise
        # the same strict checkpoint can silently behave differently when a
        # caller selects another runner config.  Legacy checkpoints are
        # migrated explicitly in ``_load_from_state_dict`` below.
        self.register_buffer(
            "gate_logit_scale",
            torch.tensor(float(gate_logit_scale), dtype=torch.float32),
        )
        self.register_buffer(
            "gate_logit_bias",
            torch.tensor(float(gate_logit_bias), dtype=torch.float32),
        )
        self.loaded_legacy_calibration = False
        self.teacher_trailing_mode = teacher_trailing_mode
        self.structured_features = bool(structured_features)

        # The fast branch already contains the validated shared temporal Hall
        # encoder.  Reusing its frozen per-foot latents makes the correction
        # path explicitly Hall-conditioned instead of asking a 1864-wide
        # dense layer to rediscover the 15x2x15x3 structure from scratch.
        # The original 480-D proprioception and four health/motion channels
        # remain available for phase-aware capture actions.
        residual_input_dim = 480 + 2 * 32 + 4 if self.structured_features else INPUT_DIM

        self.residual = nn.Sequential(
            nn.Linear(residual_input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, OUTPUT_DIM),
        )
        self.gate = nn.Sequential(
            nn.Linear(residual_input_dim, 128),
            nn.ELU(),
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Linear(32, 1),
        )
        # Initial exported behavior is exactly the fast actor.  A conservative
        # negative gate bias also protects HIGH regions during early fitting.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)

        fresh_mask = torch.ones(INPUT_DIM, dtype=torch.float32)
        fresh_mask[TRAILING_SLICE] = 0.0
        self.register_buffer("teacher_fresh_mask", fresh_mask)

    def train(self, mode: bool = True) -> FastBaseHallCaptureResidual:
        super().train(mode)
        self.teacher.freeze()
        return self

    def _validate(self, observation: torch.Tensor) -> None:
        if observation.ndim != 2 or observation.shape[1] != INPUT_DIM:
            raise ValueError(f"expected [batch,{INPUT_DIM}], got {tuple(observation.shape)}")

    def teacher_observation(self, observation: torch.Tensor) -> torch.Tensor:
        self._validate(observation)
        if self.teacher_trailing_mode == "assume_fresh":
            return observation * self.teacher_fresh_mask.to(dtype=observation.dtype)
        return observation

    def base_action(self, observation: torch.Tensor) -> torch.Tensor:
        base = self.teacher(self.teacher_observation(observation))
        return torch.clamp(torch.nan_to_num(base, nan=0.0, posinf=3.0, neginf=-3.0), -3.0, 3.0)

    def capture_features(self, observation: torch.Tensor) -> torch.Tensor:
        """Build deployable Hall/proprio features without privileged state."""

        self._validate(observation)
        if not self.structured_features:
            return observation
        teacher_observation = self.teacher_observation(observation)
        # Parameters are frozen, but autograd through this deterministic
        # encoder is harmless and keeps export a single self-contained graph.
        _, left_hall, right_hall = self.teacher.fast.encode(teacher_observation)
        return torch.cat(
            (observation[:, :480], left_hall, right_hall, observation[:, -4:]),
            dim=1,
        )

    def raw_capture_probability(self, observation: torch.Tensor) -> torch.Tensor:
        """Uncalibrated LOW probability used by the training BCE."""

        features = self.capture_features(observation)
        return torch.sigmoid(self.gate(features)).pow(self.gate_power)

    def calibrate_capture_probability(self, raw_probability: torch.Tensor) -> torch.Tensor:
        """Apply monotone deployment calibration without changing gate ranking."""

        if self.gate_logit_scale == 1.0 and self.gate_logit_bias == 0.0:
            # Preserve exact historical numerics for every old/default task.
            return raw_probability
        safe = raw_probability.clamp(1.0e-6, 1.0 - 1.0e-6)
        raw_logit = torch.log(safe) - torch.log1p(-safe)
        return torch.sigmoid(
            self.gate_logit_scale * raw_logit + self.gate_logit_bias
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Migrate old checkpoints, then persist calibration unambiguously."""

        scale_key = prefix + "gate_logit_scale"
        bias_key = prefix + "gate_logit_bias"
        legacy = scale_key not in state_dict and bias_key not in state_dict
        if (scale_key in state_dict) != (bias_key in state_dict):
            error_msgs.append(
                "FastBase checkpoint contains only one gate calibration value"
            )
        if legacy:
            state_dict[scale_key] = self.gate_logit_scale.detach().clone()
            state_dict[bias_key] = self.gate_logit_bias.detach().clone()
            self.loaded_legacy_calibration = True
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def capture_probability(self, observation: torch.Tensor) -> torch.Tensor:
        """Calibrated probability used as deployable residual authority."""

        return self.calibrate_capture_probability(
            self.raw_capture_probability(observation)
        )

    def sensor_confidence(self, observation: torch.Tensor) -> torch.Tensor:
        """Return a hard deployable confidence floor from both foot packets."""

        self._validate(observation)
        return observation[:, VALID_SLICE].amin(dim=1, keepdim=True).clamp(0.0, 1.0)

    def effective_capture_probability(self, observation: torch.Tensor) -> torch.Tensor:
        """Gate authority; any complete-foot outage removes all residual action."""

        return self.sensor_confidence(observation) * self.capture_probability(observation)

    def capture_delta(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.capture_features(observation)
        raw_probability = torch.sigmoid(self.gate(features)).pow(self.gate_power)
        calibrated = self.calibrate_capture_probability(raw_probability)
        if getattr(self, "force_capture_gate_open", False):
            probability = self.sensor_confidence(observation)
        else:
            probability = self.sensor_confidence(observation) * calibrated
        bounded = self.residual_limit * torch.tanh(self.residual(features))
        return probability * bounded

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        action = self.base_action(observation) + self.capture_delta(observation)
        return torch.clamp(torch.nan_to_num(action, nan=0.0, posinf=3.0, neginf=-3.0), -3.0, 3.0)


class FastBaseHallCaptureStabilityResidual(FastBaseHallCaptureResidual):
    """Capture residual plus an independent bounded proprioceptive safety residual.

    The Hall gate is intentionally allowed to release on the final high-grip
    patch.  A separate residual is therefore required to learn rare long-tail
    heading/roll recovery without falsely keeping the LOW gate open.  Its
    authority is a smooth function of deployable IMU/heading observations and
    it is exactly zero in the nominal region.  The final affine layer is also
    zero initialized, so importing an existing capture-only checkpoint is
    bit-exact before the first new optimizer update.
    """

    def __init__(
        self,
        teacher: FrozenSpeedBoostTeacher,
        *,
        stability_limit: float = 0.25,
        stability_heading_start: float = 0.25,
        stability_heading_full: float = 0.55,
        stability_tilt_start: float = 0.08,
        stability_tilt_full: float = 0.25,
        stability_omega_start: float = 0.60,
        stability_omega_full: float = 1.80,
        stability_turning_yaw_threshold: float = 0.05,
        **capture_kwargs,
    ) -> None:
        super().__init__(teacher, **capture_kwargs)
        limits = {
            "stability_limit": stability_limit,
            "stability_heading_start": stability_heading_start,
            "stability_heading_full": stability_heading_full,
            "stability_tilt_start": stability_tilt_start,
            "stability_tilt_full": stability_tilt_full,
            "stability_omega_start": stability_omega_start,
            "stability_omega_full": stability_omega_full,
            "stability_turning_yaw_threshold": stability_turning_yaw_threshold,
        }
        if not all(math.isfinite(float(value)) for value in limits.values()):
            raise ValueError("stability residual limits must be finite")
        if stability_limit <= 0.0:
            raise ValueError("stability_limit must be positive")
        for name, low, high in (
            ("heading", stability_heading_start, stability_heading_full),
            ("tilt", stability_tilt_start, stability_tilt_full),
            ("omega", stability_omega_start, stability_omega_full),
        ):
            if low < 0.0 or high <= low:
                raise ValueError(
                    f"stability {name} thresholds must satisfy 0 <= start < full"
                )
        if stability_turning_yaw_threshold < 0.0:
            raise ValueError("stability_turning_yaw_threshold must be non-negative")

        # Persist every deployment-semantic scalar in the actor state.  A
        # strict checkpoint must reconstruct behavior independently of which
        # runner config happens to instantiate it.
        for name, value in limits.items():
            self.register_buffer(name, torch.tensor(float(value), dtype=torch.float32))
        self.loaded_legacy_stability = False

        stability_input_dim = 480 + 2
        self.stability_residual = nn.Sequential(
            nn.Linear(stability_input_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, OUTPUT_DIM),
        )
        nn.init.zeros_(self.stability_residual[-1].weight)
        nn.init.zeros_(self.stability_residual[-1].bias)

    @staticmethod
    def _smooth_authority(
        value: torch.Tensor, start: torch.Tensor, full: torch.Tensor
    ) -> torch.Tensor:
        ratio = ((value - start) / (full - start)).clamp(0.0, 1.0)
        return ratio.square() * (3.0 - 2.0 * ratio)

    def stability_features(self, observation: torch.Tensor) -> torch.Tensor:
        """Return only causal proprioception and motion feedback (482 values)."""

        self._validate(observation)
        return torch.cat(
            (observation[:, :480], observation[:, MOTION_FEEDBACK_SLICE]), dim=1
        )

    def stability_authority(self, observation: torch.Tensor) -> torch.Tensor:
        """Smooth observation-only correction authority in ``[0,1]``."""

        self._validate(observation)
        yaw_command = observation[:, list(COMMAND_YAW_INDICES)].mean(dim=1)
        straight = yaw_command.abs() <= self.stability_turning_yaw_threshold
        heading = observation[:, MOTION_FEEDBACK_SLICE.stop - 1].abs()
        heading_risk = self._smooth_authority(
            heading, self.stability_heading_start, self.stability_heading_full
        ) * straight.to(dtype=observation.dtype)

        gravity = observation[:, LATEST_PROJECTED_GRAVITY_SLICE]
        tilt = torch.linalg.vector_norm(gravity[:, :2], dim=1)
        tilt_risk = self._smooth_authority(
            tilt, self.stability_tilt_start, self.stability_tilt_full
        )

        angular_velocity = (
            observation[:, LATEST_BASE_ANG_VEL_SLICE]
            / BASE_ANG_VEL_OBSERVATION_SCALE
        )
        omega_xy = torch.linalg.vector_norm(angular_velocity[:, :2], dim=1)
        omega_risk = self._smooth_authority(
            omega_xy, self.stability_omega_start, self.stability_omega_full
        )
        return torch.stack((heading_risk, tilt_risk, omega_risk), dim=1).amax(
            dim=1, keepdim=True
        )

    def stability_delta(self, observation: torch.Tensor) -> torch.Tensor:
        authority = self.stability_authority(observation)
        bounded = self.stability_limit * torch.tanh(
            self.stability_residual(self.stability_features(observation))
        )
        return authority * bounded

    def anchor_action_without_stability(
        self, observation: torch.Tensor
    ) -> torch.Tensor:
        """Return the frozen-base + Hall-capture path used by HIGH anchoring.

        The long-tail stability branch must be allowed to correct a rare
        final-HIGH attitude failure.  Anchoring the *composite* mean to the
        frozen teacher would directly cancel that PPO gradient.  This method
        retains the original high-speed/capture contract while keeping the
        independent stability correction outside the teacher loss.
        """

        action = self.base_action(observation) + self.capture_delta(observation)
        return torch.clamp(
            torch.nan_to_num(action, nan=0.0, posinf=3.0, neginf=-3.0),
            -3.0,
            3.0,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Import a capture-only actor with an exact-zero stability branch."""

        current = self.state_dict()
        stability_names = tuple(
            name
            for name in current
            if name.startswith("stability_")
        )
        present = tuple(prefix + name in state_dict for name in stability_names)
        if any(present) and not all(present):
            error_msgs.append(
                "FastBase checkpoint contains only part of the stability residual state"
            )
        elif not any(present):
            for name in stability_names:
                state_dict[prefix + name] = current[name].detach().clone()
            self.loaded_legacy_stability = True
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        action = (
            self.base_action(observation)
            + self.capture_delta(observation)
            + self.stability_delta(observation)
        )
        return torch.clamp(
            torch.nan_to_num(action, nan=0.0, posinf=3.0, neginf=-3.0),
            -3.0,
            3.0,
        )


def trainable_parameters(model: FastBaseHallCaptureResidual):
    """Yield only residual/gate parameters and assert that the base is frozen."""

    if any(parameter.requires_grad for parameter in model.teacher.parameters()):
        raise RuntimeError("speedboost teacher must remain frozen")
    yield from model.residual.parameters()
    yield from model.gate.parameters()


class FastBaseHallCaptureRslModel(MLPModel):
    """RSL-RL 5 Actor with a frozen speedboost112 mean and Hall residual.

    RSL still owns the Gaussian distribution, log probability, entropy and KL
    calculation.  Only the deterministic mean module is replaced, so native
    PPO storage, checkpoints and ``as_jit``/``as_onnx`` exporters keep working.
    The Teacher artifact is reconstructed before every strict checkpoint load;
    its tensors are nevertheless included in the actor state dict for a fully
    self-contained, auditable checkpoint.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        teacher_checkpoint: str,
        residual_limit: float = 1.25,
        gate_power: float = 1.0,
        gate_logit_scale: float = 1.0,
        gate_logit_bias: float = 0.0,
        teacher_trailing_mode: TeacherTrailingMode = "assume_fresh",
        structured_features: bool = True,
        hidden_dims: tuple[int, ...] | list[int] = (1,),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        if output_dim != OUTPUT_DIM:
            raise ValueError(f"speedboost capture Actor requires {OUTPUT_DIM} actions, got {output_dim}")
        if obs_normalization:
            raise ValueError(
                "speedboost capture Actor must receive the raw audited 1864-D schema; "
                "external empirical normalization is unsupported"
            )
        checkpoint = Path(teacher_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"frozen speedboost Teacher not found: {checkpoint}")

        # Let RSL construct its native Gaussian distribution and public model
        # bookkeeping.  The temporary 1-wide MLP is immediately replaced.
        del hidden_dims
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=(1,),
            activation=activation,
            obs_normalization=False,
            distribution_cfg=distribution_cfg,
        )
        if self.obs_dim != INPUT_DIM or self.obs_groups != ["policy"]:
            raise ValueError(
                "fast-base capture Actor requires exactly policy[1864], got "
                f"groups={self.obs_groups!r}, dim={self.obs_dim}"
            )
        teacher = load_frozen_speedboost_teacher(checkpoint, device="cpu")
        self.mlp = FastBaseHallCaptureResidual(
            teacher,
            residual_limit=residual_limit,
            gate_power=gate_power,
            gate_logit_scale=gate_logit_scale,
            gate_logit_bias=gate_logit_bias,
            teacher_trailing_mode=teacher_trailing_mode,
            structured_features=structured_features,
        )
        self.teacher_checkpoint = str(checkpoint)

    def train(self, mode: bool = True) -> FastBaseHallCaptureRslModel:
        super().train(mode)
        self.mlp.teacher.freeze()
        return self

    def capture_probability(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
    ) -> torch.Tensor:
        """Expose the deployable gate to an optional training-only BCE loss."""

        latent = self.get_latent(obs, masks, hidden_state)
        return self.mlp.capture_probability(latent)

    def raw_capture_probability(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
    ) -> torch.Tensor:
        """Expose the uncalibrated gate specifically for training supervision."""

        latent = self.get_latent(obs, masks, hidden_state)
        return self.mlp.raw_capture_probability(latent)

    def trainable_actor_parameters(self):
        """Yield residual/gate/distribution parameters, never Teacher weights."""

        yield from trainable_parameters(self.mlp)
        if self.distribution is not None:
            yield from self.distribution.parameters()

    def as_jit(self) -> nn.Module:
        """Return a traced deterministic graph accepted by RSL's JIT exporter.

        The recovered Teacher contains eager-only validation branches.  Its
        tensor computation is trace-safe, so tracing here avoids weakening
        those runtime checks merely to satisfy TorchScript's parser.  RSL's
        subsequent ``torch.jit.script`` call is a no-op on the returned
        ``ScriptModule``.
        """

        export = super().as_jit().cpu().eval()
        return torch.jit.trace(
            export,
            torch.zeros(1, INPUT_DIM),
            check_trace=False,
        )


class FastBaseHallCaptureOpenGateRslModel(FastBaseHallCaptureRslModel):
    """Expert-only actor with full healthy-Hall residual authority."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp.force_capture_gate_open = True

class FastBaseHallCaptureStabilityRslModel(FastBaseHallCaptureRslModel):
    """Isolated RSL actor with Hall capture and proprioceptive stability paths."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        stability_limit: float = 0.25,
        stability_heading_start: float = 0.25,
        stability_heading_full: float = 0.55,
        stability_tilt_start: float = 0.08,
        stability_tilt_full: float = 0.25,
        stability_omega_start: float = 0.60,
        stability_omega_full: float = 1.80,
        stability_turning_yaw_threshold: float = 0.05,
        freeze_capture_branches: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        previous = self.mlp
        self.mlp = FastBaseHallCaptureStabilityResidual(
            previous.teacher,
            residual_limit=previous.residual_limit,
            gate_power=previous.gate_power,
            gate_logit_scale=float(previous.gate_logit_scale),
            gate_logit_bias=float(previous.gate_logit_bias),
            teacher_trailing_mode=previous.teacher_trailing_mode,
            structured_features=previous.structured_features,
            stability_limit=stability_limit,
            stability_heading_start=stability_heading_start,
            stability_heading_full=stability_heading_full,
            stability_tilt_start=stability_tilt_start,
            stability_tilt_full=stability_tilt_full,
            stability_omega_start=stability_omega_start,
            stability_omega_full=stability_omega_full,
            stability_turning_yaw_threshold=stability_turning_yaw_threshold,
        )
        if freeze_capture_branches:
            # Transition-retention mode.  AnchoredPPO recognizes this
            # actor-owned contract, keeps both capture optimizer roles at
            # lr=0 and reasserts requires_grad=False after checkpoint loads,
            # so the validated LOW adaptation cannot be relearned.
            self.mlp.freeze_capture_branches = True
            for module in (self.mlp.gate, self.mlp.residual):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def trainable_actor_parameters(self):
        yield from trainable_parameters(self.mlp)
        yield from self.mlp.stability_residual.parameters()
        if self.distribution is not None:
            yield from self.distribution.parameters()


class FastBaseHallCaptureHighEndRecoveryResidual(
    FastBaseHallCaptureStabilityResidual
):
    """Training-only expert head with full HighEnd recovery authority.

    State-bank resets begin in an already dangerous mechanical state.  Making
    the expert wait for a newly auto-zeroed Hall packet disables it during the
    only frames this isolated task is meant to learn.  The final deployable
    bounded residual retains the ordinary sensor-health fail-closed gate; this
    teacher intentionally answers the preceding feasibility question: can a
    controller recover the state at all under the real actuator limits?
    """

    def stability_authority(self, observation: torch.Tensor) -> torch.Tensor:
        self._validate(observation)
        return torch.ones(
            (observation.shape[0], 1),
            device=observation.device,
            dtype=observation.dtype,
        )


class FastBaseHallCaptureHighEndRecoveryRslModel(
    FastBaseHallCaptureStabilityRslModel
):
    """Model55 capture path plus a fresh unrestricted HighEnd recovery head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        previous = self.mlp
        self.mlp = FastBaseHallCaptureHighEndRecoveryResidual(
            previous.teacher,
            residual_limit=previous.residual_limit,
            gate_power=previous.gate_power,
            gate_logit_scale=float(previous.gate_logit_scale),
            gate_logit_bias=float(previous.gate_logit_bias),
            teacher_trailing_mode=previous.teacher_trailing_mode,
            structured_features=previous.structured_features,
            stability_limit=float(previous.stability_limit),
            stability_heading_start=float(previous.stability_heading_start),
            stability_heading_full=float(previous.stability_heading_full),
            stability_tilt_start=float(previous.stability_tilt_start),
            stability_tilt_full=float(previous.stability_tilt_full),
            stability_omega_start=float(previous.stability_omega_start),
            stability_omega_full=float(previous.stability_omega_full),
            stability_turning_yaw_threshold=float(
                previous.stability_turning_yaw_threshold
            ),
        )
        # This feasibility teacher answers only whether a dedicated HighEnd
        # correction can recover the frozen locomotion backbone.  Its rollout
        # must not simultaneously relearn the Hall LOW gate/residual and erase
        # the already validated H->L->H behavior.  AnchoredPPO recognizes this
        # actor-owned contract, keeps both capture optimizer roles at lr=0 and
        # reasserts requires_grad=False after checkpoint loads.
        self.mlp.freeze_capture_branches = True
        for module in (self.mlp.gate, self.mlp.residual):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
