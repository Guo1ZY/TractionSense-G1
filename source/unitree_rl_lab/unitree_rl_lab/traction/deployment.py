"""Shared deployment preprocessing, safety checks, adapters, and policy runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import torch

from .governor import GovernorOutput, TractionAwareCommandGovernor
from .history import TemporalHistoryBuffer
from .schema import (
    ACTION_DIM,
    G1_29DOF_JOINT_ORDER,
    POLICY_DT_S,
    TEMPORAL_STUDENT_FRAME_SCHEMA,
    concatenate_terms,
)
from .sensor_layout import DualFootForceInput


NOMINAL_ROBOT_MASS_KG = 35.2793

DEFAULT_JOINT_POSITION = (
    -0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3,
    0.3, 0.3, 0.3, -0.2, -0.2, 0.25, -0.25, 0.0, 0.0, 0.0,
    0.0, 0.97, 0.97, 0.15, -0.15, 0.0, 0.0, 0.0, 0.0,
)
JOINT_STIFFNESS = (
    100.0, 100.0, 200.0, 100.0, 100.0, 40.0, 100.0, 100.0, 40.0,
    150.0, 150.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
    40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
    40.0,
)
JOINT_DAMPING = (
    2.0, 2.0, 5.0, 2.0, 2.0, 5.0, 2.0, 2.0, 5.0, 4.0, 4.0,
    1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
)
JOINT_EFFORT_LIMIT = (
    88.0, 88.0, 88.0, 139.0, 139.0, 25.0, 88.0, 88.0, 25.0,
    139.0, 139.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0, 5.0, 5.0,
)


@dataclass(frozen=True)
class DeploymentObservationCfg:
    """Audited physical-to-policy transforms."""

    robot_mass_kg: float = NOMINAL_ROBOT_MASS_KG
    base_angular_velocity_scale: float = 0.2
    joint_velocity_scale: float = 0.05
    force_normalized_clip: tuple[float, float] = (-2.0, 2.0)
    maximum_force_age_s: float = 1.0
    force_saturation_n: float = 900.0
    action_scale: float = 0.25
    policy_action_clip: tuple[float, float] = (-10.0, 10.0)
    slip_probability_on: float = 0.50
    slip_probability_off: float = 0.30

    def __post_init__(self) -> None:
        if self.robot_mass_kg <= 0.0 or self.maximum_force_age_s <= 0.0:
            raise ValueError("robot mass and maximum force age must be positive")
        if self.force_saturation_n <= 0.0 or self.action_scale <= 0.0:
            raise ValueError("force saturation and action scale must be positive")
        if not (
            0.0
            <= self.slip_probability_off
            < self.slip_probability_on
            <= 1.0
        ):
            raise ValueError("slip probability hysteresis is invalid")


@dataclass(frozen=True)
class ProprioceptiveState:
    timestamp: float
    base_angular_velocity: np.ndarray
    projected_gravity: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    previous_action: np.ndarray
    base_linear_velocity: np.ndarray

    def __post_init__(self) -> None:
        expected = {
            "base_angular_velocity": 3,
            "projected_gravity": 3,
            "joint_position": ACTION_DIM,
            "joint_velocity": ACTION_DIM,
            "previous_action": ACTION_DIM,
            "base_linear_velocity": 3,
        }
        for name, dimension in expected.items():
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != (dimension,):
                raise ValueError(
                    f"{name} must have shape ({dimension},), got {value.shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
            object.__setattr__(self, name, value)


class PolicyRuntimeOutput(NamedTuple):
    action: torch.Tensor
    joint_position_target: torch.Tensor
    slip_probability: torch.Tensor
    traction_score: torch.Tensor
    sensor_confidence: torch.Tensor
    governor: GovernorOutput
    safety_flags: tuple[str, ...]


class CanonicalObservationBuilder:
    """Create the unique 106-D frame and 15-frame time-major history."""

    def __init__(
        self,
        *,
        cfg: DeploymentObservationCfg = DeploymentObservationCfg(),
        device: str | torch.device = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.history = TemporalHistoryBuffer(
            1,
            TEMPORAL_STUDENT_FRAME_SCHEMA.history_frames,
            TEMPORAL_STUDENT_FRAME_SCHEMA.frame_dimension,
            device=self.device,
        )
        self.last_timestamp = -np.inf
        self.sample_count = 0

    def reset(self) -> None:
        self.history.reset()
        self.last_timestamp = -np.inf
        self.sample_count = 0

    def append(
        self,
        state: ProprioceptiveState,
        force: DualFootForceInput,
        policy_command: np.ndarray,
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        if state.timestamp <= self.last_timestamp:
            raise ValueError(
                f"proprio timestamp did not increase: {state.timestamp} <= "
                f"{self.last_timestamp}"
            )
        self.last_timestamp = state.timestamp
        command = np.asarray(policy_command, dtype=np.float32)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("policy_command must be a finite shape-(3,) vector")

        flags: list[str] = []
        force_vector = force.force_vector.copy()
        saturated = np.abs(force_vector) >= self.cfg.force_saturation_n
        if saturated.any():
            flags.append("force_saturation")
            force_vector = np.clip(
                force_vector,
                -self.cfg.force_saturation_n,
                self.cfg.force_saturation_n,
            )
        valid = force.valid_vector
        age = np.minimum(force.age_vector, self.cfg.maximum_force_age_s)
        if not bool(force.left_valid):
            flags.append("left_force_invalid")
        if not bool(force.right_valid):
            flags.append("right_force_invalid")
        if force.timestamp > state.timestamp + POLICY_DT_S:
            flags.append("force_timestamp_in_future")
            valid[:] = 0.0

        normalized_force = np.clip(
            force_vector / (self.cfg.robot_mass_kg * 9.81),
            *self.cfg.force_normalized_clip,
        )
        values = {
            "base_ang_vel": state.base_angular_velocity
            * self.cfg.base_angular_velocity_scale,
            "projected_gravity": state.projected_gravity,
            "joint_pos_rel": state.joint_position
            - np.asarray(DEFAULT_JOINT_POSITION, dtype=np.float32),
            "joint_vel_rel": state.joint_velocity * self.cfg.joint_velocity_scale,
            "previous_action": state.previous_action,
            "raw_command": command,
            "observed_foot_force": normalized_force,
            "foot_force_valid": valid,
            "foot_force_age": age,
        }
        frame = concatenate_terms(TEMPORAL_STUDENT_FRAME_SCHEMA, values)
        frame_tensor = torch.as_tensor(
            frame,
            dtype=torch.float32,
            device=self.device,
        )[None]
        self.history.append(frame_tensor)
        self.sample_count += 1
        return self.history.flatten(), tuple(flags)


class TractionPolicyRuntime:
    """Two-pass causal estimator/governor/policy runtime.

    Pass one estimates traction from the history containing the raw joystick
    command. The governor computes an adjusted command. Pass two replaces only
    the newest command field and evaluates the fixed policy action. No model
    weights are updated in this runtime.
    """

    def __init__(
        self,
        policy: Callable[
            [torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        *,
        observation_cfg: DeploymentObservationCfg = DeploymentObservationCfg(),
        device: str | torch.device = "cpu",
        governor_enabled: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.policy = policy
        self.observation = CanonicalObservationBuilder(
            cfg=observation_cfg,
            device=self.device,
        )
        self.governor = TractionAwareCommandGovernor(1, device=self.device)
        self.cfg = observation_cfg
        self.governor_enabled = governor_enabled
        self.slip_duration = torch.zeros((1, 2), device=self.device)
        self.slipping = torch.zeros(
            (1, 2),
            dtype=torch.bool,
            device=self.device,
        )

    def reset(self) -> None:
        self.observation.reset()
        self.governor.reset()
        self.slip_duration.zero_()
        self.slipping.zero_()

    def _evaluate(
        self,
        history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.policy(history)
        if not isinstance(output, (tuple, list)) or len(output) != 4:
            raise ValueError("policy must return action, slip, traction, confidence")
        expected = ((1, 29), (1, 2), (1, 1), (1, 1))
        result = tuple(value.to(self.device) for value in output)
        for value, shape in zip(result, expected, strict=True):
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"policy output shape {tuple(value.shape)}, expected {shape}"
                )
        return result  # type: ignore[return-value]

    def step(
        self,
        state: ProprioceptiveState,
        force: DualFootForceInput,
        raw_command: np.ndarray,
    ) -> PolicyRuntimeOutput:
        history, flags = self.observation.append(state, force, raw_command)
        action, slip, traction, confidence = self._evaluate(history)
        nonfinite = any(
            not bool(torch.isfinite(value).all())
            for value in (action, slip, traction, confidence)
        )
        safety_flags = list(flags)
        if nonfinite:
            safety_flags.append("nonfinite_policy_output")
            action = torch.zeros_like(action)
            slip = torch.ones_like(slip)
            traction = torch.zeros_like(traction)
            confidence = torch.zeros_like(confidence)

        history_ready = (
            self.observation.sample_count
            >= TEMPORAL_STUDENT_FRAME_SCHEMA.history_frames
        )
        if history_ready:
            enter_slip = slip >= self.cfg.slip_probability_on
            exit_slip = slip <= self.cfg.slip_probability_off
            self.slipping = torch.where(
                self.slipping,
                ~exit_slip,
                enter_slip,
            )
        else:
            self.slipping.zero_()
        self.slip_duration = torch.where(
            self.slipping,
            self.slip_duration + POLICY_DT_S,
            torch.zeros_like(self.slip_duration),
        )
        raw = torch.as_tensor(
            np.asarray(raw_command, dtype=np.float32),
            device=self.device,
        )[None]
        current_velocity = torch.as_tensor(
            state.base_linear_velocity,
            device=self.device,
        )[None]
        if self.governor_enabled:
            governor_slip = slip.clamp(0.0, 1.0)
            governor_traction = traction.clamp(0.0, 1.0)
            governor_confidence = confidence.clamp(0.0, 1.0)
            if not history_ready:
                # A zero-filled reset history is part of the canonical schema,
                # but estimator outputs are not acted on until one real
                # temporal window has arrived.
                governor_slip = torch.zeros_like(governor_slip)
                governor_traction = torch.ones_like(governor_traction)
                governor_confidence = torch.ones_like(governor_confidence)
            governor = self.governor.update(
                raw,
                governor_slip,
                governor_traction,
                governor_confidence,
                self.slip_duration,
                current_velocity,
            )
        else:
            # Explicit ablation/reference path. The fixed policy still emits
            # diagnostics, but the joystick command reaches the actor unchanged.
            ones = torch.ones((1, 1), device=self.device)
            governor = GovernorOutput(
                raw,
                ones * self.governor.cfg.normal_acceleration_limit,
                ones * self.governor.cfg.normal_deceleration_limit,
                ones,
                ones,
                ones * self.governor.cfg.normal_max_yaw,
                torch.zeros((1, 1), device=self.device),
                torch.zeros((1,), device=self.device, dtype=torch.long),
            )

        # Re-evaluate the fixed model with the adjusted command in the newest
        # frame. Estimator outputs retained above caused the governor decision.
        governed_history = history.clone().reshape(
            1,
            TEMPORAL_STUDENT_FRAME_SCHEMA.history_frames,
            TEMPORAL_STUDENT_FRAME_SCHEMA.frame_dimension,
        )
        command_slice = TEMPORAL_STUDENT_FRAME_SCHEMA.term_slice("raw_command")
        governed_history[:, -1, command_slice] = governor.adjusted_command
        governed_action, _, _, _ = self._evaluate(governed_history.flatten(1))
        if torch.isfinite(governed_action).all() and not nonfinite:
            action = governed_action
        else:
            safety_flags.append("nonfinite_governed_policy_output")
            action = torch.zeros_like(action)
        action = action.clamp(*self.cfg.policy_action_clip)
        target = torch.as_tensor(
            DEFAULT_JOINT_POSITION,
            device=self.device,
        )[None] + self.cfg.action_scale * action
        return PolicyRuntimeOutput(
            action,
            target,
            slip,
            traction,
            confidence,
            governor,
            tuple(safety_flags),
        )


class IsaacForceAdapter:
    @staticmethod
    def adapt(
        timestamp: float,
        force_xyz_n: np.ndarray,
        *,
        valid: np.ndarray | tuple[bool, bool] = (True, True),
        age_s: np.ndarray | tuple[float, float] = (0.0, 0.0),
    ) -> DualFootForceInput:
        force = np.asarray(force_xyz_n, dtype=np.float32)
        valid_array = np.asarray(valid, dtype=bool)
        age = np.asarray(age_s, dtype=np.float32)
        if force.shape != (6,) or valid_array.shape != (2,) or age.shape != (2,):
            raise ValueError("force/valid/age shapes must be (6,)/(2,)/(2,)")
        return DualFootForceInput(
            timestamp=float(timestamp),
            left_force_xyz=force[:3],
            right_force_xyz=force[3:],
            left_valid=bool(valid_array[0]),
            right_valid=bool(valid_array[1]),
            left_age=float(age[0]),
            right_age=float(age[1]),
            left_source="isaac_contact_sensor",
            right_source="isaac_contact_sensor",
        )


class OfflineRecordedForceAdapter:
    """Read force-level samples without treating Hall-only data as force."""

    def __init__(
        self,
        path: str | Path,
        *,
        force_key: str = "ideal_force_xyz_n",
        normalized_force: bool = False,
        robot_mass_kg: float = NOMINAL_ROBOT_MASS_KG,
    ) -> None:
        with np.load(path, allow_pickle=False) as archive:
            if force_key not in archive:
                raise ValueError(
                    f"{path} has no {force_key!r}; raw Hall data is not calibrated force"
                )
            self.force = np.asarray(archive[force_key], dtype=np.float32)
            self.timestamp = np.asarray(archive["timestamp_s"]).reshape(-1)
            self.valid = np.asarray(
                archive["sensor_valid"]
                if "sensor_valid" in archive
                else np.ones((len(self.force), 2))
            )
            self.age = np.asarray(
                archive["sensor_age_s"]
                if "sensor_age_s" in archive
                else np.zeros((len(self.force), 2))
            )
        if self.force.shape != (len(self.timestamp), 6):
            raise ValueError("offline force must have shape [samples,6]")
        if normalized_force:
            self.force *= float(robot_mass_kg) * 9.81

    def __len__(self) -> int:
        return len(self.force)

    def sample(self, index: int) -> DualFootForceInput:
        return IsaacForceAdapter.adapt(
            float(self.timestamp[index]),
            self.force[index],
            valid=self.valid[index],
            age_s=self.age[index],
        )


def deployment_metadata(cfg: DeploymentObservationCfg) -> dict[str, object]:
    """Serializable controller facts included with every exported policy."""
    return {
        "observation": TEMPORAL_STUDENT_FRAME_SCHEMA.to_dict(),
        "action_dimension": ACTION_DIM,
        "joint_order": G1_29DOF_JOINT_ORDER,
        "default_joint_position": DEFAULT_JOINT_POSITION,
        "joint_stiffness": JOINT_STIFFNESS,
        "joint_damping": JOINT_DAMPING,
        "joint_effort_limit": JOINT_EFFORT_LIMIT,
        "control_frequency_hz": 1.0 / POLICY_DT_S,
        "physics_frequency_hz": 200.0,
        "decimation": 4,
        "preprocessing": asdict(cfg),
        "runtime": {
            "governor": "TractionAwareCommandGovernor",
            "governor_order": (
                "raw command -> traction estimate -> adjusted command -> fixed policy"
            ),
            "missing_foot": "zero force, valid=0, age clipped; conservative fallback",
            "real_robot_control_authorized": False,
        },
    }
