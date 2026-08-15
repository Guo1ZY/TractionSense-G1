"""Strict HoloSoma FastSAC adapter for matched Isaac Sim evaluation.

The bundled HoloSoma policy does not consume the project's 1864-D Hall
observation.  It consumes a 100-D, single-frame proprioceptive observation and
returns 29 normalized joint-position offsets.  This adapter reconstructs that
published interface from the live Isaac articulation and maps the resulting
joint targets through the *existing* Isaac action term.  It never reads Hall,
contact, force, friction, terrain stage, or other privileged quantities.

This module is intentionally an evaluation adapter, not a deployment path.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper


HOLOSOMA_FASTSAC_SHA256 = (
    "8346fd90778439395922a8c7256f24125ae84b8dea949128bac9e23c02bc7717"
)
HOLOSOMA_OBSERVATION_DIM = 100
HOLOSOMA_ACTION_DIM = 29
HOLOSOMA_POLICY_DT_S = 0.02
HOLOSOMA_GAIT_PERIOD_S = 1.0
HOLOSOMA_ACTION_SCALE_RAD = 0.25
HOLOSOMA_OBSERVATION_TERMS = (
    # HoloSoma BasePolicy sorts configured term names alphabetically.
    "actions",
    "base_ang_vel",
    "command_ang_vel",
    "command_lin_vel",
    "cos_phase",
    "dof_pos",
    "dof_vel",
    "projected_gravity",
    "sin_phase",
)
HOLOSOMA_OBSERVATION_SLICES = {
    "actions": (0, 29),
    "base_ang_vel": (29, 32),
    "command_ang_vel": (32, 33),
    "command_lin_vel": (33, 35),
    "cos_phase": (35, 37),
    "dof_pos": (37, 66),
    "dof_vel": (66, 95),
    "projected_gravity": (95, 98),
    "sin_phase": (98, 100),
}
HOLOSOMA_G1_29DOF_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
HOLOSOMA_DEFAULT_JOINT_POSITION = (
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
)

_HOLOSOMA_FASTSAC_NODE_OPS = (
    "Sub",
    "Div",
    "Constant",
    "Constant",
    "Constant",
    "Constant",
    "Slice",
    "Concat",
    "Gemm",
    "ReduceMean",
    "Sub",
    "Constant",
    "Pow",
    "ReduceMean",
    "Constant",
    "Add",
    "Sqrt",
    "Div",
    "Mul",
    "Add",
    "Sigmoid",
    "Mul",
    "Gemm",
    "ReduceMean",
    "Sub",
    "Constant",
    "Pow",
    "ReduceMean",
    "Constant",
    "Add",
    "Sqrt",
    "Div",
    "Mul",
    "Add",
    "Sigmoid",
    "Mul",
    "Gemm",
    "ReduceMean",
    "Sub",
    "Constant",
    "Pow",
    "ReduceMean",
    "Constant",
    "Add",
    "Sqrt",
    "Div",
    "Mul",
    "Add",
    "Sigmoid",
    "Mul",
    "Gemm",
    "Tanh",
    "Mul",
    "Add",
)

_HOLOSOMA_FASTSAC_INITIALIZER_SHAPES = {
    "actor.action_scale": (29,),
    "actor.action_bias": (29,),
    "actor.net.0.weight": (512, 100),
    "actor.net.0.bias": (512,),
    "actor.net.1.weight": (512,),
    "actor.net.1.bias": (512,),
    "actor.net.3.weight": (256, 512),
    "actor.net.3.bias": (256,),
    "actor.net.4.weight": (256,),
    "actor.net.4.bias": (256,),
    "actor.net.6.weight": (128, 256),
    "actor.net.6.bias": (128,),
    "actor.net.7.weight": (128,),
    "actor.net.7.bias": (128,),
    "actor.fc_mu.0.weight": (29, 128),
    "actor.fc_mu.0.bias": (29,),
    "obs_normalizer._mean": (1, 100),
    "onnx::Div_86": (1, 100),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_unique_complete_order(name: str, actual: tuple[str, ...]) -> None:
    expected = set(HOLOSOMA_G1_29DOF_JOINT_ORDER)
    if len(actual) != HOLOSOMA_ACTION_DIM or len(set(actual)) != len(actual):
        raise RuntimeError(f"{name} must contain 29 unique joint names, got {actual}")
    missing = sorted(expected - set(actual))
    extra = sorted(set(actual) - expected)
    if missing or extra:
        raise RuntimeError(f"{name} joint ABI mismatch: missing={missing}, extra={extra}")


def _onnx_value_shape(value: Any) -> tuple[int, ...]:
    return tuple(int(dim.dim_value) for dim in value.type.tensor_type.shape.dim)


def _load_and_audit_fastsac_onnx(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Load the pinned artifact without requiring ONNX Runtime inside Isaac Sim."""

    model = onnx.load(str(path), load_external_data=False)
    if [(item.domain, int(item.version)) for item in model.opset_import] != [("", 13)]:
        raise RuntimeError("unexpected HoloSoma ONNX opset")
    if len(model.graph.input) != 1 or model.graph.input[0].name != "actor_obs":
        raise RuntimeError("unexpected HoloSoma ONNX input")
    if _onnx_value_shape(model.graph.input[0]) != (1, HOLOSOMA_OBSERVATION_DIM):
        raise RuntimeError("unexpected HoloSoma ONNX input shape")
    if len(model.graph.output) != 1 or model.graph.output[0].name != "action":
        raise RuntimeError("unexpected HoloSoma ONNX output")
    if _onnx_value_shape(model.graph.output[0]) != (1, HOLOSOMA_ACTION_DIM):
        raise RuntimeError("unexpected HoloSoma ONNX output shape")
    node_ops = tuple(node.op_type for node in model.graph.node)
    if node_ops != _HOLOSOMA_FASTSAC_NODE_OPS:
        raise RuntimeError("unexpected HoloSoma FastSAC ONNX graph")

    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    actual_shapes = {name: tuple(int(v) for v in value.shape) for name, value in arrays.items()}
    if actual_shapes != _HOLOSOMA_FASTSAC_INITIALIZER_SHAPES:
        missing = sorted(set(_HOLOSOMA_FASTSAC_INITIALIZER_SHAPES) - set(actual_shapes))
        extra = sorted(set(actual_shapes) - set(_HOLOSOMA_FASTSAC_INITIALIZER_SHAPES))
        raise RuntimeError(
            "unexpected HoloSoma FastSAC initializers: "
            f"missing={missing}, extra={extra}, actual_shapes={actual_shapes}"
        )
    tensors = {
        name: torch.from_numpy(value.copy()).to(dtype=torch.float32)
        for name, value in arrays.items()
    }
    if any(not torch.isfinite(value).all() for value in tensors.values()):
        raise FloatingPointError("HoloSoma ONNX initializers contain NaN/Inf")
    if torch.any(tensors["onnx::Div_86"] <= 0):
        raise RuntimeError("HoloSoma observation divisor must be positive")
    metadata = {item.key: item.value for item in model.metadata_props}
    return tensors, metadata


class HoloSomaFastSacTorchModule(torch.nn.Module):
    """Exact, frozen Torch reconstruction of the pinned FastSAC ONNX graph."""

    def __init__(self, initializers: dict[str, torch.Tensor]) -> None:
        super().__init__()
        names = {
            "action_scale": "actor.action_scale",
            "action_bias": "actor.action_bias",
            "linear0_weight": "actor.net.0.weight",
            "linear0_bias": "actor.net.0.bias",
            "norm0_weight": "actor.net.1.weight",
            "norm0_bias": "actor.net.1.bias",
            "linear1_weight": "actor.net.3.weight",
            "linear1_bias": "actor.net.3.bias",
            "norm1_weight": "actor.net.4.weight",
            "norm1_bias": "actor.net.4.bias",
            "linear2_weight": "actor.net.6.weight",
            "linear2_bias": "actor.net.6.bias",
            "norm2_weight": "actor.net.7.weight",
            "norm2_bias": "actor.net.7.bias",
            "output_weight": "actor.fc_mu.0.weight",
            "output_bias": "actor.fc_mu.0.bias",
            "obs_mean": "obs_normalizer._mean",
            "obs_divisor": "onnx::Div_86",
        }
        for local_name, onnx_name in names.items():
            self.register_buffer(local_name, initializers[onnx_name].detach().clone())

    @classmethod
    def from_onnx(cls, path: str | Path) -> tuple["HoloSomaFastSacTorchModule", dict[str, str]]:
        initializers, metadata = _load_and_audit_fastsac_onnx(Path(path))
        return cls(initializers), metadata

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != HOLOSOMA_OBSERVATION_DIM:
            raise RuntimeError(
                "HoloSoma Torch inference requires [N,100], got "
                f"{tuple(observation.shape)}"
            )
        x = (observation - self.obs_mean) / self.obs_divisor
        x = F.linear(x, self.linear0_weight, self.linear0_bias)
        x = F.layer_norm(x, (512,), self.norm0_weight, self.norm0_bias, 1.0e-5)
        x = F.silu(x)
        x = F.linear(x, self.linear1_weight, self.linear1_bias)
        x = F.layer_norm(x, (256,), self.norm1_weight, self.norm1_bias, 1.0e-5)
        x = F.silu(x)
        x = F.linear(x, self.linear2_weight, self.linear2_bias)
        x = F.layer_norm(x, (128,), self.norm2_weight, self.norm2_bias, 1.0e-5)
        x = F.silu(x)
        x = torch.tanh(F.linear(x, self.output_weight, self.output_bias))
        return x * self.action_scale + self.action_bias


class HoloSomaFastSacIsaacPolicy:
    """Run HoloSoma FastSAC through a pinned Torch reconstruction in Isaac."""

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        robot: Any,
        action_term: Any,
        command_x_m_s: float,
        policy_dt_s: float,
    ) -> None:
        self.path = Path(onnx_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.sha256 = _sha256(self.path)
        if self.sha256 != HOLOSOMA_FASTSAC_SHA256:
            raise RuntimeError(
                "unexpected HoloSoma FastSAC artifact SHA256: "
                f"expected={HOLOSOMA_FASTSAC_SHA256}, actual={self.sha256}"
            )
        if not math.isfinite(float(command_x_m_s)):
            raise ValueError("command_x_m_s must be finite")
        if not math.isclose(
            float(policy_dt_s), HOLOSOMA_POLICY_DT_S, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise RuntimeError(
                f"HoloSoma FastSAC requires 50 Hz, got dt={policy_dt_s}"
            )

        self.robot = robot
        self.action_term = action_term
        self.command_x_m_s = float(command_x_m_s)
        self.device = robot.data.joint_pos.device
        self.num_envs = int(robot.data.joint_pos.shape[0])

        robot_names = tuple(str(name) for name in robot.joint_names)
        action_names = tuple(str(name) for name in action_term._joint_names)
        _require_unique_complete_order("Isaac articulation", robot_names)
        _require_unique_complete_order("Isaac action term", action_names)
        holo_lookup = {
            name: index for index, name in enumerate(HOLOSOMA_G1_29DOF_JOINT_ORDER)
        }
        robot_lookup = {name: index for index, name in enumerate(robot_names)}
        self._robot_to_holo = torch.tensor(
            [robot_lookup[name] for name in HOLOSOMA_G1_29DOF_JOINT_ORDER],
            dtype=torch.long,
            device=self.device,
        )
        self._holo_to_action = torch.tensor(
            [holo_lookup[name] for name in action_names],
            dtype=torch.long,
            device=self.device,
        )
        self._default_holo = torch.tensor(
            HOLOSOMA_DEFAULT_JOINT_POSITION,
            dtype=robot.data.joint_pos.dtype,
            device=self.device,
        ).view(1, -1)

        raw_scale = action_term._scale
        if isinstance(raw_scale, torch.Tensor):
            scale = raw_scale.detach().to(device=self.device)
            if scale.ndim == 1:
                scale = scale.view(1, -1)
            if scale.shape[-1] != HOLOSOMA_ACTION_DIM:
                raise RuntimeError(f"Isaac action scale has shape {tuple(scale.shape)}")
            if not torch.allclose(
                scale,
                torch.full_like(scale, HOLOSOMA_ACTION_SCALE_RAD),
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RuntimeError("Isaac action term is not uniformly scaled by 0.25 rad")
            self._isaac_scale: float | torch.Tensor = scale
        else:
            scale_value = float(raw_scale)
            if not math.isclose(
                scale_value,
                HOLOSOMA_ACTION_SCALE_RAD,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError(
                    f"Isaac action scale must be 0.25 rad, got {scale_value}"
                )
            self._isaac_scale = scale_value

        raw_offset = action_term._offset
        if not isinstance(raw_offset, torch.Tensor):
            raise RuntimeError("Isaac JointPositionAction must use tensor default offsets")
        offset = raw_offset.detach().to(device=self.device)
        if offset.shape != (self.num_envs, HOLOSOMA_ACTION_DIM):
            raise RuntimeError(
                "Isaac action offset must be [num_envs,29], got "
                f"{tuple(offset.shape)}"
            )
        if not torch.isfinite(offset).all():
            raise FloatingPointError("Isaac action offsets contain NaN/Inf")
        self._isaac_offset = offset

        self.torch_module, metadata = HoloSomaFastSacTorchModule.from_onnx(self.path)
        self.torch_module.to(device=self.device, dtype=robot.data.joint_pos.dtype)
        self.torch_module.eval()
        try:
            metadata_joint_names = tuple(json.loads(metadata["dof_names"]))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("ONNX dof_names metadata is missing or invalid") from exc
        if metadata_joint_names != HOLOSOMA_G1_29DOF_JOINT_ORDER:
            raise RuntimeError("ONNX dof_names do not match the audited HoloSoma ABI")
        self.onnx_kp = tuple(float(value) for value in json.loads(metadata["kp"]))
        self.onnx_kd = tuple(float(value) for value in json.loads(metadata["kd"]))
        if len(self.onnx_kp) != 29 or len(self.onnx_kd) != 29:
            raise RuntimeError("ONNX KP/KD metadata must each contain 29 values")

        self.phase_dt = 2.0 * math.pi * policy_dt_s / HOLOSOMA_GAIT_PERIOD_S
        self.phase = torch.empty(
            self.num_envs, 2, dtype=robot.data.joint_pos.dtype, device=self.device
        )
        self.last_policy_action = torch.zeros(
            self.num_envs,
            HOLOSOMA_ACTION_DIM,
            dtype=robot.data.joint_pos.dtype,
            device=self.device,
        )
        self.last_observation = torch.zeros(
            self.num_envs,
            HOLOSOMA_OBSERVATION_DIM,
            dtype=robot.data.joint_pos.dtype,
            device=self.device,
        )
        self.reset()

    def reset(self, env_mask: torch.Tensor | None = None) -> None:
        """Reset HoloSoma's phase and previous-action state for selected envs."""

        if env_mask is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            mask = env_mask.to(device=self.device, dtype=torch.bool).reshape(-1)
            if mask.shape != (self.num_envs,):
                raise RuntimeError(
                    f"reset mask must be [{self.num_envs}], got {tuple(mask.shape)}"
                )
            ids = torch.nonzero(mask, as_tuple=False).flatten()
        if ids.numel() == 0:
            return
        self.phase[ids, 0] = 0.0
        self.phase[ids, 1] = math.pi
        self.last_policy_action[ids] = 0.0
        self.last_observation[ids] = 0.0

    def _build_observation(self) -> torch.Tensor:
        data = self.robot.data
        q_holo = data.joint_pos.index_select(1, self._robot_to_holo)
        dq_holo = data.joint_vel.index_select(1, self._robot_to_holo)
        self.phase.add_(self.phase_dt)
        self.phase.copy_(torch.remainder(self.phase + math.pi, 2.0 * math.pi) - math.pi)
        command_ang = torch.zeros(
            self.num_envs, 1, dtype=q_holo.dtype, device=self.device
        )
        command_lin = torch.zeros(
            self.num_envs, 2, dtype=q_holo.dtype, device=self.device
        )
        command_lin[:, 0] = self.command_x_m_s
        observation = torch.cat(
            (
                self.last_policy_action,
                data.root_ang_vel_b * 0.25,
                command_ang,
                command_lin,
                torch.cos(self.phase),
                q_holo - self._default_holo,
                dq_holo * 0.05,
                data.projected_gravity_b,
                torch.sin(self.phase),
            ),
            dim=1,
        )
        if observation.shape != (self.num_envs, HOLOSOMA_OBSERVATION_DIM):
            raise RuntimeError(
                "constructed HoloSoma observation has wrong shape: "
                f"{tuple(observation.shape)}"
            )
        if not torch.isfinite(observation).all():
            raise FloatingPointError("constructed HoloSoma observation contains NaN/Inf")
        self.last_observation.copy_(observation)
        return observation

    def __call__(self, _environment_observation: Any = None) -> torch.Tensor:
        observation = self._build_observation()
        with torch.inference_mode():
            raw_holo = self.torch_module(observation)
        if raw_holo.shape != (self.num_envs, HOLOSOMA_ACTION_DIM):
            raise RuntimeError(f"Torch model returned action with shape {raw_holo.shape}")
        if not torch.isfinite(raw_holo).all():
            raise FloatingPointError("HoloSoma Torch action contains NaN/Inf")
        self.last_policy_action.copy_(raw_holo)

        target_holo = self._default_holo + HOLOSOMA_ACTION_SCALE_RAD * raw_holo
        target_action_order = target_holo.index_select(1, self._holo_to_action)
        isaac_action = (target_action_order - self._isaac_offset) / self._isaac_scale
        if isaac_action.shape != (self.num_envs, HOLOSOMA_ACTION_DIM):
            raise RuntimeError(f"mapped Isaac action has shape {tuple(isaac_action.shape)}")
        if not torch.isfinite(isaac_action).all():
            raise FloatingPointError("mapped Isaac action contains NaN/Inf")
        return isaac_action

    def manifest(self) -> dict[str, object]:
        return {
            "adapter": "HoloSomaFastSacIsaacPolicy",
            "inference_backend": "frozen_torch_reconstruction_of_pinned_onnx",
            "source_model_sha256": self.sha256,
            "source_observation_dimension": HOLOSOMA_OBSERVATION_DIM,
            "source_action_dimension": HOLOSOMA_ACTION_DIM,
            "source_observation_terms_sorted": list(HOLOSOMA_OBSERVATION_TERMS),
            "source_observation_slices": {
                key: list(value) for key, value in HOLOSOMA_OBSERVATION_SLICES.items()
            },
            "source_joint_order": list(HOLOSOMA_G1_29DOF_JOINT_ORDER),
            "source_default_joint_position": list(HOLOSOMA_DEFAULT_JOINT_POSITION),
            "source_policy_dt_s": HOLOSOMA_POLICY_DT_S,
            "source_gait_period_s": HOLOSOMA_GAIT_PERIOD_S,
            "source_action_scale_rad": HOLOSOMA_ACTION_SCALE_RAD,
            "requested_command_x_m_s": self.command_x_m_s,
            "phase_update": "increment_before_inference_matching_HoloSoma_run_loop",
            "previous_action_semantics": "previous_raw_ONNX_action",
            "action_mapping": (
                "q_target_holo=q_default_holo+0.25*onnx_action; "
                "isaac_raw=(q_target_reordered-isaac_default_offset)/isaac_scale"
            ),
            "uses_hall": False,
            "uses_force_contact_friction_mu_slip_or_stage": False,
            "onnx_kp_metadata": list(self.onnx_kp),
            "onnx_kd_metadata": list(self.onnx_kd),
            "actuator_note": (
                "ONNX KP/KD are recorded but not applied; both compared policies use "
                "the identical existing Isaac articulation and actuator dynamics"
            ),
        }
