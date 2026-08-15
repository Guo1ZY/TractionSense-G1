"""Isaac privileged Teacher adapter, isolated from deployable Student code."""

from __future__ import annotations

import torch

from unitree_rl_lab.traction.isaac_observations import (
    _robot_mass_kg,
    canonical_current_proprio,
    canonical_privileged_traction,
)

from .isaac_observations import (
    IsaacTorqueEstimatorCfg,
    _IsaacTorqueEstimatorState,
    torque_traction_packet,
)
from .teacher_schema import TORQUE_TEACHER_FRAME_DIM


def torque_teacher_observation(
    env,
    left_sensor_cfg,
    right_sensor_cfg,
    asset_cfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Return 96 proprio + command + 149 privileged values.

    Ground-truth force/friction/contact fields enter through
    ``canonical_privileged_traction`` and are restricted to this Teacher term.
    """
    if not hasattr(env, "_isaac_torque_traction_state"):
        env._isaac_torque_traction_state = _IsaacTorqueEstimatorState(
            env, IsaacTorqueEstimatorCfg(asset_name=asset_cfg.name)
        )
    packet = torque_traction_packet(env)
    mass = _robot_mass_kg(env, asset_cfg)
    privilege = torch.cat((
        canonical_privileged_traction(env, left_sensor_cfg, right_sensor_cfg, asset_cfg),
        packet.analytical_force_local_n / (mass * 9.81),
        packet.contact_probability,
        packet.force_confidence,
        packet.residual_norm_nm * 0.05,
        packet.condition_score,
    ), dim=-1)
    value = torch.cat((
        canonical_current_proprio(env, asset_cfg, command_name),
        env.command_manager.get_command(command_name),
        privilege,
    ), dim=-1)
    if value.shape[-1] != TORQUE_TEACHER_FRAME_DIM:
        raise RuntimeError(f"torque Teacher frame changed to {value.shape[-1]}")
    return torch.nan_to_num(value)
