"""Training-only supervision and anti-degenerate traction rewards."""

from __future__ import annotations

import torch

from unitree_rl_lab.traction.isaac_observations import _true_force_local_n

from .isaac_observations import torque_traction_packet


def force_estimation_error(env, left_sensor_cfg, right_sensor_cfg, asset_cfg) -> torch.Tensor:
    """Normalized analytical-vs-ContactSensor force error (training only)."""
    truth = _true_force_local_n(env, left_sensor_cfg, right_sensor_cfg, asset_cfg)
    estimate = torque_traction_packet(env).analytical_force_local_n
    mass = env.scene[asset_cfg.name].root_physx_view.get_masses().sum(dim=1).to(env.device)
    return torch.square((estimate - truth) / (mass[:, None] * 9.81)).mean(dim=1)


def contact_estimation_error(env, left_sensor_cfg, right_sensor_cfg, asset_cfg, threshold_n: float = 10.0) -> torch.Tensor:
    truth = _true_force_local_n(env, left_sensor_cfg, right_sensor_cfg, asset_cfg).reshape(env.num_envs, 2, 3)
    label = (truth[..., 2].abs() > threshold_n).float()
    return torch.square(torque_traction_packet(env).contact_probability - label).mean(dim=1)


def estimated_traction_utilization_penalty(env, warning: float = 0.65) -> torch.Tensor:
    utilization = torque_traction_packet(env).traction_utilization
    contact = torque_traction_packet(env).contact_probability
    return (torch.square((utilization - warning).clamp_min(0.0)) * contact).mean(dim=1)


def force_estimator_temporal_consistency(env) -> torch.Tensor:
    packet = torque_traction_packet(env)
    # Residual is used rather than truth, so the regularizer stays deployable.
    return torch.square(packet.residual_norm_nm / 50.0).mean(dim=1)


def ground_truth_tangential_push(env, left_sensor_cfg, right_sensor_cfg, asset_cfg) -> torch.Tensor:
    truth = _true_force_local_n(env, left_sensor_cfg, right_sensor_cfg, asset_cfg).reshape(env.num_envs, 2, 3)
    mass = env.scene[asset_cfg.name].root_physx_view.get_masses().sum(dim=1).to(env.device)
    return torch.square(torch.linalg.vector_norm(truth[..., :2], dim=-1) / (mass[:, None] * 9.81)).mean(dim=1)

