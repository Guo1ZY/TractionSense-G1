from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def lateral_motion_feedback(
    env: ManagerBasedRLEnv,
    asset_name: str = "robot",
    lateral_velocity_clip: float = 1.5,
    heading_error_clip: float = 1.0,
) -> torch.Tensor:
    """Return deployable lateral velocity and relative heading feedback.

    The heading reference is latched at every episode reset.  It is invariant
    to randomized world yaw and can be reproduced on hardware by latching the
    IMU orientation when a straight-walking episode starts.  Body lateral
    velocity is privileged during Teacher training; the Student later receives
    its contact-aided estimate.
    """

    asset = env.scene[asset_name]
    quat = asset.data.root_quat_w
    yaw = torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (torch.square(quat[:, 2]) + torch.square(quat[:, 3])),
    )

    if (
        not hasattr(env, "motion_feedback_initial_yaw")
        or env.motion_feedback_initial_yaw.shape[0] != env.num_envs
    ):
        env.motion_feedback_initial_yaw = yaw.clone()

    reset = env.episode_length_buf <= 1
    if reset.any():
        env.motion_feedback_initial_yaw[reset] = yaw[reset]

    heading_error = torch.atan2(
        torch.sin(yaw - env.motion_feedback_initial_yaw),
        torch.cos(yaw - env.motion_feedback_initial_yaw),
    )
    lateral_velocity = asset.data.root_lin_vel_b[:, 1]
    return torch.stack(
        (
            torch.clamp(
                lateral_velocity,
                -lateral_velocity_clip,
                lateral_velocity_clip,
            ),
            torch.clamp(
                heading_error,
                -heading_error_clip,
                heading_error_clip,
            ),
        ),
        dim=-1,
    )
