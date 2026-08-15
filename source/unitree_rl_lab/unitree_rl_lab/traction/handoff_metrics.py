"""Pure tensor helpers for the Hall high-momentum handoff evaluator.

These functions deliberately know nothing about simulator friction, contact
force, or slip.  They convert the deployable robot pose/velocity state into
diagnostic quantities used *after* a policy rollout.  Keeping them independent
of Isaac Sim makes the safety calculations directly unit-testable.
"""

from __future__ import annotations

import torch


def validate_wxyz_quaternion(quaternion: torch.Tensor) -> None:
    """Validate a tensor of scalar-first quaternions."""

    if quaternion.ndim < 1 or quaternion.shape[-1] != 4:
        raise ValueError(
            "quaternion must have final dimension 4 in wxyz order, got "
            f"{tuple(quaternion.shape)}"
        )


def body_forward_axis_world(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Return the body ``+x`` unit vector expressed in world coordinates."""

    validate_wxyz_quaternion(quaternion_wxyz)
    quaternion = quaternion_wxyz / torch.linalg.vector_norm(
        quaternion_wxyz, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y.square() + z.square()),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ),
        dim=-1,
    )


def roll_pitch_from_wxyz(quaternion_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return roll and pitch in radians for scalar-first quaternions."""

    validate_wxyz_quaternion(quaternion_wxyz)
    quaternion = quaternion_wxyz / torch.linalg.vector_norm(
        quaternion_wxyz, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    w, x, y, z = quaternion.unbind(dim=-1)
    roll = torch.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x.square() + y.square()),
    )
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    return roll, pitch


def one_second_deceleration(
    initial_forward_speed: torch.Tensor,
    forward_speed_after_one_second: torch.Tensor,
    valid_mask: torch.Tensor,
    speed_limit: float,
) -> dict[str, float]:
    """Summarize forward-speed reduction for non-fallen finite environments.

    Args:
        initial_forward_speed: Body-forward speed immediately after the impulse.
        forward_speed_after_one_second: Body-forward speed one second later.
        valid_mask: Environments that remained valid and did not fall.
        speed_limit: Required absolute body-forward speed after one second.
    """

    if initial_forward_speed.shape != forward_speed_after_one_second.shape:
        raise ValueError("initial and one-second speed tensors must have the same shape")
    if valid_mask.shape != initial_forward_speed.shape:
        raise ValueError("valid_mask must have the same shape as the speed tensors")
    if speed_limit < 0.0:
        raise ValueError("speed_limit must be non-negative")

    finite = torch.isfinite(initial_forward_speed) & torch.isfinite(
        forward_speed_after_one_second
    )
    selected = valid_mask.bool() & finite
    if not torch.any(selected):
        nan = float("nan")
        return {
            "valid_count": 0.0,
            "initial_mean_m_s": nan,
            "after_mean_m_s": nan,
            "reduction_mean_m_s": nan,
            "pass_fraction": 0.0,
        }

    initial = initial_forward_speed[selected]
    after = forward_speed_after_one_second[selected]
    return {
        "valid_count": float(selected.sum().item()),
        "initial_mean_m_s": float(initial.mean().item()),
        "after_mean_m_s": float(after.mean().item()),
        "reduction_mean_m_s": float((initial - after).mean().item()),
        "pass_fraction": float((torch.abs(after) <= speed_limit).float().mean().item()),
    }
