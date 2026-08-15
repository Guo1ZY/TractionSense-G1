"""Pure-Torch contact-point slip metrics for simulator-only supervision.

The Hall actor never consumes these quantities.  They are ground-truth
outcomes used by offline evaluation/label construction only.  In particular,
this module does not infer force or friction from Hall magnetic measurements.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch


CONTACT_POINT_TANGENTIAL_SLIP_SCHEMA = (
    "static_ground_contact_point_tangential_speed.v1"
)
CONTACT_POINT_TANGENTIAL_SLIP_KEY = "contact_point_tangent_slip"
CONTACT_POINT_TANGENTIAL_SLIP_VALID_KEY = (
    "contact_point_tangent_slip_valid"
)
LEGACY_LINK_ORIGIN_PLANAR_SLIP_KEY = "legacy_link_origin_planar_slip"
CONTACT_POINT_TANGENTIAL_SLIP_FORMULA = (
    "v_point_w=v_com_w+cross(omega_w,contact_pos_w-com_pos_w);"
    "v_tangent_w=v_point_w-dot(v_point_w,n_hat_w)*n_hat_w;"
    "slip=sum(norm(v_tangent_w)*norm(F_normal_w))/sum(norm(F_normal_w))"
    " over finite left/right dedicated filtered contacts with norm(F_normal_w)>5N;"
    " static ground velocity=0"
)


class ContactPointTangentialSlip(NamedTuple):
    """Force-weighted static-ground contact-point tangential speed.

    ``speed_per_env`` has shape ``[num_envs]`` and ``speed_per_foot`` has
    shape ``[num_envs, 2]`` in left/right order.  A zero in either speed
    tensor is meaningful only where its corresponding validity flag is true.
    No-contact and non-finite rows return zero with validity false.
    """

    speed_per_env: torch.Tensor
    valid_per_env: torch.Tensor
    speed_per_foot: torch.Tensor
    valid_per_foot: torch.Tensor
    normal_load_per_foot_n: torch.Tensor


class LegacyLinkOriginPlanarSlip(NamedTuple):
    """Historical link/COM-origin XY speed retained only for diagnostics."""

    speed_per_env: torch.Tensor
    valid_per_env: torch.Tensor


def _require_body_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.ndim != 3 or value.shape[1:] != (2, 3):
        raise ValueError(
            f"{name} must have shape [num_envs,2,3] in left/right order; "
            f"got {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating dtype")


def _require_contact_pair(
    foot_name: str,
    contact_pos_w: torch.Tensor,
    normal_force_w: torch.Tensor,
    *,
    num_envs: int,
    device: torch.device,
) -> None:
    for name, value in (
        (f"{foot_name}_contact_pos_w", contact_pos_w),
        (f"{foot_name}_normal_force_w", normal_force_w),
    ):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
        if (
            value.ndim != 4
            or value.shape[0] != num_envs
            or value.shape[1] != 1
            or value.shape[2] < 1
            or value.shape[3] != 3
        ):
            raise ValueError(
                f"{name} must have shape [num_envs,1,num_filters,3] with "
                f"num_filters>=1; got {tuple(value.shape)}"
            )
        if not value.is_floating_point():
            raise ValueError(f"{name} must use a floating dtype")
        if value.device != device:
            raise ValueError(
                f"{name} device {value.device} does not match body device {device}"
            )
    if contact_pos_w.shape != normal_force_w.shape:
        raise ValueError(
            f"{foot_name} contact positions and normal forces must have the "
            f"same shape; got {tuple(contact_pos_w.shape)} and "
            f"{tuple(normal_force_w.shape)}"
        )


def static_ground_contact_point_tangential_speed(
    body_com_pos_w: torch.Tensor,
    body_com_lin_vel_w: torch.Tensor,
    body_com_ang_vel_w: torch.Tensor,
    contact_pos_w_by_foot: Sequence[torch.Tensor],
    normal_force_w_by_foot: Sequence[torch.Tensor],
    min_normal_force_n: float = 5.0,
) -> ContactPointTangentialSlip:
    """Compute rigid-body contact-point tangential speed on static ground.

    Args:
        body_com_pos_w: Foot body COM positions, ``[N,2,3]`` (left, right).
        body_com_lin_vel_w: Foot body COM linear velocities, ``[N,2,3]``.
        body_com_ang_vel_w: Foot body angular velocities, ``[N,2,3]``.
        contact_pos_w_by_foot: Two dedicated ContactSensor position tensors,
            each ``[N,1,M,3]``.  The two feet may have different ``M``.
        normal_force_w_by_foot: Matching filtered normal-force tensors.
        min_normal_force_n: A pair is active only above this load.

    Returns:
        A :class:`ContactPointTangentialSlip` result.  Contact pairs containing
        NaN/Inf are ignored.  Environments with no valid loaded pair return
        zero and ``valid_per_env=False``.

    Raises:
        ValueError: On a shape, dtype, device, or left/right ordering contract
        violation.  Rejecting malformed tensors is deliberate fail-closed
        behavior: a bad schema must never be interpreted as zero slip.
    """

    for name, value in (
        ("body_com_pos_w", body_com_pos_w),
        ("body_com_lin_vel_w", body_com_lin_vel_w),
        ("body_com_ang_vel_w", body_com_ang_vel_w),
    ):
        _require_body_tensor(name, value)
    if not (
        body_com_pos_w.shape
        == body_com_lin_vel_w.shape
        == body_com_ang_vel_w.shape
    ):
        raise ValueError("body COM position/linear/angular tensors must match")
    if not (
        body_com_pos_w.device
        == body_com_lin_vel_w.device
        == body_com_ang_vel_w.device
    ):
        raise ValueError("body COM position/linear/angular devices must match")
    if not (
        body_com_pos_w.dtype
        == body_com_lin_vel_w.dtype
        == body_com_ang_vel_w.dtype
    ):
        raise ValueError("body COM position/linear/angular dtypes must match")
    if len(contact_pos_w_by_foot) != 2 or len(normal_force_w_by_foot) != 2:
        raise ValueError("exactly two dedicated contact tensors are required: left, right")
    if not torch.isfinite(
        torch.as_tensor(min_normal_force_n, dtype=body_com_pos_w.dtype)
    ) or min_normal_force_n < 0.0:
        raise ValueError("min_normal_force_n must be finite and non-negative")

    num_envs = int(body_com_pos_w.shape[0])
    device = body_com_pos_w.device
    dtype = body_com_pos_w.dtype
    numerator_by_foot: list[torch.Tensor] = []
    normal_load_by_foot: list[torch.Tensor] = []

    for foot_index, foot_name in enumerate(("left", "right")):
        contact_pos_w = contact_pos_w_by_foot[foot_index]
        normal_force_w = normal_force_w_by_foot[foot_index]
        _require_contact_pair(
            foot_name,
            contact_pos_w,
            normal_force_w,
            num_envs=num_envs,
            device=device,
        )
        if contact_pos_w.dtype != dtype or normal_force_w.dtype != dtype:
            raise ValueError(
                f"{foot_name} contact tensors must match body dtype {dtype}"
            )

        # Dedicated sensors contain exactly one foot body.  Remove that
        # singleton while preserving every ground/patch filter independently.
        point = contact_pos_w[:, 0]
        force = normal_force_w[:, 0]
        body_pos = body_com_pos_w[:, foot_index, None, :]
        body_lin = body_com_lin_vel_w[:, foot_index, None, :]
        body_ang = body_com_ang_vel_w[:, foot_index, None, :]

        body_finite = (
            torch.isfinite(body_pos).all(dim=-1)
            & torch.isfinite(body_lin).all(dim=-1)
            & torch.isfinite(body_ang).all(dim=-1)
        )
        pair_finite = (
            torch.isfinite(point).all(dim=-1)
            & torch.isfinite(force).all(dim=-1)
            & body_finite
        )
        safe_point = torch.where(
            pair_finite[..., None], point, torch.zeros_like(point)
        )
        safe_force = torch.where(
            pair_finite[..., None], force, torch.zeros_like(force)
        )
        safe_body_pos = torch.nan_to_num(body_pos)
        safe_body_lin = torch.nan_to_num(body_lin)
        safe_body_ang = torch.nan_to_num(body_ang)
        normal_force_n = torch.linalg.vector_norm(safe_force, dim=-1)
        active = pair_finite & (normal_force_n > float(min_normal_force_n))
        normal_hat = safe_force / normal_force_n.clamp_min(
            torch.finfo(dtype).eps
        )[..., None]
        radius_w = safe_point - safe_body_pos
        point_velocity_w = safe_body_lin + torch.linalg.cross(
            safe_body_ang.expand_as(radius_w), radius_w, dim=-1
        )
        tangent_velocity_w = point_velocity_w - (
            point_velocity_w * normal_hat
        ).sum(dim=-1, keepdim=True) * normal_hat
        tangent_speed = torch.linalg.vector_norm(tangent_velocity_w, dim=-1)
        active = active & torch.isfinite(tangent_speed)
        weight = torch.where(active, normal_force_n, torch.zeros_like(normal_force_n))
        numerator_by_foot.append((torch.nan_to_num(tangent_speed) * weight).sum(dim=1))
        normal_load_by_foot.append(weight.sum(dim=1))

    numerator = torch.stack(numerator_by_foot, dim=1)
    normal_load = torch.stack(normal_load_by_foot, dim=1)
    valid_per_foot = normal_load > 0.0
    speed_per_foot = torch.where(
        valid_per_foot,
        numerator / normal_load.clamp_min(torch.finfo(dtype).eps),
        torch.zeros_like(numerator),
    )
    total_load = normal_load.sum(dim=1)
    valid_per_env = total_load > 0.0
    speed_per_env = torch.where(
        valid_per_env,
        numerator.sum(dim=1) / total_load.clamp_min(torch.finfo(dtype).eps),
        torch.zeros_like(total_load),
    )
    return ContactPointTangentialSlip(
        speed_per_env=speed_per_env,
        valid_per_env=valid_per_env,
        speed_per_foot=speed_per_foot,
        valid_per_foot=valid_per_foot,
        normal_load_per_foot_n=normal_load,
    )


def legacy_link_origin_planar_speed(
    body_com_lin_vel_w: torch.Tensor,
    normal_force_w: torch.Tensor,
    min_vertical_force_n: float = 5.0,
) -> LegacyLinkOriginPlanarSlip:
    """Reproduce the former ankle/link-origin XY-speed proxy exactly.

    This function exists only so newly collected NPZ files retain an explicit
    diagnostic bridge to old results.  It must not be used for prospective
    contact-point slip labels.
    """

    _require_body_tensor("body_com_lin_vel_w", body_com_lin_vel_w)
    _require_body_tensor("normal_force_w", normal_force_w)
    if body_com_lin_vel_w.shape != normal_force_w.shape:
        raise ValueError("legacy velocity and normal-force shapes must match")
    if body_com_lin_vel_w.device != normal_force_w.device:
        raise ValueError("legacy velocity and normal-force devices must match")
    if not torch.isfinite(
        torch.as_tensor(min_vertical_force_n, dtype=body_com_lin_vel_w.dtype)
    ) or min_vertical_force_n < 0.0:
        raise ValueError("min_vertical_force_n must be finite and non-negative")

    finite = (
        torch.isfinite(body_com_lin_vel_w).all(dim=-1)
        & torch.isfinite(normal_force_w).all(dim=-1)
    )
    contact = finite & (
        torch.abs(torch.nan_to_num(normal_force_w[..., 2]))
        > float(min_vertical_force_n)
    )
    speed = torch.linalg.vector_norm(
        torch.nan_to_num(body_com_lin_vel_w[..., :2]), dim=-1
    )
    weight = contact.to(dtype=speed.dtype)
    count = weight.sum(dim=1)
    valid = count > 0.0
    per_env = torch.where(
        valid,
        (speed * weight).sum(dim=1) / count.clamp_min(1.0),
        torch.zeros_like(count),
    )
    return LegacyLinkOriginPlanarSlip(per_env, valid)
