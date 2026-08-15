"""Pure-Torch contact-patch distribution for the Scheme-A Hall sole.

This module intentionally has no Isaac Sim imports.  Isaac's detailed normal
contact and friction buffers are ragged; :func:`indexed_buffer_indices`
unpacks their ``count/start`` tables, while
:func:`distribute_point_forces_to_hall_sites` maps each point force into the
canonical foot frame and distributes it over the Hall sites.

All positions are metres and all forces are newtons.  The output is simulation
mechanics ground truth, not a force reconstructed from Hall measurements.
"""

from __future__ import annotations

import torch


def indexed_buffer_indices(
    counts: torch.Tensor,
    starts: torch.Tensor,
    *,
    buffer_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return group-row and flat-buffer indices for a ragged PhysX buffer.

    ``counts`` and ``starts`` must have identical ``[sensor, filter]`` shape.
    Empty groups produce no indices.  Invalid or out-of-bounds metadata raises
    instead of silently using aggregate data; this is the detailed adapter's
    fail-closed boundary.
    """

    if counts.ndim != 2 or starts.shape != counts.shape:
        raise ValueError(
            "counts and starts must have identical [sensor,filter] shape, got "
            f"{tuple(counts.shape)} and {tuple(starts.shape)}"
        )
    if buffer_length < 0:
        raise ValueError("buffer_length must be non-negative")
    if counts.device != starts.device:
        raise ValueError("counts and starts must be on the same device")
    if counts.dtype.is_floating_point or starts.dtype.is_floating_point:
        raise TypeError("counts and starts must use integer dtypes")

    flat_counts = counts.reshape(-1).to(dtype=torch.long)
    flat_starts = starts.reshape(-1).to(dtype=torch.long)
    if torch.any(flat_counts < 0) or torch.any(flat_starts < 0):
        raise ValueError("counts and starts must be non-negative")
    if torch.any(flat_starts + flat_counts > buffer_length):
        raise ValueError("contact count/start metadata exceeds the raw buffer")

    total = int(flat_counts.sum().item())
    if total == 0:
        empty = torch.empty(0, device=counts.device, dtype=torch.long)
        return empty, empty

    group_ids = torch.repeat_interleave(
        torch.arange(flat_counts.numel(), device=counts.device),
        flat_counts,
        output_size=total,
    )
    packed_starts = flat_counts.cumsum(0) - flat_counts
    offsets = (
        torch.arange(total, device=counts.device)
        - torch.repeat_interleave(packed_starts, flat_counts, output_size=total)
    )
    buffer_indices = flat_starts[group_ids] + offsets
    # A group row is a sensor/filter pair.  The filter component is irrelevant
    # after its samples have been gathered, so return the sensor row only.
    sensor_rows = torch.div(group_ids, counts.shape[1], rounding_mode="floor")
    return sensor_rows, buffer_indices


def sum_vectors_by_index(
    vectors: torch.Tensor,
    indices: torch.Tensor,
    *,
    output_count: int,
) -> torch.Tensor:
    """Sum ``[K,3]`` vectors into ``output_count`` indexed rows."""

    if vectors.ndim != 2 or vectors.shape[-1] != 3:
        raise ValueError(f"vectors must have shape [K,3], got {tuple(vectors.shape)}")
    if indices.ndim != 1 or indices.shape[0] != vectors.shape[0]:
        raise ValueError("indices must have shape [K] matching vectors")
    if indices.device != vectors.device:
        raise ValueError("indices and vectors must be on the same device")
    if output_count < 1:
        raise ValueError("output_count must be positive")
    if indices.dtype != torch.long:
        indices = indices.to(dtype=torch.long)
    if indices.numel() and (torch.any(indices < 0) or torch.any(indices >= output_count)):
        raise ValueError("indices are outside the requested output range")
    result = torch.zeros((output_count, 3), device=vectors.device, dtype=vectors.dtype)
    if vectors.numel():
        result.index_add_(0, indices, vectors)
    return result


def distribute_point_forces_to_hall_sites(
    *,
    num_envs: int,
    hall_positions_f: torch.Tensor,
    foot_positions_w: torch.Tensor,
    foot_rotations_w: torch.Tensor,
    point_forces_w: torch.Tensor,
    contact_points_w: torch.Tensor,
    contact_env_indices: torch.Tensor,
    contact_foot_indices: torch.Tensor,
    spread_sigma_f: torch.Tensor | float,
) -> torch.Tensor:
    """Distribute detailed world-frame point forces to local Hall sites.

    Args:
        num_envs: Batch environment count ``N``.
        hall_positions_f: Hall centres in each foot frame, shape ``[2,S,3]``.
        foot_positions_w: Foot-frame origins in world, shape ``[N,2,3]``.
        foot_rotations_w: Proper local-to-world rotations, shape
            ``[N,2,3,3]``.
        point_forces_w: Detailed point forces in newtons, shape ``[K,3]``.
        contact_points_w: Corresponding world points in metres, shape
            ``[K,3]``.
        contact_env_indices: Environment index for each sample, shape ``[K]``.
        contact_foot_indices: Foot index (0 left, 1 right), shape ``[K]``.
        spread_sigma_f: Gaussian length scale in metres.  It may be scalar,
            ``[N,2]``, or ``[N,2,1]``.

    Returns:
        Foot-local force tensor ``[N,2,S,3]``.  Each point's Gaussian weights
        are a stable softmax over sites, so summing over ``S`` exactly preserves
        its local-frame force up to floating-point reduction error.

    Raises:
        ValueError: For an invalid shape, index, rotation, sigma, or non-finite
            referenced sample.  Detailed mode deliberately fails closed.
    """

    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if hall_positions_f.ndim != 3 or hall_positions_f.shape[0] != 2 or hall_positions_f.shape[-1] != 3:
        raise ValueError(f"hall_positions_f must be [2,S,3], got {tuple(hall_positions_f.shape)}")
    num_sites = hall_positions_f.shape[1]
    if num_sites < 1:
        raise ValueError("at least one Hall site is required")
    if foot_positions_w.shape != (num_envs, 2, 3):
        raise ValueError(f"foot_positions_w must be [{num_envs},2,3]")
    if foot_rotations_w.shape != (num_envs, 2, 3, 3):
        raise ValueError(f"foot_rotations_w must be [{num_envs},2,3,3]")
    if point_forces_w.ndim != 2 or point_forces_w.shape[-1] != 3:
        raise ValueError(f"point_forces_w must be [K,3], got {tuple(point_forces_w.shape)}")
    if contact_points_w.shape != point_forces_w.shape:
        raise ValueError("contact_points_w must match point_forces_w")
    sample_count = point_forces_w.shape[0]
    if contact_env_indices.shape != (sample_count,) or contact_foot_indices.shape != (sample_count,):
        raise ValueError("contact indices must each have shape [K]")

    device = foot_positions_w.device
    dtype = foot_positions_w.dtype
    tensors = (
        hall_positions_f,
        foot_rotations_w,
        point_forces_w,
        contact_points_w,
        contact_env_indices,
        contact_foot_indices,
    )
    if any(value.device != device for value in tensors):
        raise ValueError("all contact-distribution tensors must share one device")
    if any(value.dtype != dtype for value in tensors[:4]):
        raise ValueError("all floating contact-distribution tensors must share one dtype")
    if not dtype.is_floating_point:
        raise TypeError("contact geometry and forces must use a floating dtype")

    env_indices = contact_env_indices.to(dtype=torch.long)
    foot_indices = contact_foot_indices.to(dtype=torch.long)
    if sample_count:
        if torch.any(env_indices < 0) or torch.any(env_indices >= num_envs):
            raise ValueError("contact environment index is out of range")
        if torch.any(foot_indices < 0) or torch.any(foot_indices > 1):
            raise ValueError("contact foot index must be 0 (left) or 1 (right)")
    if not torch.isfinite(foot_positions_w).all() or not torch.isfinite(foot_rotations_w).all():
        raise ValueError("foot poses must be finite")
    if not torch.isfinite(hall_positions_f).all():
        raise ValueError("Hall positions must be finite")

    if isinstance(spread_sigma_f, torch.Tensor):
        sigma = spread_sigma_f.to(device=device, dtype=dtype)
    else:
        sigma = torch.as_tensor(spread_sigma_f, device=device, dtype=dtype)
    if sigma.ndim == 0:
        sigma = sigma.expand(num_envs, 2)
    elif sigma.shape == (num_envs, 2, 1):
        sigma = sigma.squeeze(-1)
    elif sigma.shape != (num_envs, 2):
        raise ValueError("spread_sigma_f must be scalar, [N,2], or [N,2,1]")
    if not torch.isfinite(sigma).all() or torch.any(sigma <= 0.0):
        raise ValueError("spread_sigma_f must be finite and strictly positive")

    output = torch.zeros((num_envs, 2, num_sites, 3), device=device, dtype=dtype)
    if sample_count == 0:
        return output
    if not torch.isfinite(point_forces_w).all() or not torch.isfinite(contact_points_w).all():
        raise ValueError("referenced detailed contact samples must be finite")

    sample_rotations = foot_rotations_w[env_indices, foot_indices]
    relative_points_w = contact_points_w - foot_positions_w[env_indices, foot_indices]
    contact_points_f = torch.einsum("kji,kj->ki", sample_rotations, relative_points_w)
    contact_forces_f = torch.einsum("kji,kj->ki", sample_rotations, point_forces_w)

    site_xy = hall_positions_f[foot_indices, :, :2]
    squared_distance = torch.sum(
        torch.square(site_xy - contact_points_f[:, None, :2]), dim=-1
    )
    sample_sigma = sigma[env_indices, foot_indices]
    logits = -0.5 * squared_distance / torch.square(sample_sigma[:, None])
    weights = torch.softmax(logits, dim=-1)

    contributions = weights[..., None] * contact_forces_f[:, None, :]
    site_indices = torch.arange(num_sites, device=device).view(1, -1)
    destination = (
        (env_indices * 2 + foot_indices).view(-1, 1) * num_sites + site_indices
    )
    output.view(-1, 3).index_add_(
        0,
        destination.reshape(-1),
        contributions.reshape(-1, 3),
    )
    return output


__all__ = [
    "distribute_point_forces_to_hall_sites",
    "indexed_buffer_indices",
    "sum_vectors_by_index",
]
