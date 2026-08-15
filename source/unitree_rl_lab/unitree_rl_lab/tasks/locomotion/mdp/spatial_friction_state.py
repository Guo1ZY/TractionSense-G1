# Copyright (c) 2026 local Hall-foot extension.
# SPDX-License-Identifier: BSD-3-Clause
"""Pure tensor state machine for the physical high--low--high course.

This module intentionally depends only on PyTorch.  The simulator adapter in
``spatial_friction.py`` supplies filtered contact evidence, while this module
owns the transition rules and can therefore be unit-tested without starting
Isaac Sim or allocating a GPU.
"""

from __future__ import annotations

import torch


SPATIAL_HIGH_START = 0
SPATIAL_LOW = 1
SPATIAL_HIGH_END = 2


def stratified_high_patch_reset_x(
    band_sample: torch.Tensor,
    within_band_sample: torch.Tensor,
    x_bands: tuple[tuple[float, float], ...],
    band_probabilities: tuple[float, ...],
    low_boundary_x: float = 0.0,
    minimum_high_margin: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map two uniform samples to a safe, stratified HighStart reset position.

    This helper is deliberately pure Torch so the sampling contract can be
    tested without Isaac Sim.  Every band must lie strictly before the first
    LOW collider boundary.  Randomizing distance to that boundary de-synchronizes
    parallel rollouts, while the course state still changes only after filtered
    LOW contact.  Neither the sampled band nor root position is an actor input.

    Args:
        band_sample: Uniform samples in ``[0, 1]`` selecting a distance band.
        within_band_sample: Uniform samples in ``[0, 1]`` selecting x in a band.
        x_bands: Ordered, non-overlapping ``(min_x, max_x)`` HighStart bands.
        band_probabilities: Non-negative probability for each band.
        low_boundary_x: Local x coordinate of the first LOW boundary.
        minimum_high_margin: Required clearance between every reset and LOW.

    Returns:
        Sampled local x positions and the selected integer band indices.
    """

    if band_sample.shape != within_band_sample.shape:
        raise ValueError("band_sample and within_band_sample must have the same shape")
    if not band_sample.is_floating_point() or not within_band_sample.is_floating_point():
        raise TypeError("stratified reset samples must use floating dtypes")
    if len(x_bands) == 0 or len(x_bands) != len(band_probabilities):
        raise ValueError("x_bands and band_probabilities must have equal non-zero length")
    if minimum_high_margin < 0.0:
        raise ValueError("minimum_high_margin must be non-negative")

    previous_max: float | None = None
    for lower, upper in x_bands:
        lower = float(lower)
        upper = float(upper)
        if not lower <= upper:
            raise ValueError(f"invalid spatial reset band {(lower, upper)}")
        if previous_max is not None and lower < previous_max:
            raise ValueError("spatial reset bands must be ordered and non-overlapping")
        if upper > float(low_boundary_x) - float(minimum_high_margin):
            raise ValueError("spatial reset band is too close to or inside LOW")
        previous_max = upper

    probabilities = torch.as_tensor(
        band_probabilities,
        device=band_sample.device,
        dtype=band_sample.dtype,
    )
    if not torch.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError("band probabilities must be finite and non-negative")
    probability_sum = probabilities.sum()
    if not bool((probability_sum > 0.0).item()):
        raise ValueError("at least one band probability must be positive")
    probabilities = probabilities / probability_sum
    cumulative = torch.cumsum(probabilities, dim=0)

    # Clamp adversarial/static-test inputs as well as the vanishingly unlikely
    # endpoint 1.0. searchsorted can otherwise return len(x_bands).
    selector = torch.clamp(
        torch.nan_to_num(band_sample, nan=0.0, posinf=1.0, neginf=0.0),
        min=0.0,
        max=1.0,
    )
    band_index = torch.searchsorted(cumulative, selector, right=False)
    band_index = torch.clamp(band_index, max=len(x_bands) - 1).to(torch.long)

    bounds = torch.as_tensor(x_bands, device=band_sample.device, dtype=band_sample.dtype)
    lower = bounds[:, 0][band_index]
    upper = bounds[:, 1][band_index]
    within = torch.clamp(
        torch.nan_to_num(within_band_sample, nan=0.5, posinf=1.0, neginf=0.0),
        min=0.0,
        max=1.0,
    )
    return lower + (upper - lower) * within, band_index


def update_low_capture_timing(
    previous_stage: torch.Tensor,
    current_stage: torch.Tensor,
    episode_step: torch.Tensor,
    forward_speed: torch.Tensor,
    previous_entry_step: torch.Tensor,
    previous_entry_speed: torch.Tensor,
    previous_elapsed_s: torch.Tensor,
    reset: torch.Tensor,
    control_dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Latch the first real LOW contact and measure causal capture time.

    The simulator adapter supplies ``current_stage`` only after filtered LOW
    collider contact.  Entry time/speed are latched once per episode and never
    inferred from root position or a friction label.  Elapsed time remains
    frozen after leaving LOW so evaluation can inspect the completed segment.
    """

    tensors = (
        current_stage,
        episode_step,
        forward_speed,
        previous_entry_step,
        previous_entry_speed,
        previous_elapsed_s,
        reset,
    )
    if any(value.shape != previous_stage.shape for value in tensors):
        raise ValueError("all low-capture timing tensors must have the same shape")
    if previous_stage.dtype == torch.bool or previous_stage.is_floating_point():
        raise TypeError("previous_stage must use an integer dtype")
    if current_stage.dtype == torch.bool or current_stage.is_floating_point():
        raise TypeError("current_stage must use an integer dtype")
    if episode_step.dtype == torch.bool or episode_step.is_floating_point():
        raise TypeError("episode_step must use an integer dtype")
    if previous_entry_step.dtype == torch.bool or previous_entry_step.is_floating_point():
        raise TypeError("previous_entry_step must use an integer dtype")
    for name, value in (
        ("forward_speed", forward_speed),
        ("previous_entry_speed", previous_entry_speed),
        ("previous_elapsed_s", previous_elapsed_s),
    ):
        if not value.is_floating_point():
            raise TypeError(f"{name} must use a floating dtype")
    if reset.dtype != torch.bool:
        raise TypeError("reset must use bool dtype")
    if control_dt <= 0.0:
        raise ValueError("control_dt must be positive")

    entry_step = previous_entry_step.clone()
    entry_speed = previous_entry_speed.clone()
    elapsed_s = previous_elapsed_s.clone()

    first_low_contact = (
        ~reset
        & (current_stage == SPATIAL_LOW)
        & (previous_stage != SPATIAL_LOW)
        & (previous_entry_step < 0)
    )
    finite_speed = torch.nan_to_num(
        torch.abs(forward_speed), nan=0.0, posinf=0.0, neginf=0.0
    )
    entry_step[first_low_contact] = episode_step[first_low_contact]
    entry_speed[first_low_contact] = finite_speed[first_low_contact]

    # Include the transition-to-HighEnd step so the frozen duration describes
    # the complete LOW segment rather than ending one policy tick early.
    in_or_leaving_low = ~reset & (
        (current_stage == SPATIAL_LOW)
        | (
            (previous_stage == SPATIAL_LOW)
            & (current_stage == SPATIAL_HIGH_END)
        )
    )
    has_entry = entry_step >= 0
    measured = torch.clamp(
        (episode_step - entry_step).to(elapsed_s.dtype) * float(control_dt),
        min=0.0,
    )
    elapsed_s = torch.where(in_or_leaving_low & has_entry, measured, elapsed_s)

    entry_step[reset] = -1
    entry_speed[reset] = 0.0
    elapsed_s[reset] = 0.0
    return entry_step, entry_speed, elapsed_s


def update_low_capture_stability(
    current_stage: torch.Tensor,
    forward_speed: torch.Tensor,
    previous_stable_count: torch.Tensor,
    previous_success: torch.Tensor,
    elapsed_s: torch.Tensor,
    reset: torch.Tensor,
    target_speed: float,
    speed_tolerance: float,
    stable_steps: int,
    deadline_s: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update consecutive target-speed samples and latch capture success.

    Success is retained through HighEnd for episode diagnostics.  ``new`` is a
    one-update pulse used for a completion bonus, while ``timely`` identifies a
    capture completed no later than the configured response deadline.
    """

    tensors = (
        forward_speed,
        previous_stable_count,
        previous_success,
        elapsed_s,
        reset,
    )
    if any(value.shape != current_stage.shape for value in tensors):
        raise ValueError("all low-capture stability tensors must have the same shape")
    if current_stage.dtype == torch.bool or current_stage.is_floating_point():
        raise TypeError("current_stage must use an integer dtype")
    if previous_stable_count.dtype == torch.bool or previous_stable_count.is_floating_point():
        raise TypeError("previous_stable_count must use an integer dtype")
    if previous_success.dtype != torch.bool or reset.dtype != torch.bool:
        raise TypeError("previous_success and reset must use bool dtype")
    if not forward_speed.is_floating_point() or not elapsed_s.is_floating_point():
        raise TypeError("forward_speed and elapsed_s must use floating dtypes")
    if target_speed < 0.0 or speed_tolerance < 0.0:
        raise ValueError("target_speed and speed_tolerance must be non-negative")
    if stable_steps < 1 or deadline_s <= 0.0:
        raise ValueError("stable_steps and deadline_s must be positive")

    low = ~reset & (current_stage == SPATIAL_LOW)
    finite = torch.isfinite(forward_speed)
    within_target = (
        finite
        & (torch.abs(forward_speed) <= float(target_speed + speed_tolerance))
    )
    stable_count = torch.where(
        low & within_target,
        previous_stable_count + 1,
        torch.zeros_like(previous_stable_count),
    )
    reached = low & (stable_count >= int(stable_steps))
    success = (~reset) & (previous_success | reached)
    new_success = success & ~previous_success
    timely = new_success & torch.isfinite(elapsed_s) & (
        elapsed_s <= float(deadline_s)
    )
    return stable_count, success, new_success, timely


def capture_speed_envelope(
    elapsed_s: torch.Tensor,
    entry_speed: torch.Tensor,
    target_speed: float,
    deadline_s: float,
    decay_power: float = 1.0,
) -> torch.Tensor:
    """Return a monotonic speed ceiling reaching ``target_speed`` at deadline."""

    if elapsed_s.shape != entry_speed.shape:
        raise ValueError("elapsed_s and entry_speed must have the same shape")
    if not elapsed_s.is_floating_point() or not entry_speed.is_floating_point():
        raise TypeError("elapsed_s and entry_speed must use floating dtypes")
    if target_speed < 0.0 or deadline_s <= 0.0 or decay_power <= 0.0:
        raise ValueError("invalid capture envelope parameters")
    elapsed = torch.nan_to_num(
        elapsed_s, nan=0.0, posinf=float(deadline_s), neginf=0.0
    )
    initial = torch.maximum(
        torch.nan_to_num(
            torch.abs(entry_speed),
            nan=float(target_speed),
            posinf=float(target_speed),
            neginf=float(target_speed),
        ),
        torch.full_like(entry_speed, float(target_speed)),
    )
    progress = torch.clamp(elapsed / float(deadline_s), 0.0, 1.0)
    remaining = torch.pow(1.0 - progress, float(decay_power))
    return float(target_speed) + (initial - float(target_speed)) * remaining


def advance_spatial_course_stage(
    previous_stage: torch.Tensor,
    low_contact: torch.Tensor,
    high_end_contact: torch.Tensor,
    reset: torch.Tensor,
) -> torch.Tensor:
    """Advance a batched causal H--L--H contact state.

    ``LOW`` is latched through flight and split-boundary steps.  It is released
    only by a measured contact with the final high-friction collider.  If low
    and final-high contact are both present in one frame, low contact wins so a
    boundary-straddling sole cannot clear the conservative state early.
    """

    if not (
        previous_stage.shape
        == low_contact.shape
        == high_end_contact.shape
        == reset.shape
    ):
        raise ValueError("all spatial-course state tensors must have the same shape")
    if previous_stage.dtype == torch.bool or previous_stage.is_floating_point():
        raise TypeError("previous_stage must use an integer dtype")
    for name, value in (
        ("low_contact", low_contact),
        ("high_end_contact", high_end_contact),
        ("reset", reset),
    ):
        if value.dtype != torch.bool:
            raise TypeError(f"{name} must use bool dtype")

    stage = previous_stage.clone()
    active = ~reset
    # Final-high may be accepted only after a prior low contact.  This rejects
    # resets/teleports directly onto the last patch as successful traversals.
    enter_high_end = (
        active
        & ~low_contact
        & high_end_contact
        & (previous_stage == SPATIAL_LOW)
    )
    stage[enter_high_end] = SPATIAL_HIGH_END
    # Apply LOW second: it has explicit priority for simultaneous split contact
    # and also supports a conservative transition back when walking backwards.
    stage[active & low_contact] = SPATIAL_LOW
    stage[reset] = SPATIAL_HIGH_START
    return stage


def spatial_course_success_mask(
    stage: torch.Tensor,
    high_end_contact: torch.Tensor,
    root_local_x: torch.Tensor,
    minimum_local_x: float,
) -> torch.Tensor:
    """Return success only after LOW, on real final-high contact, near its end."""

    if not (stage.shape == high_end_contact.shape == root_local_x.shape):
        raise ValueError("all spatial-course success tensors must have the same shape")
    if high_end_contact.dtype != torch.bool:
        raise TypeError("high_end_contact must use bool dtype")
    if not root_local_x.is_floating_point():
        raise TypeError("root_local_x must use a floating dtype")
    return (
        (stage == SPATIAL_HIGH_END)
        & high_end_contact
        & torch.isfinite(root_local_x)
        & (root_local_x >= float(minimum_local_x))
    )


def update_transition_retention_latch(
    previous_stage: torch.Tensor,
    current_stage: torch.Tensor,
    heading_error: torch.Tensor,
    previous_entry_heading: torch.Tensor,
    previous_high_end_elapsed_s: torch.Tensor,
    reset: torch.Tensor,
    control_dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Latch heading error at LOW entry and time spent in HIGH_END.

    ``entry_heading`` freezes at the first LOW contact and is retained through
    HIGH_END so a reward can demand ``heading == heading_at_low_entry`` after
    the maneuver.  ``high_end_elapsed_s`` accumulates only while the course
    stage is HIGH_END, giving the reward a causal post-transition window.
    """

    tensors = (
        previous_stage,
        current_stage,
        heading_error,
        previous_entry_heading,
        previous_high_end_elapsed_s,
        reset,
    )
    if any(value.shape != previous_stage.shape for value in tensors):
        raise ValueError("all transition-retention tensors must have the same shape")
    if previous_stage.dtype == torch.bool or previous_stage.is_floating_point():
        raise TypeError("previous_stage must use an integer dtype")
    if current_stage.dtype == torch.bool or current_stage.is_floating_point():
        raise TypeError("current_stage must use an integer dtype")
    if reset.dtype != torch.bool:
        raise TypeError("reset must use bool dtype")
    if not heading_error.is_floating_point():
        raise TypeError("heading_error must use a floating dtype")
    if not previous_entry_heading.is_floating_point():
        raise TypeError("previous_entry_heading must use a floating dtype")
    if not previous_high_end_elapsed_s.is_floating_point():
        raise TypeError("previous_high_end_elapsed_s must use a floating dtype")
    if control_dt <= 0.0:
        raise ValueError("control_dt must be positive")

    entry_heading = previous_entry_heading.clone()
    high_end_elapsed_s = previous_high_end_elapsed_s.clone()

    first_low_contact = (
        ~reset
        & (current_stage == SPATIAL_LOW)
        & (previous_stage != SPATIAL_LOW)
        & torch.isnan(previous_entry_heading)
    )
    finite_heading = torch.nan_to_num(
        heading_error, nan=0.0, posinf=0.0, neginf=0.0
    )
    entry_heading[first_low_contact] = finite_heading[first_low_contact]

    in_high_end = ~reset & (current_stage == SPATIAL_HIGH_END)
    high_end_elapsed_s = torch.where(
        in_high_end,
        previous_high_end_elapsed_s + float(control_dt),
        torch.zeros_like(previous_high_end_elapsed_s),
    )

    entry_heading[reset] = float("nan")
    high_end_elapsed_s[reset] = 0.0
    return entry_heading, high_end_elapsed_s


def transition_stage_heading_weight(
    stage: torch.Tensor,
    high_end_elapsed_s: torch.Tensor,
    low_weight: float,
    high_start_weight: float,
    high_end_peak_weight: float,
    high_end_decay_s: float,
) -> torch.Tensor:
    """Smooth stage/time weighting for the transition heading penalty.

    LOW and the first seconds of HIGH_END receive the largest weight; nominal
    HIGH_START keeps its smaller value.  HIGH_END weight decays exponentially
    toward ``high_start_weight`` so long steady-state walking is not retrained.
    """

    if stage.shape != high_end_elapsed_s.shape:
        raise ValueError("stage and high_end_elapsed_s must have the same shape")
    if stage.dtype == torch.bool or stage.is_floating_point():
        raise TypeError("stage must use an integer dtype")
    if not high_end_elapsed_s.is_floating_point():
        raise TypeError("high_end_elapsed_s must use a floating dtype")
    for name, value in (
        ("low_weight", low_weight),
        ("high_start_weight", high_start_weight),
        ("high_end_peak_weight", high_end_peak_weight),
        ("high_end_decay_s", high_end_decay_s),
    ):
        if not isinstance(value, (int, float)) or not value >= 0.0:
            raise ValueError(f"{name} must be a non-negative number")

    elapsed = torch.nan_to_num(
        high_end_elapsed_s, nan=0.0, posinf=0.0, neginf=0.0
    )
    decay = torch.exp(
        -elapsed / max(float(high_end_decay_s), 1.0e-6)
    )
    high_end_weight = float(high_start_weight) + (
        float(high_end_peak_weight) - float(high_start_weight)
    ) * decay
    low = stage == SPATIAL_LOW
    high_end = stage == SPATIAL_HIGH_END
    return torch.where(
        low,
        torch.full_like(elapsed, float(low_weight)),
        torch.where(
            high_end,
            high_end_weight,
            torch.full_like(elapsed, float(high_start_weight)),
        ),
    )
