"""Replaceable magnetic-field models for the flexible Hall foot sensor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch


MU0_OVER_4PI = 1.0e-7  # T m / A in SI


class MagneticFieldModel(ABC):
    """Interface between mechanical state and Hall-local magnetic flux density."""

    @abstractmethod
    def compute(
        self,
        sensor_positions_w: torch.Tensor,
        magnet_positions_w: torch.Tensor,
        magnetic_moments_w: torch.Tensor,
        *,
        sensor_rotation_w: torch.Tensor | None = None,
        local_deformation: torch.Tensor | None = None,
        loading_history: torch.Tensor | None = None,
        dt: float | None = None,
    ) -> torch.Tensor:
        """Return ``Bx, By, Bz`` at every Hall site, in tesla.

        Args:
            sensor_positions_w: Shape ``[..., sensors, 3]``.
            magnet_positions_w: Shape ``[..., sensors, magnets, 3]``.
            magnetic_moments_w: Same shape as ``magnet_positions_w``; A m^2.
            sensor_rotation_w: Optional local-to-world matrices with shape
                ``[..., sensors, 3, 3]``.  When supplied, the result is rotated
                back to the Hall-local frame.
        """


class DipoleMagneticFieldModel(MagneticFieldModel):
    """Vectorized point-dipole approximation in SI units."""

    def __init__(self, min_distance: float = 5.0e-4) -> None:
        if min_distance <= 0.0:
            raise ValueError("min_distance must be positive")
        self.min_distance = float(min_distance)

    def compute(
        self,
        sensor_positions_w: torch.Tensor,
        magnet_positions_w: torch.Tensor,
        magnetic_moments_w: torch.Tensor,
        *,
        sensor_rotation_w: torch.Tensor | None = None,
        local_deformation: torch.Tensor | None = None,
        loading_history: torch.Tensor | None = None,
        dt: float | None = None,
    ) -> torch.Tensor:
        del local_deformation, loading_history, dt
        if sensor_positions_w.shape[-1] != 3:
            raise ValueError("sensor_positions_w must end in XYZ")
        expected_magnet_prefix = (*sensor_positions_w.shape[:-1], magnet_positions_w.shape[-2], 3)
        if tuple(magnet_positions_w.shape) != expected_magnet_prefix:
            raise ValueError(
                "magnet_positions_w must be [..., sensors, magnets, 3]; "
                f"got sensor={tuple(sensor_positions_w.shape)}, magnet={tuple(magnet_positions_w.shape)}"
            )
        if magnetic_moments_w.shape != magnet_positions_w.shape:
            raise ValueError("magnetic_moments_w must match magnet_positions_w")

        # r points from each magnet center to its Hall sampling point.
        r = sensor_positions_w.unsqueeze(-2) - magnet_positions_w
        distance_sq = torch.sum(r * r, dim=-1, keepdim=True)
        safe_distance_sq = torch.clamp_min(distance_sq, self.min_distance**2)
        inv_distance = torch.rsqrt(safe_distance_sq)
        r_hat = r * inv_distance
        moment_dot_r = torch.sum(magnetic_moments_w * r_hat, dim=-1, keepdim=True)
        field_each_w = (
            MU0_OVER_4PI
            * inv_distance.pow(3)
            * (3.0 * moment_dot_r * r_hat - magnetic_moments_w)
        )
        field_w = field_each_w.sum(dim=-2)
        if sensor_rotation_w is None:
            return field_w
        if sensor_rotation_w.shape != (*sensor_positions_w.shape[:-1], 3, 3):
            raise ValueError("sensor_rotation_w must be [..., sensors, 3, 3]")
        # R maps Hall-local vectors to world; R^T maps an axial vector back.
        return torch.einsum("...ji,...j->...i", sensor_rotation_w, field_w)


@dataclass(frozen=True)
class CalibratedMagneticInputs:
    """Inputs reserved for experimental/FEM/lookup-table field models."""

    local_deformation: torch.Tensor
    loading_history: torch.Tensor | None
    sensor_positions_w: torch.Tensor
    magnet_positions_w: torch.Tensor
    magnetic_moments_w: torch.Tensor
    sensor_rotation_w: torch.Tensor | None
    dt: float | None

    @property
    def compression_dz(self) -> torch.Tensor:
        return self.local_deformation[..., 2]

    @property
    def shear_dx_dy(self) -> torch.Tensor:
        return self.local_deformation[..., :2]

    @property
    def local_rotation(self) -> torch.Tensor:
        return self.local_deformation[..., 3:6]


CalibratedPredictor = Callable[[CalibratedMagneticInputs], torch.Tensor]


class CalibratedMagneticFieldModel(MagneticFieldModel):
    """Adapter for a learned calibration, FEM surrogate, or lookup table.

    The predictor receives current ``dx, dy, dz, roll, pitch, yaw`` plus the
    optional loading history.  Its output must already use Hall-local axes and
    have shape ``[..., sensors, 3]`` in tesla.  No Isaac scene types enter this
    interface, so a future calibration can be trained and tested independently.
    """

    def __init__(self, predictor: CalibratedPredictor | None = None) -> None:
        self.predictor = predictor

    def compute(
        self,
        sensor_positions_w: torch.Tensor,
        magnet_positions_w: torch.Tensor,
        magnetic_moments_w: torch.Tensor,
        *,
        sensor_rotation_w: torch.Tensor | None = None,
        local_deformation: torch.Tensor | None = None,
        loading_history: torch.Tensor | None = None,
        dt: float | None = None,
    ) -> torch.Tensor:
        if self.predictor is None:
            raise RuntimeError(
                "CalibratedMagneticFieldModel needs a predictor trained from experiment, FEM, or a lookup table"
            )
        if local_deformation is None:
            raise ValueError("calibrated field models require local_deformation")
        inputs = CalibratedMagneticInputs(
            local_deformation=local_deformation,
            loading_history=loading_history,
            sensor_positions_w=sensor_positions_w,
            magnet_positions_w=magnet_positions_w,
            magnetic_moments_w=magnetic_moments_w,
            sensor_rotation_w=sensor_rotation_w,
            dt=dt,
        )
        result = self.predictor(inputs)
        if result.shape != sensor_positions_w.shape:
            raise ValueError(
                f"calibrated predictor returned {tuple(result.shape)}, expected {tuple(sensor_positions_w.shape)}"
            )
        return torch.nan_to_num(result)
