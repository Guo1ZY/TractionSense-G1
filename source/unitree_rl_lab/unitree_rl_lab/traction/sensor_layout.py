"""Canonical 15-point flexible magnetic foot layout and deployment adapters."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from unitree_rl_lab.sensors.hall_sensor_config import DEFAULT_HALL_POSITIONS_NORMALIZED

from .ble import FootSensorFrame


FootId = Literal["left", "right"]


def _rotation_z(angle_rad: float) -> tuple[tuple[float, float, float], ...]:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


@dataclass(frozen=True)
class FootSensorLayoutCfg:
    """Explicit channel, location, region, axis, and mirror semantics.

    ``sensor_positions_xy`` is for physical sensor sites only.  Any IDW grid
    used by a visualization remains outside this configuration.
    """

    sensor_names: tuple[str, ...]
    sensor_positions_xy: tuple[tuple[float, float], ...]
    region_ids: tuple[int, ...]
    ble_channel_to_sensor_index: tuple[int, ...]
    per_channel_axis_transform: tuple[tuple[tuple[float, float, float], ...], ...]
    left_right_mirror_transform: tuple[tuple[float, float, float], ...]
    coordinate_frame: str
    position_unit: str
    is_provisional: bool
    provenance: str
    toe_direction: str = "+x"
    foot_width_direction: str = "+y points to robot left"

    def __post_init__(self) -> None:
        if len(self.sensor_names) != 15 or self.sensor_names != tuple(
            f"P{i:02d}" for i in range(15)
        ):
            raise ValueError("sensor_names must be exactly P00..P14")
        positions = np.asarray(self.sensor_positions_xy, dtype=np.float64)
        regions = np.asarray(self.region_ids)
        mapping = np.asarray(self.ble_channel_to_sensor_index)
        axes = np.asarray(self.per_channel_axis_transform, dtype=np.float64)
        mirror = np.asarray(self.left_right_mirror_transform, dtype=np.float64)
        if positions.shape != (15, 2):
            raise ValueError(f"sensor_positions_xy must be (15,2), got {positions.shape}")
        if regions.shape != (15,) or set(regions.tolist()) != {0, 1, 2}:
            raise ValueError("region_ids must be 15 entries spanning forefoot/midfoot/heel")
        if mapping.shape != (15,) or sorted(mapping.tolist()) != list(range(15)):
            raise ValueError("BLE channel mapping must be a permutation of P00..P14")
        if axes.shape != (15, 3, 3):
            raise ValueError(f"axis transforms must be (15,3,3), got {axes.shape}")
        if mirror.shape != (3, 3):
            raise ValueError(f"mirror transform must be (3,3), got {mirror.shape}")
        for index, transform in enumerate((*axes, mirror)):
            if not np.allclose(transform @ transform.T, np.eye(3), atol=1.0e-6):
                raise ValueError(f"axis transform {index} is not orthonormal")

    @property
    def positions_array(self) -> np.ndarray:
        return np.asarray(self.sensor_positions_xy, dtype=np.float32)

    @property
    def axis_transform_array(self) -> np.ndarray:
        return np.asarray(self.per_channel_axis_transform, dtype=np.float32)

    @property
    def mirror_transform_array(self) -> np.ndarray:
        return np.asarray(self.left_right_mirror_transform, dtype=np.float32)


# Rotations documented in ble_viz_dashboard_demo.py.  They align chip-local
# Hall XY for visualization.  They are explicit here, but remain provisional
# until a signed magnetic-axis experiment validates the signs and right-foot
# mirror rule.  This is not a force calibration.
_CHIP_XY_ROTATIONS = (
    -math.pi / 2,
    -math.pi / 2,
    0.0,
    math.pi / 2,
    0.0,
    -math.pi / 2,
    -math.pi / 2,
    math.pi,
    math.pi / 2,
    -math.pi / 2,
    0.0,
    -math.pi / 2,
    math.pi,
    math.pi / 2,
    0.0,
)

PROVISIONAL_NORMALIZED_LAYOUT = FootSensorLayoutCfg(
    sensor_names=tuple(f"P{i:02d}" for i in range(15)),
    sensor_positions_xy=DEFAULT_HALL_POSITIONS_NORMALIZED,
    region_ids=(0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2),
    ble_channel_to_sensor_index=tuple(range(15)),
    per_channel_axis_transform=tuple(_rotation_z(angle) for angle in _CHIP_XY_ROTATIONS),
    left_right_mirror_transform=((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    coordinate_frame=(
        "foot_local: +x toe, +y robot-left, +z up; provisional Hall-axis alignment"
    ),
    position_unit="normalized_sole_length_width_from_a4_scan",
    is_provisional=True,
    provenance=(
        "15 square centres digitized from the supplied 1:1 A4 vola_sensor/2.png; "
        "normalized by the drawn sole ink bounds u=7..508, v=488..1780 and converted "
        "to canonical +x toe, +y image-left foot coordinates"
    ),
)


@dataclass(frozen=True)
class DualFootForceInput:
    """Independent force-source input retained for legacy force policies.

    This type may be populated by Isaac ContactSensor, a load cell or another
    real force sensor.  It must never be populated by ``FootSensorFrame``.
    Hall-only policies use :class:`DualFootMagneticInput` instead.
    """

    timestamp: float
    left_force_xyz: np.ndarray
    right_force_xyz: np.ndarray
    left_valid: bool
    right_valid: bool
    left_age: float
    right_age: float
    left_source: str = "missing"
    right_source: str = "missing"

    def __post_init__(self) -> None:
        for name in ("left_force_xyz", "right_force_xyz"):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != (3,):
                raise ValueError(f"{name} must be shape (3,), got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
            object.__setattr__(self, name, value)
        if self.left_age < 0.0 or self.right_age < 0.0:
            raise ValueError("sample ages must be non-negative")

    @property
    def force_vector(self) -> np.ndarray:
        return np.concatenate((self.left_force_xyz, self.right_force_xyz))

    @property
    def valid_vector(self) -> np.ndarray:
        return np.asarray((self.left_valid, self.right_valid), dtype=np.float32)

    @property
    def age_vector(self) -> np.ndarray:
        return np.asarray((self.left_age, self.right_age), dtype=np.float32)


@dataclass(frozen=True)
class DualFootMagneticInput:
    """Canonical dual-foot view of the data the real hardware actually emits.

    ``hall_xyz`` has shape ``[2,15,3]`` and remains in raw unwrapped counts
    after configured channel/axis transforms.  It is neither tesla nor force.
    Baseline/temperature normalization is a separate deployment step.
    """

    timestamp: float
    hall_xyz: np.ndarray
    temperature_c: np.ndarray
    valid: np.ndarray
    age_s: np.ndarray
    sample_period_s: np.ndarray
    sequence: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "hall_xyz": (self.hall_xyz, (2, 15, 3), np.float32),
            "temperature_c": (self.temperature_c, (2, 15), np.float32),
            "valid": (self.valid, (2,), np.float32),
            "age_s": (self.age_s, (2,), np.float32),
            "sample_period_s": (self.sample_period_s, (2,), np.float32),
            "sequence": (self.sequence, (2,), np.int64),
        }
        for name, (source, shape, dtype) in arrays.items():
            value = np.asarray(source, dtype=dtype)
            if value.shape != shape:
                raise ValueError(f"{name} must be shape {shape}, got {value.shape}")
            if name != "sequence" and not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
            object.__setattr__(self, name, value)
        if np.any(self.age_s < 0.0) or np.any(self.sample_period_s <= 0.0):
            raise ValueError("age_s must be non-negative and sample periods positive")

    @property
    def left_foot(self) -> np.ndarray:
        return self.hall_xyz[0]

    @property
    def right_foot(self) -> np.ndarray:
        return self.hall_xyz[1]


class SingleFootSensorAdapter:
    """Map one real BLE frame into the canonical Hall channel/axis order."""

    def __init__(
        self,
        foot_id: FootId,
        *,
        layout: FootSensorLayoutCfg = PROVISIONAL_NORMALIZED_LAYOUT,
    ) -> None:
        if foot_id not in ("left", "right"):
            raise ValueError(f"invalid foot_id {foot_id!r}")
        self.foot_id = foot_id
        self.layout = layout
        self._last_timestamp = -math.inf

    def adapt(self, wire_frame: FootSensorFrame, *, now: float | None = None) -> FootSensorFrame:
        if wire_frame.foot_id not in (self.foot_id, "unassigned"):
            raise ValueError(
                f"frame belongs to {wire_frame.foot_id}, adapter is {self.foot_id}"
            )
        mapping = np.asarray(self.layout.ble_channel_to_sensor_index, dtype=np.int64)
        hall = np.empty((15, 3), dtype=np.float32)
        temperature = np.empty(15, dtype=np.float32)
        hall[mapping] = wire_frame.hall_xyz
        temperature[mapping] = wire_frame.temperature
        hall = np.einsum(
            "nij,nj->ni",
            self.layout.axis_transform_array,
            hall,
            optimize=True,
        )
        if self.foot_id == "right":
            hall = hall @ self.layout.mirror_transform_array.T

        valid = bool(wire_frame.valid)
        if wire_frame.timestamp < self._last_timestamp:
            valid = False
        self._last_timestamp = max(self._last_timestamp, wire_frame.timestamp)
        if not np.isfinite(hall).all() or not np.isfinite(temperature).all():
            valid = False

        sample_age = (
            max(0.0, float(now) - wire_frame.timestamp)
            if now is not None
            else wire_frame.sample_age
        )
        return FootSensorFrame(
            timestamp=wire_frame.timestamp,
            foot_id=self.foot_id,
            hall_xyz=hall,
            temperature=temperature,
            valid=valid,
            sequence=wire_frame.sequence,
            header_byte_1=wire_frame.header_byte_1,
            sample_age=sample_age,
        )


class DualFootSensorAggregator:
    """Aggregate independent Hall feet without fabricating missing data."""

    def __init__(
        self,
        *,
        timeout_s: float = 0.10,
        missing_age_s: float = 1.0e6,
    ) -> None:
        if timeout_s <= 0.0 or missing_age_s < timeout_s:
            raise ValueError("invalid timeout/missing age")
        self.timeout_s = timeout_s
        self.missing_age_s = missing_age_s
        self._frames: dict[FootId, FootSensorFrame] = {}
        self._sample_period_s: dict[FootId, float] = {"left": 0.02, "right": 0.02}

    def update(self, frame: FootSensorFrame) -> None:
        if frame.foot_id not in ("left", "right"):
            raise ValueError(f"cannot aggregate foot_id={frame.foot_id!r}")
        previous = self._frames.get(frame.foot_id)
        if previous is not None and frame.timestamp < previous.timestamp:
            return
        if previous is not None:
            period = frame.timestamp - previous.timestamp
            if 0.001 <= period <= 0.25:
                self._sample_period_s[frame.foot_id] = float(period)
        self._frames[frame.foot_id] = frame

    def magnetic_input(self, now: float) -> DualFootMagneticInput:
        """Return raw magnetic counts, temperature and independent foot health."""
        hall = np.zeros((2, 15, 3), dtype=np.float32)
        temperature = np.zeros((2, 15), dtype=np.float32)
        valid = np.zeros(2, dtype=np.float32)
        age = np.full(2, self.missing_age_s, dtype=np.float32)
        period = np.asarray(
            (self._sample_period_s["left"], self._sample_period_s["right"]),
            dtype=np.float32,
        )
        sequence = np.zeros(2, dtype=np.int64)
        for foot_index, foot_id in enumerate(("left", "right")):
            frame = self._frames.get(foot_id)
            if frame is None:
                continue
            sample_age = max(0.0, float(now) - frame.timestamp)
            hall[foot_index] = frame.hall_xyz
            temperature[foot_index] = frame.temperature_c
            valid[foot_index] = float(frame.valid and sample_age <= self.timeout_s)
            age[foot_index] = sample_age
            sequence[foot_index] = frame.sequence
        return DualFootMagneticInput(
            timestamp=float(now),
            hall_xyz=hall,
            temperature_c=temperature,
            valid=valid,
            age_s=age,
            sample_period_s=period,
            sequence=sequence,
        )
