"""FootSensor15 notification protocol and per-device int16 unwrapping."""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass

import numpy as np


DEVICE_NAME = "FootSensor15"
CHAR_UUID = "0000ab01-0000-1000-8000-00805f9b34fb"
FRAME_LEN = 125
FRAME_HEADER = 0x7D
DATA_OFFSET = 4
DATA_BYTES = 120
NUM_SENSORS = 15
SENSOR_STRUCT = struct.Struct(">hhhh")
INT16_RANGE = 65536
SAT_RAIL = 32767
OVER_RANGE = 32768


class FrameError(ValueError):
    """A BLE notification is not a valid FootSensor15 frame."""


class Int16Unwrapper3D:
    """Extend three signed int16 streams per sensor without sharing state.

    Standard int16 modular wrap (32767 -> -32768) is unwrapped by delta logic.
    When the unwrapped value diverges more than one full int16 range from the
    first-frame origin, or the raw value sits at the rail (+/-32767), the channel
    is flagged saturated and the extended value is wrapped back into the band
    [origin - 32768, origin + 32767] so a faulty/over-range sensor cannot blow
    the counts up to millions.  Saturated frames are unreliable measurements.
    """

    def __init__(self, num_sensors: int = NUM_SENSORS) -> None:
        self.num_sensors = num_sensors
        self.reset()

    def reset(self) -> None:
        self._previous: np.ndarray | None = None
        self._extended: np.ndarray | None = None
        self._origin: np.ndarray | None = None
        self.last_saturation: np.ndarray | None = None

    def push(self, wire_xyz: np.ndarray) -> np.ndarray:
        wire = np.asarray(wire_xyz, dtype=np.int64)
        if wire.shape != (self.num_sensors, 3):
            raise ValueError(f"expected {(self.num_sensors, 3)}, got {wire.shape}")
        at_rail = np.abs(wire) >= SAT_RAIL
        if self._previous is None:
            self._previous = wire.copy()
            self._extended = wire.copy()
            self._origin = wire.copy()
            self.last_saturation = at_rail
            return self._extended.copy()
        delta = wire - self._previous
        delta = np.where(delta > 32767, delta - INT16_RANGE, delta)
        delta = np.where(delta < -32768, delta + INT16_RANGE, delta)
        self._extended = self._extended + delta
        diverged = self._extended - self._origin
        over = np.abs(diverged) > OVER_RANGE
        if bool(np.any(over)):
            wrapped = self._origin + (
                np.mod(diverged + 32768, INT16_RANGE) - 32768
            )
            self._extended = np.where(over, wrapped, self._extended)
        self.last_saturation = at_rail | over
        self._previous = wire.copy()
        return self._extended.copy()


@dataclass(frozen=True)
class SensorFrame:
    source_sequence: int
    received_wall_ns: int
    received_monotonic_ns: int
    received_monotonic: float
    temperature_x10: np.ndarray
    magnetic_xyz: np.ndarray
    saturation_xyz: np.ndarray | None = None


class FrameParser:
    """Stateful parser. Create exactly one instance for each physical foot."""

    def __init__(self) -> None:
        self._unwrap = Int16Unwrapper3D()

    def reset(self) -> None:
        self._unwrap.reset()

    def parse(self, data: bytes) -> SensorFrame:
        if len(data) < FRAME_LEN:
            raise FrameError(f"short frame: {len(data)} < {FRAME_LEN}")
        if data[0] != FRAME_HEADER or data[2] != 0xF0 or data[3] != 0x02:
            raise FrameError("invalid frame header/type")
        raw = data[DATA_OFFSET : DATA_OFFSET + DATA_BYTES]
        temperature = np.empty(NUM_SENSORS, dtype=np.int32)
        xyz = np.empty((NUM_SENSORS, 3), dtype=np.int32)
        for index in range(NUM_SENSORS):
            start = index * SENSOR_STRUCT.size
            try:
                t_x10, x, y, z = SENSOR_STRUCT.unpack_from(raw, start)
            except struct.error as error:
                raise FrameError(f"invalid sensor record {index}") from error
            temperature[index] = t_x10
            xyz[index] = (x, y, z)
        received_monotonic_ns = time.monotonic_ns()
        received_wall_ns = time.time_ns()
        magnetic_xyz = self._unwrap.push(xyz)
        return SensorFrame(
            source_sequence=int(data[1]),
            received_wall_ns=received_wall_ns,
            received_monotonic_ns=received_monotonic_ns,
            received_monotonic=received_monotonic_ns * 1.0e-9,
            temperature_x10=temperature,
            magnetic_xyz=magnetic_xyz,
            saturation_xyz=None
            if self._unwrap.last_saturation is None
            else self._unwrap.last_saturation.copy(),
        )


def transform_magnetic(
    magnetic_xyz: np.ndarray,
    sensor_permutation: list[int],
    axis_sign: list[float],
) -> np.ndarray:
    """Apply a configured channel order and XYZ sign convention."""

    if sorted(sensor_permutation) != list(range(NUM_SENSORS)):
        raise ValueError("sensor_permutation must contain each index 0..14 exactly once")
    sign = np.asarray(axis_sign, dtype=np.float64)
    if sign.shape != (3,) or not np.all(np.isin(sign, (-1.0, 1.0))):
        raise ValueError("axis_sign must contain three values, each -1 or 1")
    value = np.asarray(magnetic_xyz, dtype=np.float64)
    if value.shape != (NUM_SENSORS, 3):
        raise ValueError(f"magnetic_xyz must have shape {(NUM_SENSORS, 3)}")
    return value[np.asarray(sensor_permutation, dtype=np.int64)] * sign


def make_test_frame(
    xyz: np.ndarray,
    temperature_x10: np.ndarray | None = None,
    sequence: int = 0,
) -> bytes:
    """Build a protocol frame for offline tests; it is never used for hardware."""

    xyz_i16 = np.asarray(xyz, dtype=np.int16)
    if xyz_i16.shape != (NUM_SENSORS, 3):
        raise ValueError(f"xyz must have shape {(NUM_SENSORS, 3)}")
    temperature = (
        np.full(NUM_SENSORS, 250, dtype=np.int16)
        if temperature_x10 is None
        else np.asarray(temperature_x10, dtype=np.int16)
    )
    if temperature.shape != (NUM_SENSORS,):
        raise ValueError(f"temperature must have shape {(NUM_SENSORS,)}")
    payload = bytearray(FRAME_LEN)
    payload[0] = FRAME_HEADER
    payload[1] = sequence & 0xFF
    payload[2] = 0xF0
    payload[3] = 0x02
    for index in range(NUM_SENSORS):
        SENSOR_STRUCT.pack_into(
            payload,
            DATA_OFFSET + index * SENSOR_STRUCT.size,
            int(temperature[index]),
            int(xyz_i16[index, 0]),
            int(xyz_i16[index, 1]),
            int(xyz_i16[index, 2]),
        )
    return bytes(payload)
