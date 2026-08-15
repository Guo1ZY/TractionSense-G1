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


class FrameError(ValueError):
    """A BLE notification is not a valid FootSensor15 frame."""


class Int16Unwrapper3D:
    """Conservatively extend endpoint-crossing int16 streams per sensor."""

    def __init__(
        self,
        num_sensors: int = NUM_SENSORS,
        wrap_threshold: int = 60_000,
        limit: int = 2_000_000,
    ) -> None:
        self.num_sensors = num_sensors
        self.wrap_threshold = int(wrap_threshold)
        self.limit = int(limit)
        self.reset()

    def reset(self) -> None:
        self._previous: np.ndarray | None = None
        self._extended: np.ndarray | None = None
        self.wrap_events = 0

    def push(self, wire_xyz: np.ndarray) -> np.ndarray:
        wire = np.asarray(wire_xyz, dtype=np.int64)
        if wire.shape != (self.num_sensors, 3):
            raise ValueError(f"expected {(self.num_sensors, 3)}, got {wire.shape}")
        if self._previous is None:
            self._previous = wire.copy()
            self._extended = wire.copy()
            return self._extended.copy()
        delta = wire - self._previous
        wraps_negative = delta < -self.wrap_threshold
        wraps_positive = delta > self.wrap_threshold
        step = np.where(wraps_negative, delta + 65536, delta)
        step = np.where(wraps_positive, delta - 65536, step)
        self.wrap_events += int(np.count_nonzero(wraps_negative | wraps_positive))
        self._extended = np.clip(
            self._extended + step, -self.limit, self.limit
        ).astype(np.int64)
        self._previous = wire.copy()
        return self._extended.copy()


@dataclass(frozen=True)
class SensorFrame:
    # Legacy field name retained for the existing health schema.  This is raw
    # header byte 1; current real firmware emits constant 1, so it is not a
    # usable device sequence counter.
    source_sequence: int
    received_wall_ns: int
    received_monotonic: float
    temperature_x10: np.ndarray
    magnetic_xyz: np.ndarray


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
        return SensorFrame(
            source_sequence=int(data[1]),
            received_wall_ns=time.time_ns(),
            received_monotonic=time.monotonic(),
            temperature_x10=temperature,
            magnetic_xyz=self._unwrap.push(xyz),
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
