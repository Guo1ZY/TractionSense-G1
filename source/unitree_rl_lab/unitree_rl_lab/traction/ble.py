"""BLE wire parser for the current single-foot 15-channel sensor."""

from __future__ import annotations

from dataclasses import dataclass
import struct
import time
from typing import Literal

import numpy as np


DEVICE_NAME = "FootSensor15"
NOTIFY_CHARACTERISTIC_UUID = "0000ab01-0000-1000-8000-00805f9b34fb"
FRAME_LENGTH = 125
FRAME_HEADER = (0x7D, None, 0xF0, 0x02)
PAYLOAD_OFFSET = 4
CHANNEL_COUNT = 15
CHANNEL_BYTES = 8
PAYLOAD_BYTES = CHANNEL_COUNT * CHANNEL_BYTES
CHANNEL_FORMAT = ">hhhh"


@dataclass(frozen=True)
class FootSensorFrame:
    """One real ``FootSensor15`` sample in the device's observable domain.

    ``hall_xyz`` contains unwrapped ADC/Hall counts, not tesla and not force.
    The known wire packet has no calibrated force field, so force estimates
    deliberately do not belong to this data structure.
    """

    timestamp: float
    foot_id: Literal["left", "right", "unassigned"]
    hall_xyz: np.ndarray
    temperature: np.ndarray
    valid: bool = True
    sequence: int = 0
    header_byte_1: int | None = None
    sample_age: float = 0.0

    def __post_init__(self) -> None:
        hall = np.asarray(self.hall_xyz)
        temperature = np.asarray(self.temperature)
        if hall.shape != (15, 3):
            raise ValueError(f"hall_xyz must be (15,3), got {hall.shape}")
        if temperature.shape != (15,):
            raise ValueError(f"temperature must be (15,), got {temperature.shape}")
        if self.sample_age < 0.0 or self.sequence < 0:
            raise ValueError("sample_age and sequence must be non-negative")
        if self.header_byte_1 is not None and not 0 <= self.header_byte_1 <= 255:
            raise ValueError("header_byte_1 must be an unsigned byte")
        object.__setattr__(self, "hall_xyz", hall)
        object.__setattr__(self, "temperature", temperature)

    @property
    def temperature_c(self) -> np.ndarray:
        """Temperature in degrees Celsius, matching ``record_raw_hall.py``."""
        return self.temperature


class Int16Unwrapper:
    """Conservatively unwrap endpoint crossings into bounded int32 counts.

    A generic jump larger than half the int16 range is not sufficient evidence
    of wrapping: corrupted data or a real transient must not silently add a
    65536-count turn.  Only transitions between the two endpoint regions are
    unwrapped, matching the real-sensor visualization and reference tooling.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        limit: int = 2_000_000,
        wrap_threshold: int = 60_000,
    ) -> None:
        self.shape = shape
        self.limit = int(limit)
        self.wrap_threshold = int(wrap_threshold)
        self.last: np.ndarray | None = None
        self.extended: np.ndarray | None = None
        self.wrap_events = 0

    def reset(self) -> None:
        self.last = None
        self.extended = None
        self.wrap_events = 0

    def push(self, wire: np.ndarray) -> np.ndarray:
        current = np.asarray(wire, dtype=np.int32)
        if current.shape != self.shape:
            raise ValueError(f"wire shape {current.shape}, expected {self.shape}")
        if self.last is None:
            self.last = current.copy()
            self.extended = current.copy()
        else:
            delta = current.astype(np.int64) - self.last.astype(np.int64)
            wraps_negative = delta < -self.wrap_threshold
            wraps_positive = delta > self.wrap_threshold
            step = np.where(wraps_negative, delta + 65536, delta)
            step = np.where(wraps_positive, delta - 65536, step)
            self.wrap_events += int(
                np.count_nonzero(wraps_negative | wraps_positive)
            )
            extended = self.extended.astype(np.int64) + step
            self.extended = np.clip(
                extended, -self.limit, self.limit
            ).astype(np.int32)
            self.last = current.copy()
        return self.extended.copy()


class BleFrameParser:
    """Parse whole notifications or an arbitrary chunked byte stream.

    The final byte of the 125-byte frame is preserved as ``reserved`` only in
    parser diagnostics; its device-side semantics are currently undocumented.
    Timestamp and monotonic ``sequence`` are host-generated because the frame
    has no known device timestamp.  Header byte 1 is retained separately, but
    its firmware semantics are unknown (current real devices emit constant 1),
    so it must not be used as a frame counter or synchronization timestamp.
    """

    def __init__(self, *, foot_id: str = "unassigned") -> None:
        if foot_id not in ("left", "right", "unassigned"):
            raise ValueError(foot_id)
        self.foot_id = foot_id
        self.unwrap = Int16Unwrapper((CHANNEL_COUNT, 3))
        self.sequence = 0
        self.buffer = bytearray()
        self.last_reserved_byte: int | None = None
        self.rejected_bytes = 0

    @staticmethod
    def has_header(data: bytes | bytearray, offset: int = 0) -> bool:
        return (
            len(data) >= offset + 4
            and data[offset] == 0x7D
            and data[offset + 2] == 0xF0
            and data[offset + 3] == 0x02
        )

    def reset(self) -> None:
        self.unwrap.reset()
        self.sequence = 0
        self.buffer.clear()
        self.last_reserved_byte = None
        self.rejected_bytes = 0

    def parse_frame(
        self,
        data: bytes,
        *,
        timestamp: float | None = None,
    ) -> FootSensorFrame | None:
        if len(data) != FRAME_LENGTH or not self.has_header(data):
            return None
        raw = data[PAYLOAD_OFFSET : PAYLOAD_OFFSET + PAYLOAD_BYTES]
        hall_wire = np.empty((CHANNEL_COUNT, 3), dtype=np.int32)
        temperature_x10 = np.empty(CHANNEL_COUNT, dtype=np.int32)
        try:
            for channel in range(CHANNEL_COUNT):
                t_x10, x, y, z = struct.unpack_from(
                    CHANNEL_FORMAT, raw, channel * CHANNEL_BYTES
                )
                temperature_x10[channel] = t_x10
                hall_wire[channel] = (x, y, z)
        except struct.error:
            return None
        self.sequence += 1
        self.last_reserved_byte = data[-1]
        return FootSensorFrame(
            timestamp=time.monotonic() if timestamp is None else float(timestamp),
            foot_id=self.foot_id,
            hall_xyz=self.unwrap.push(hall_wire),
            temperature=temperature_x10.astype(np.float32) / 10.0,
            valid=True,
            sequence=self.sequence,
            header_byte_1=int(data[1]),
            sample_age=0.0,
        )

    def feed(
        self,
        chunk: bytes,
        *,
        timestamp: float | None = None,
    ) -> list[FootSensorFrame]:
        self.buffer.extend(chunk)
        frames: list[FootSensorFrame] = []
        while len(self.buffer) >= 4:
            if not self.has_header(self.buffer):
                del self.buffer[0]
                self.rejected_bytes += 1
                continue
            if len(self.buffer) < FRAME_LENGTH:
                break
            candidate = bytes(self.buffer[:FRAME_LENGTH])
            frame = self.parse_frame(candidate, timestamp=timestamp)
            if frame is None:
                del self.buffer[0]
                self.rejected_bytes += 1
                continue
            del self.buffer[:FRAME_LENGTH]
            frames.append(frame)
        return frames
