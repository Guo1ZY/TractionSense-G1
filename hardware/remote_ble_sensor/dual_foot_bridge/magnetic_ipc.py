"""F0M1 IPC: normalized dual-foot 15xXYZ magnetic arrays for g1_ctrl."""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ipc import atomic_write
from .protocol import NUM_SENSORS


MAGIC_F0M1 = 0x46304D31
HEADER = struct.Struct("<IIQffffff")
MAGNETIC_FLOATS = 2 * NUM_SENSORS * 3
PACKET_SIZE = HEADER.size + MAGNETIC_FLOATS * 4


@dataclass(frozen=True)
class MagneticIpcSample:
    sequence: int
    stamp_ns: int
    valid: tuple[float, float]
    age_s: tuple[float, float]
    period_s: tuple[float, float]
    magnetic: np.ndarray


class F0M1Writer:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.sequence = 0

    def write(
        self,
        magnetic_left: np.ndarray,
        magnetic_right: np.ndarray,
        *,
        valid_left: float,
        valid_right: float,
        age_left_s: float,
        age_right_s: float,
        period_left_s: float,
        period_right_s: float,
    ) -> MagneticIpcSample:
        magnetic = np.stack(
            (
                np.asarray(magnetic_left, dtype=np.float32),
                np.asarray(magnetic_right, dtype=np.float32),
            )
        )
        if magnetic.shape != (2, NUM_SENSORS, 3):
            raise ValueError(f"magnetic must have shape {(2, NUM_SENSORS, 3)}")
        metadata = [
            valid_left,
            valid_right,
            age_left_s,
            age_right_s,
            period_left_s,
            period_right_s,
        ]
        if not np.isfinite(magnetic).all() or not all(math.isfinite(float(x)) for x in metadata):
            raise ValueError("F0M1 values must be finite")
        sample = MagneticIpcSample(
            sequence=self.sequence,
            stamp_ns=time.time_ns(),
            valid=(float(np.clip(valid_left, 0.0, 1.0)), float(np.clip(valid_right, 0.0, 1.0))),
            age_s=(max(0.0, age_left_s), max(0.0, age_right_s)),
            period_s=(
                float(np.clip(period_left_s, 0.001, 0.25)),
                float(np.clip(period_right_s, 0.001, 0.25)),
            ),
            magnetic=np.clip(magnetic, -6.0, 6.0),
        )
        payload = HEADER.pack(
            MAGIC_F0M1,
            sample.sequence,
            sample.stamp_ns,
            sample.valid[0],
            sample.valid[1],
            sample.age_s[0],
            sample.age_s[1],
            sample.period_s[0],
            sample.period_s[1],
        ) + sample.magnetic.astype("<f4", copy=False).tobytes(order="C")
        if len(payload) != PACKET_SIZE:
            raise AssertionError(f"internal F0M1 size mismatch: {len(payload)}")
        atomic_write(self.path, payload)
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        return sample


def read_packet(path: Path) -> MagneticIpcSample:
    payload = path.read_bytes()
    if len(payload) != PACKET_SIZE:
        raise ValueError(f"expected {PACKET_SIZE} bytes, got {len(payload)}")
    header = HEADER.unpack_from(payload)
    if header[0] != MAGIC_F0M1:
        raise ValueError("bad F0M1 magic")
    magnetic = np.frombuffer(payload, dtype="<f4", offset=HEADER.size).copy()
    return MagneticIpcSample(
        sequence=header[1],
        stamp_ns=header[2],
        valid=(header[3], header[4]),
        age_s=(header[5], header[6]),
        period_s=(header[7], header[8]),
        magnetic=magnetic.reshape(2, NUM_SENSORS, 3),
    )

