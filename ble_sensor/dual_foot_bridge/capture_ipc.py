"""F0R1 raw Hall capture IPC and synchronized dual-foot CSV logging.

The packet and CSV contain only measured Hall Bx/By/Bz counts, temperature,
timestamps and link-health metadata.  Values are never converted to force,
pressure, contact force or friction.  Array order is always left, right and
P00..P14; the two feet are never mixed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any

import numpy as np

from .ipc import atomic_write
from .protocol import NUM_SENSORS


MAGIC_F0R1 = 0x46305231
VERSION_F0R1 = 1
SIDES = ("left", "right")
AXES = ("bx", "by", "bz")
MISSING_AGE_S = 1.0e9

# magic, version, sequence, publish wall/monotonic timestamps,
# left/right frame wall timestamps, left/right frame monotonic timestamps,
# left/right source sequence, left/right valid, left/right age and period.
HEADER = struct.Struct("<IIQqqqqqqIIIIffff")
MAGNETIC_VALUES = 2 * NUM_SENSORS * 3
TEMPERATURE_VALUES = 2 * NUM_SENSORS
PACKET_SIZE = HEADER.size + MAGNETIC_VALUES * 8 + TEMPERATURE_VALUES * 4


def _mag_columns(side: str) -> list[str]:
    return [
        f"{side}_P{sensor:02d}_{axis}"
        for sensor in range(NUM_SENSORS)
        for axis in AXES
    ]


def _temp_columns(side: str) -> list[str]:
    return [f"{side}_P{sensor:02d}_temp_x10" for sensor in range(NUM_SENSORS)]


PAIR_COLUMNS = [
    "publish_sequence",
    "publish_wall_ns",
    "publish_monotonic_ns",
    "left_right_frame_skew_ns",
]
for _side in SIDES:
    PAIR_COLUMNS.extend(
        [
            f"{_side}_valid",
            f"{_side}_age_s",
            f"{_side}_sample_period_s",
            f"{_side}_frame_wall_ns",
            f"{_side}_frame_monotonic_ns",
            f"{_side}_source_sequence",
            *_temp_columns(_side),
            *_mag_columns(_side),
        ]
    )


@dataclass(frozen=True)
class RawCaptureSample:
    sequence: int
    publish_wall_ns: int
    publish_monotonic_ns: int
    frame_wall_ns: tuple[int, int]
    frame_monotonic_ns: tuple[int, int]
    source_sequence: tuple[int, int]
    valid: tuple[bool, bool]
    age_s: tuple[float, float]
    period_s: tuple[float, float]
    magnetic: np.ndarray
    temperature_x10: np.ndarray

    @property
    def frame_skew_ns(self) -> int:
        if not all(self.valid):
            return 0
        return self.frame_monotonic_ns[0] - self.frame_monotonic_ns[1]


class F0R1Writer:
    """Atomically publish the latest synchronized raw dual-foot snapshot."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.sequence = 0

    def write(
        self,
        left_magnetic: np.ndarray,
        right_magnetic: np.ndarray,
        left_temperature_x10: np.ndarray,
        right_temperature_x10: np.ndarray,
        *,
        publish_wall_ns: int,
        publish_monotonic_ns: int,
        frame_wall_ns: tuple[int, int],
        frame_monotonic_ns: tuple[int, int],
        source_sequence: tuple[int, int],
        valid: tuple[bool, bool],
        age_s: tuple[float, float],
        period_s: tuple[float, float],
    ) -> RawCaptureSample:
        magnetic = np.stack(
            [
                np.asarray(left_magnetic, dtype=np.int64),
                np.asarray(right_magnetic, dtype=np.int64),
            ]
        )
        temperature = np.stack(
            [
                np.asarray(left_temperature_x10, dtype=np.int32),
                np.asarray(right_temperature_x10, dtype=np.int32),
            ]
        )
        if magnetic.shape != (2, NUM_SENSORS, 3):
            raise ValueError("raw magnetic arrays must each have shape (15, 3)")
        if temperature.shape != (2, NUM_SENSORS):
            raise ValueError("temperature arrays must each have shape (15,)")
        ages = tuple(float(value) for value in age_s)
        periods = tuple(float(value) for value in period_s)
        if not np.all(np.isfinite((*ages, *periods))):
            raise ValueError("F0R1 age and period values must be finite")
        if any(value < 0.0 for value in (*ages, *periods)):
            raise ValueError("F0R1 age and period values must be non-negative")

        sample = RawCaptureSample(
            sequence=self.sequence,
            publish_wall_ns=int(publish_wall_ns),
            publish_monotonic_ns=int(publish_monotonic_ns),
            frame_wall_ns=(int(frame_wall_ns[0]), int(frame_wall_ns[1])),
            frame_monotonic_ns=(
                int(frame_monotonic_ns[0]),
                int(frame_monotonic_ns[1]),
            ),
            source_sequence=(
                int(source_sequence[0]) & 0xFFFFFFFF,
                int(source_sequence[1]) & 0xFFFFFFFF,
            ),
            valid=(bool(valid[0]), bool(valid[1])),
            age_s=ages,
            period_s=periods,
            magnetic=magnetic,
            temperature_x10=temperature,
        )
        header = HEADER.pack(
            MAGIC_F0R1,
            VERSION_F0R1,
            sample.sequence,
            sample.publish_wall_ns,
            sample.publish_monotonic_ns,
            sample.frame_wall_ns[0],
            sample.frame_wall_ns[1],
            sample.frame_monotonic_ns[0],
            sample.frame_monotonic_ns[1],
            sample.source_sequence[0],
            sample.source_sequence[1],
            int(sample.valid[0]),
            int(sample.valid[1]),
            sample.age_s[0],
            sample.age_s[1],
            sample.period_s[0],
            sample.period_s[1],
        )
        payload = (
            header
            + sample.magnetic.astype("<i8", copy=False).tobytes(order="C")
            + sample.temperature_x10.astype("<i4", copy=False).tobytes(order="C")
        )
        if len(payload) != PACKET_SIZE:
            raise AssertionError(f"internal F0R1 size mismatch: {len(payload)}")
        atomic_write(self.path, payload)
        self.sequence = (self.sequence + 1) & 0xFFFFFFFFFFFFFFFF
        return sample


def read_packet(path: Path) -> RawCaptureSample:
    payload = Path(path).read_bytes()
    if len(payload) != PACKET_SIZE:
        raise ValueError(f"F0R1 packet must be {PACKET_SIZE} bytes")
    header = HEADER.unpack_from(payload)
    if header[0] != MAGIC_F0R1 or header[1] != VERSION_F0R1:
        raise ValueError("bad F0R1 magic or version")
    offset = HEADER.size
    magnetic = np.frombuffer(
        payload, dtype="<i8", count=MAGNETIC_VALUES, offset=offset
    ).copy().reshape(2, NUM_SENSORS, 3)
    offset += MAGNETIC_VALUES * 8
    temperature = np.frombuffer(
        payload, dtype="<i4", count=TEMPERATURE_VALUES, offset=offset
    ).copy().reshape(2, NUM_SENSORS)
    return RawCaptureSample(
        sequence=header[2],
        publish_wall_ns=header[3],
        publish_monotonic_ns=header[4],
        frame_wall_ns=(header[5], header[6]),
        frame_monotonic_ns=(header[7], header[8]),
        source_sequence=(header[9], header[10]),
        valid=(bool(header[11]), bool(header[12])),
        age_s=(header[13], header[14]),
        period_s=(header[15], header[16]),
        magnetic=magnetic,
        temperature_x10=temperature,
    )


class PairedCsvLogger:
    """Write one left/right snapshot per publish tick with explicit identities."""

    def __init__(self, path: Path | None) -> None:
        self._stream = None
        self._writer = None
        self._rows = 0
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = path.open("x", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._stream, fieldnames=PAIR_COLUMNS)
            self._writer.writeheader()

    def write(self, sample: RawCaptureSample) -> None:
        if self._writer is None:
            return
        row: dict[str, Any] = {
            "publish_sequence": sample.sequence,
            "publish_wall_ns": sample.publish_wall_ns,
            "publish_monotonic_ns": sample.publish_monotonic_ns,
            "left_right_frame_skew_ns": sample.frame_skew_ns,
        }
        for side_index, side in enumerate(SIDES):
            row.update(
                {
                    f"{side}_valid": int(sample.valid[side_index]),
                    f"{side}_age_s": f"{sample.age_s[side_index]:.9f}",
                    f"{side}_sample_period_s": (
                        f"{sample.period_s[side_index]:.9f}"
                    ),
                    f"{side}_frame_wall_ns": sample.frame_wall_ns[side_index],
                    f"{side}_frame_monotonic_ns": (
                        sample.frame_monotonic_ns[side_index]
                    ),
                    f"{side}_source_sequence": sample.source_sequence[side_index],
                }
            )
            row.update(
                dict(
                    zip(
                        _temp_columns(side),
                        sample.temperature_x10[side_index].astype(int),
                    )
                )
            )
            row.update(
                dict(
                    zip(
                        _mag_columns(side),
                        sample.magnetic[side_index].reshape(-1).astype(int),
                    )
                )
            )
        self._writer.writerow(row)
        self._rows += 1
        if self._rows % 20 == 0:
            self._stream.flush()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
