"""Atomic writer/reader for the controller's legacy F0T1 packet."""

from __future__ import annotations

import math
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path


MAGIC_F0T1 = 0x46305431
PACKET = struct.Struct("<IIQffffff")
FORCE_SCALE = 0.01


@dataclass(frozen=True)
class IpcSample:
    sequence: int
    stamp_ns: int
    contact_left: float
    contact_right: float
    normal_left_policy: float
    normal_right_policy: float
    tangent_left_policy: float
    tangent_right_policy: float


class F0T1Writer:
    def __init__(
        self,
        path: Path,
        *,
        contact_threshold_n: float = 5.0,
        max_force_n: float = 500.0,
    ) -> None:
        self.path = path
        self.contact_threshold_n = contact_threshold_n
        self.max_force_n = max_force_n
        self.sequence = 0

    def write(
        self,
        normal_left_n: float,
        normal_right_n: float,
        tangent_left_n: float = 0.0,
        tangent_right_n: float = 0.0,
    ) -> IpcSample:
        values = [normal_left_n, normal_right_n, tangent_left_n, tangent_right_n]
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
            raise ValueError("all force values must be finite and non-negative")
        nl, nr, tl, tr = [
            min(float(value), self.max_force_n) * FORCE_SCALE for value in values
        ]
        sample = IpcSample(
            sequence=self.sequence,
            stamp_ns=time.time_ns(),
            contact_left=float(normal_left_n >= self.contact_threshold_n),
            contact_right=float(normal_right_n >= self.contact_threshold_n),
            normal_left_policy=nl,
            normal_right_policy=nr,
            tangent_left_policy=tl,
            tangent_right_policy=tr,
        )
        payload = PACKET.pack(
            MAGIC_F0T1,
            sample.sequence,
            sample.stamp_ns,
            sample.contact_left,
            sample.contact_right,
            sample.normal_left_policy,
            sample.normal_right_policy,
            sample.tangent_left_policy,
            sample.tangent_right_policy,
        )
        atomic_write(self.path, payload)
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        return sample


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_packet(path: Path) -> IpcSample:
    payload = path.read_bytes()
    if len(payload) != PACKET.size:
        raise ValueError(f"expected {PACKET.size} bytes, got {len(payload)}")
    values = PACKET.unpack(payload)
    if values[0] != MAGIC_F0T1:
        raise ValueError("bad F0T1 magic")
    return IpcSample(
        sequence=values[1],
        stamp_ns=values[2],
        contact_left=values[3],
        contact_right=values[4],
        normal_left_policy=values[5],
        normal_right_policy=values[6],
        tangent_left_policy=values[7],
        tangent_right_policy=values[8],
    )

