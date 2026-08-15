from __future__ import annotations

from pathlib import Path
import struct
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "unitree_rl_lab"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from unitree_rl_lab.traction.ble import (  # noqa: E402
    BleFrameParser,
    CHANNEL_COUNT,
    FRAME_LENGTH,
)
from unitree_rl_lab.traction.schema import (  # noqa: E402
    ACTION_DIM,
    FlatHistorySchema,
    ObservationTermSpec,
    TEMPORAL_STUDENT_FRAME_SCHEMA,
    legacy_actor_schema,
    legacy_critic_schema,
    old_to_new_flat_index,
)
from unitree_rl_lab.traction.sensor_layout import (  # noqa: E402
    PROVISIONAL_NORMALIZED_LAYOUT,
    DualFootSensorAggregator,
    SingleFootSensorAdapter,
)
from unitree_rl_lab.utils.partial_checkpoint import merge_state_dicts  # noqa: E402


def _wire_frame(
    *,
    xyz_offset: int = 0,
    temperature_x10: int = 253,
    reserved: int = 0xA5,
) -> bytes:
    def i16(value: int) -> int:
        return (value + 32768) % 65536 - 32768

    data = bytearray(FRAME_LENGTH)
    data[0:4] = bytes((0x7D, 0x00, 0xF0, 0x02))
    for channel in range(CHANNEL_COUNT):
        struct.pack_into(
            ">hhhh",
            data,
            4 + channel * 8,
            temperature_x10 + channel,
            i16(xyz_offset + channel),
            i16(xyz_offset - channel),
            i16(xyz_offset + 2 * channel),
        )
    data[-1] = reserved
    return bytes(data)


def test_audited_dimensions_and_action_order() -> None:
    assert ACTION_DIM == 29
    assert legacy_actor_schema(include_force=False).flat_dimension == 480
    assert legacy_actor_schema(include_force=True).flat_dimension == 510
    assert legacy_critic_schema(include_force=False).flat_dimension == 495
    assert legacy_critic_schema(include_force=True).flat_dimension == 525
    assert TEMPORAL_STUDENT_FRAME_SCHEMA.frame_dimension == 106
    assert TEMPORAL_STUDENT_FRAME_SCHEMA.history_frames == 15
    assert TEMPORAL_STUDENT_FRAME_SCHEMA.flat_dimension == 1590


def test_semantic_column_mapping_does_not_assume_append() -> None:
    old = FlatHistorySchema(
        name="old",
        history_frames=2,
        terms=(
            ObservationTermSpec("a", 2),
            ObservationTermSpec("b", 1),
        ),
    )
    new = FlatHistorySchema(
        name="new",
        history_frames=2,
        terms=(
            ObservationTermSpec("inserted", 3),
            ObservationTermSpec("b", 1),
            ObservationTermSpec("a", 2),
        ),
    )
    mapping = old_to_new_flat_index(old, new)
    # old: a0[0:2], a1[2:4], b0[4], b1[5]
    # new: inserted[0:6], b[6:8], a[8:12]
    np.testing.assert_array_equal(mapping, (8, 9, 10, 11, 6, 7))


def test_layout_is_explicit_and_only_topological() -> None:
    layout = PROVISIONAL_NORMALIZED_LAYOUT
    assert layout.sensor_names == tuple(f"P{i:02d}" for i in range(15))
    assert layout.positions_array.shape == (15, 2)
    assert layout.region_ids == (0,) * 5 + (1,) * 5 + (2,) * 5
    assert layout.ble_channel_to_sensor_index == tuple(range(15))
    assert layout.position_unit == "normalized_sole_length_width_from_a4_scan"
    assert layout.is_provisional
    # Region centers have deliberately nonuniform longitudinal separation.
    x = layout.positions_array[:, 0]
    fore_to_mid = x[:5].mean() - x[5:10].mean()
    mid_to_heel = x[5:10].mean() - x[10:].mean()
    assert not np.isclose(fore_to_mid, mid_to_heel)


def test_ble_big_endian_parse_and_stream_resynchronization() -> None:
    parser = BleFrameParser(foot_id="left")
    frame_bytes = _wire_frame()
    frames = parser.feed(b"\x01\x02bad" + frame_bytes[:41])
    assert not frames
    frames = parser.feed(frame_bytes[41:])
    assert len(frames) == 1
    frame = frames[0]
    assert frame.foot_id == "left"
    assert frame.sequence == 1
    assert frame.header_byte_1 == 0
    assert frame.hall_xyz.dtype == np.int32
    np.testing.assert_array_equal(frame.hall_xyz[7], (7, -7, 14))
    assert frame.temperature[0] == pytest.approx(25.3)
    assert frame.temperature[14] == pytest.approx(26.7)
    assert parser.last_reserved_byte == 0xA5
    assert parser.rejected_bytes == 5


def test_int16_unwrap_continues_across_wire_boundary() -> None:
    parser = BleFrameParser()
    first = parser.parse_frame(_wire_frame(xyz_offset=32760), timestamp=1.0)
    second = parser.parse_frame(_wire_frame(xyz_offset=-32760), timestamp=1.1)
    assert first is not None and second is not None
    # channel zero advances +16 counts rather than jumping -65520.
    assert int(second.hall_xyz[0, 0] - first.hall_xyz[0, 0]) == 16


def test_int16_unwrap_does_not_turn_an_ordinary_large_jump_into_a_wrap() -> None:
    parser = BleFrameParser()
    first = parser.parse_frame(_wire_frame(xyz_offset=30000), timestamp=1.0)
    second = parser.parse_frame(_wire_frame(xyz_offset=-10000), timestamp=1.1)
    assert first is not None and second is not None
    assert int(second.hall_xyz[0, 0]) == -10000


def test_single_foot_adapter_and_missing_foot_are_not_fabricated() -> None:
    parser = BleFrameParser()
    wire = parser.parse_frame(_wire_frame(xyz_offset=100), timestamp=10.0)
    assert wire is not None

    adapted = SingleFootSensorAdapter("left").adapt(wire, now=10.01)
    aggregator = DualFootSensorAggregator(timeout_s=0.1)
    aggregator.update(adapted)
    magnetic = aggregator.magnetic_input(10.02)
    assert magnetic.valid.tolist() == [1.0, 0.0]
    assert magnetic.hall_xyz.shape == (2, 15, 3)
    assert magnetic.temperature_c.shape == (2, 15)
    np.testing.assert_array_equal(magnetic.left_foot, adapted.hall_xyz)
    np.testing.assert_array_equal(magnetic.right_foot, np.zeros((15, 3)))

    # The Hall aggregate intentionally has no normal/tangential/net force API.
    assert not hasattr(magnetic, "force_vector")
    assert not hasattr(aggregator, "force_input")


def test_timestamp_regression_marks_frame_invalid() -> None:
    parser = BleFrameParser()
    first = parser.parse_frame(_wire_frame(), timestamp=10.0)
    second = parser.parse_frame(_wire_frame(), timestamp=9.0)
    assert first is not None and second is not None
    adapter = SingleFootSensorAdapter("left")
    assert adapter.adapt(first).valid
    assert not adapter.adapt(second).valid


def test_partial_checkpoint_uses_semantic_mapping_and_zero_new_columns() -> None:
    old_schema = legacy_actor_schema(include_force=False)
    new_schema = legacy_actor_schema(include_force=True)
    source = {"mlp.0.weight": np.arange(2 * 480, dtype=np.float32).reshape(2, 480)}
    current = {"mlp.0.weight": np.full((2, 510), np.nan, dtype=np.float32)}
    source_torch = {key: torch.as_tensor(value) for key, value in source.items()}
    current_torch = {key: torch.as_tensor(value) for key, value in current.items()}
    merged, stats = merge_state_dicts(current_torch, source_torch, verbose=False)
    mapping = old_to_new_flat_index(old_schema, new_schema)
    np.testing.assert_array_equal(merged["mlp.0.weight"][:, mapping].numpy(), source["mlp.0.weight"])
    new_columns = sorted(set(range(510)) - set(mapping.tolist()))
    np.testing.assert_array_equal(
        merged["mlp.0.weight"][:, new_columns].numpy(),
        np.zeros((2, 30), dtype=np.float32),
    )
    assert stats["mapping"][0]["source"] == "canonical_schema"
