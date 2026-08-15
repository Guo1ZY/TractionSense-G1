"""Regression tests for the deployable torque-traction observation contract."""

from __future__ import annotations

import numpy as np

from unitree_rl_lab.traction.schema import ACTION_DIM, G1_29DOF_JOINT_ORDER
from unitree_rl_lab.traction_torque.schema import (
    TORQUE_TRACTION_FRAME_SCHEMA,
    TORQUE_TRACTION_JOINT_INDICES,
    TORQUE_TRACTION_JOINT_ORDER,
    concatenate_frame,
)


def test_action_and_history_contract() -> None:
    assert ACTION_DIM == 29
    assert len(G1_29DOF_JOINT_ORDER) == 29
    assert TORQUE_TRACTION_FRAME_SCHEMA.frame_dimension == 125
    assert TORQUE_TRACTION_FRAME_SCHEMA.history_frames == 15
    assert TORQUE_TRACTION_FRAME_SCHEMA.flat_dimension == 1875
    assert TORQUE_TRACTION_FRAME_SCHEMA.flatten_order.startswith("time_major_oldest")


def test_leg_torque_indices_are_semantic_not_contiguous_assumption() -> None:
    actual = tuple(G1_29DOF_JOINT_ORDER[index] for index in TORQUE_TRACTION_JOINT_INDICES)
    assert actual == TORQUE_TRACTION_JOINT_ORDER
    assert TORQUE_TRACTION_JOINT_INDICES == (0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18)


def test_frame_concatenation_follows_schema() -> None:
    values: dict[str, np.ndarray] = {}
    offset = 0
    for term in TORQUE_TRACTION_FRAME_SCHEMA.terms:
        values[term.name] = np.full((2, term.dimension), offset, dtype=np.float32)
        offset += 1
    frame = concatenate_frame(values)
    assert frame.shape == (2, 125)
    for expected, term in enumerate(TORQUE_TRACTION_FRAME_SCHEMA.terms):
        assert np.all(frame[:, TORQUE_TRACTION_FRAME_SCHEMA.term_slice(term.name)] == expected)


def test_student_schema_explicitly_forbids_privileged_inputs() -> None:
    metadata = TORQUE_TRACTION_FRAME_SCHEMA.to_dict()
    names = {term["name"] for term in metadata["terms"]}
    forbidden = set(metadata["privileged_terms_forbidden"])
    assert not names.intersection(forbidden)

