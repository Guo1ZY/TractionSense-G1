"""Privileged-only Teacher observation contract."""

from __future__ import annotations

from unitree_rl_lab.traction.schema import ObservationTermSpec, PRIVILEGED_TRACTION_SCHEMA


TORQUE_TEACHER_PRIVILEGED_TERMS = (
    ObservationTermSpec("canonical_sim_privilege", PRIVILEGED_TRACTION_SCHEMA.flat_dimension),
    ObservationTermSpec("analytical_force", 6, "F_hat_N/(mass*9.81)"),
    ObservationTermSpec("estimated_contact_probability", 2),
    ObservationTermSpec("force_estimator_confidence", 2),
    ObservationTermSpec("force_residual_norm", 2, "Nm_scaled_0.05"),
    ObservationTermSpec("jacobian_condition_score", 2),
)
TORQUE_TEACHER_PRIVILEGED_DIM = sum(term.dimension for term in TORQUE_TEACHER_PRIVILEGED_TERMS)
TORQUE_TEACHER_FRAME_DIM = 96 + 3 + TORQUE_TEACHER_PRIVILEGED_DIM
TORQUE_TEACHER_HISTORY_FRAMES = 5
TORQUE_TEACHER_FLAT_DIM = TORQUE_TEACHER_FRAME_DIM * TORQUE_TEACHER_HISTORY_FRAMES

assert TORQUE_TEACHER_PRIVILEGED_DIM == 149
assert TORQUE_TEACHER_FRAME_DIM == 248
assert TORQUE_TEACHER_FLAT_DIM == 1240

