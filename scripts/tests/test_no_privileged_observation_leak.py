"""Static deployment boundary guard against privileged simulator inputs."""

from __future__ import annotations

import inspect

from unitree_rl_lab.traction_torque import networks
from unitree_rl_lab.traction_torque.governor import TorqueTractionCommandGovernor
from unitree_rl_lab.traction_torque.schema import TORQUE_TRACTION_FRAME_SCHEMA


def test_student_schema_and_forward_have_no_forbidden_input() -> None:
    forbidden = set(TORQUE_TRACTION_FRAME_SCHEMA.to_dict()["privileged_terms_forbidden"])
    schema_names = {term.name for term in TORQUE_TRACTION_FRAME_SCHEMA.terms}
    assert not forbidden.intersection(schema_names)
    signature = inspect.signature(networks.TorqueTractionStudentPolicy.forward)
    signature_names = set(signature.parameters)
    assert not forbidden.intersection(signature_names)


def test_governor_api_has_no_ground_truth_or_privileged_input() -> None:
    signature = inspect.signature(TorqueTractionCommandGovernor.update)
    names = set(signature.parameters)
    assert not {"ground_friction", "contact_sensor_force", "privileged_latent", "slip_label"}.intersection(names)
