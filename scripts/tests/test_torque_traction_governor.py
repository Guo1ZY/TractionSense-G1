"""Behavioral tests for native-signal command governance."""

from __future__ import annotations

import torch
import torch.nn as nn

from unitree_rl_lab.traction_torque.deployment import TorqueTractionPolicyRuntime
from unitree_rl_lab.traction_torque.governor import TorqueTractionCommandGovernor
from unitree_rl_lab.traction_torque.networks import TorqueTractionStudentOutput
from unitree_rl_lab.traction_torque.schema import TORQUE_TRACTION_FRAME_SCHEMA


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "raw_command": torch.tensor([[1.0, 0.4, 0.8]]),
        "slip_probability": torch.zeros(1, 2),
        "traction_utilization": torch.full((1, 2), 0.2),
        "traction_margin": torch.full((1, 2), 0.5),
        "contact_probability": torch.ones(1, 2),
        "estimator_confidence": torch.ones(1, 2),
        "foot_relative_velocity": torch.zeros(1, 2, 2),
        "slip_duration": torch.zeros(1, 2),
        "current_velocity": torch.zeros(1, 3),
    }


def test_disabled_governor_is_strict_control_group() -> None:
    governor = TorqueTractionCommandGovernor(1, enabled=False)
    output = governor.update(**_inputs())
    torch.testing.assert_close(output.adjusted_command, _inputs()["raw_command"])
    assert output.speed_scale.item() == 1.0
    assert output.state.item() == 0


def test_utilization_first_limits_acceleration_and_push_off() -> None:
    governor = TorqueTractionCommandGovernor(1)
    inputs = _inputs()
    for _ in range(6):
        governor.update(**inputs)
    normal = governor.update(**inputs)
    inputs["traction_utilization"].fill_(0.95)
    inputs["traction_margin"].fill_(0.0)
    for _ in range(5):
        limited = governor.update(**inputs)
    assert limited.acceleration_limit.item() < normal.acceleration_limit.item()
    assert limited.push_off_scale.item() < normal.push_off_scale.item()
    assert limited.speed_scale.item() > 0.5, "utilization alone must not force permanent crawl"


def test_persistent_slip_reduces_speed_yaw_lateral_and_low_confidence_falls_back() -> None:
    governor = TorqueTractionCommandGovernor(1)
    inputs = _inputs()
    inputs["slip_probability"].fill_(0.95)
    inputs["slip_duration"].fill_(0.3)
    inputs["foot_relative_velocity"].fill_(0.3)
    for _ in range(15):
        output = governor.update(**inputs)
    assert output.state.item() == 2
    assert output.speed_scale.item() < 0.5
    assert output.lateral_scale.item() < 0.6
    assert output.yaw_scale.item() < 0.6
    inputs["estimator_confidence"].zero_()
    fallback = governor.update(**inputs)
    assert fallback.state.item() == 3
    assert fallback.safety_flags.item() & 1


class _RuntimeProbePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.commands: list[torch.Tensor] = []

    def forward(self, history: torch.Tensor, adjusted_command=None) -> TorqueTractionStudentOutput:
        del adjusted_command
        command = history[:, -1, TORQUE_TRACTION_FRAME_SCHEMA.term_slice("command")]
        self.commands.append(command.clone())
        batch = history.shape[0]
        force = torch.zeros(batch, 6)
        contact = torch.ones(batch, 2)
        slip = torch.full((batch, 2), 0.95)
        utilization = torch.ones(batch, 2)
        margin = torch.full((batch, 2), -0.5)
        confidence = torch.ones(batch, 2)
        latent = torch.zeros(batch, 16)
        action = command[:, :1].expand(-1, 29)
        return TorqueTractionStudentOutput(action, force, contact, slip, utilization, margin, confidence, latent, torch.zeros(batch, 1))


def test_runtime_governor_rewrites_newest_command_for_preserved_actor() -> None:
    probe = _RuntimeProbePolicy()
    runtime = TorqueTractionPolicyRuntime(probe, torch.zeros(29))
    history = torch.zeros(1, 15, 125)
    history[:, -1, TORQUE_TRACTION_FRAME_SCHEMA.term_slice("command")] = torch.tensor([1.0, 0.3, 0.5])
    output = runtime.step(history, torch.zeros(1, 3))
    assert len(probe.commands) == 2
    torch.testing.assert_close(probe.commands[0], torch.tensor([[1.0, 0.3, 0.5]]))
    torch.testing.assert_close(probe.commands[1], output.governor.adjusted_command)
    assert output.governor.adjusted_command[0, 0] < 1.0
