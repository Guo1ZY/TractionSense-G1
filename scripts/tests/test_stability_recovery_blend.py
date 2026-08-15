"""Pure-CPU regression tests for the optional frozen recovery handoff."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from torch import nn

from unitree_rl_lab.traction.high_speed_stability_envelope import (
    EMERGENCY,
    LAST_ACTION_HISTORY_SLICE,
    NORMAL,
)
from unitree_rl_lab.traction.stability_recovery_blend import (
    FrozenStage7RecoveryActor,
    RECOVERY_COMMAND_VX_INDICES,
    RECOVERY_COMMAND_VY_INDICES,
    RECOVERY_COMMAND_YAW_INDICES,
    StabilityRecoveryBlend,
    StabilityRecoveryBlendCfg,
    rewrite_recovery_observation,
)


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "source"
    / "unitree_rl_lab"
    / "unitree_rl_lab"
    / "traction"
    / "stability_recovery_blend.py"
)
EVAL_PATH = ROOT / "scripts" / "rsl_rl" / "eval_spatial_friction_course.py"


class _RecordingRecovery(nn.Module):
    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.value = float(value)
        self.inputs: list[torch.Tensor] = []

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        self.inputs.append(observation.detach().clone())
        return torch.full(
            (observation.shape[0], 29),
            self.value,
            dtype=observation.dtype,
            device=observation.device,
        )


def _observation(num_envs: int = 1) -> torch.Tensor:
    observation = torch.arange(1864, dtype=torch.float32).repeat(num_envs, 1)
    observation[:, 335:480] = 0.0
    return observation


def test_recovery_rewrite_touches_only_commands_and_latest_actual_action() -> None:
    observation = _observation(2)
    previous = torch.stack((torch.arange(29), -torch.arange(29))).float()
    rewritten = rewrite_recovery_observation(
        observation,
        forward_command=0.16,
        actual_previous_action=previous,
    )

    assert torch.all(rewritten[:, list(RECOVERY_COMMAND_VX_INDICES)] == 0.16)
    assert torch.all(rewritten[:, list(RECOVERY_COMMAND_VY_INDICES)] == 0.0)
    assert torch.all(rewritten[:, list(RECOVERY_COMMAND_YAW_INDICES)] == 0.0)
    history = rewritten[:, LAST_ACTION_HISTORY_SLICE].reshape(2, 5, 29)
    torch.testing.assert_close(history[:, -1], previous)
    torch.testing.assert_close(history[:, :-1], torch.zeros_like(history[:, :-1]))

    allowed = set(RECOVERY_COMMAND_VX_INDICES)
    allowed.update(RECOVERY_COMMAND_VY_INDICES)
    allowed.update(RECOVERY_COMMAND_YAW_INDICES)
    allowed.update(range(451, 480))
    untouched = [index for index in range(1864) if index not in allowed]
    torch.testing.assert_close(rewritten[:, untouched], observation[:, untouched])


def test_blend_ramps_in_and_out_over_reviewed_times() -> None:
    recovery = _RecordingRecovery(1.0)
    blender = StabilityRecoveryBlend(
        recovery,
        num_envs=1,
        dt=0.02,
        device="cpu",
        cfg=StabilityRecoveryBlendCfg(
            blend_in_time_s=0.20,
            blend_out_time_s=0.30,
        ),
    )
    observation = torch.zeros((1, 1864))
    baseline = torch.zeros((1, 29))

    gates = []
    for _ in range(10):
        output = blender.update(observation, baseline, torch.tensor([EMERGENCY]))
        gates.append(output.gate.item())
    assert gates[0] == pytest.approx(0.1)
    assert gates[-1] == pytest.approx(1.0)
    torch.testing.assert_close(output.action, torch.ones_like(output.action))

    for _ in range(14):
        output = blender.update(observation, baseline, torch.tensor([NORMAL]))
        assert output.gate.item() > 0.0
    output = blender.update(observation, baseline, torch.tensor([NORMAL]))
    assert output.gate.item() == pytest.approx(0.0, abs=1.0e-6)
    torch.testing.assert_close(output.action, baseline)


def test_recovery_actor_receives_actual_blended_previous_action() -> None:
    recovery = _RecordingRecovery(1.0)
    blender = StabilityRecoveryBlend(
        recovery,
        num_envs=1,
        dt=0.02,
        device="cpu",
    )
    observation = torch.zeros((1, 1864))
    # Deliberately stale/fictitious actor history: the second call must replace
    # this newest sample with the action actually returned on call one.
    observation[:, 451:480] = -7.0
    baseline = torch.zeros((1, 29))
    first = blender.update(observation, baseline, torch.tensor([EMERGENCY]))
    second = blender.update(observation, baseline, torch.tensor([EMERGENCY]))

    assert first.gate.item() == pytest.approx(0.1)
    assert second.gate.item() == pytest.approx(0.2)
    torch.testing.assert_close(first.action, torch.full((1, 29), 0.1))
    latest_seen = recovery.inputs[-1][:, 451:480]
    torch.testing.assert_close(latest_seen, first.action)


def test_per_environment_reset_clears_gate_and_action_memory_only_for_ids() -> None:
    blender = StabilityRecoveryBlend(
        _RecordingRecovery(), num_envs=2, dt=0.02, device="cpu"
    )
    output = blender.update(
        torch.zeros((2, 1864)),
        torch.zeros((2, 29)),
        torch.tensor([EMERGENCY, EMERGENCY]),
    )
    assert torch.all(output.gate > 0.0)
    blender.reset(torch.tensor([1]))
    assert blender.gate[0] > 0.0
    assert blender.gate[1] == 0.0
    assert blender._has_last_action.tolist() == [True, False]


def test_reset_after_inference_mode_update_keeps_persistent_state_mutable() -> None:
    """Match Isaac's inference call followed by managed-reset bookkeeping."""

    blender = StabilityRecoveryBlend(
        _RecordingRecovery(), num_envs=2, dt=0.02, device="cpu"
    )
    with torch.inference_mode():
        output = blender.update(
            torch.zeros((2, 1864)),
            torch.zeros((2, 29)),
            torch.tensor([EMERGENCY, NORMAL]),
        )
    assert output.gate.tolist() == pytest.approx([0.1, 0.0])
    # This used to raise "in-place update to inference tensor outside
    # InferenceMode" after ``self.gate = torch.where(...)``.
    blender.reset(torch.tensor([0]))
    assert blender.gate.tolist() == pytest.approx([0.0, 0.0])
    assert blender._has_last_action.tolist() == [False, True]


def test_stage7_checkpoint_loader_is_strict_and_matches_actor_mean(tmp_path: Path) -> None:
    torch.manual_seed(7)
    original = FrozenStage7RecoveryActor()
    actor_state = {
        key: value.detach().clone() for key, value in original.state_dict().items()
    }
    # The deterministic evaluator ignores Gaussian sampling variance, but the
    # real legacy checkpoint contains this extra distribution tensor.
    actor_state["distribution.std_param"] = torch.full((29,), 0.22)
    checkpoint = tmp_path / "model_6149.pt"
    torch.save({"actor_state_dict": actor_state, "iter": 6149}, checkpoint)

    loaded = FrozenStage7RecoveryActor.from_checkpoint(checkpoint, device="cpu")
    observation = torch.randn((3, 1864))
    with torch.inference_mode():
        torch.testing.assert_close(loaded(observation), original(observation))
    assert not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())

    broken = tmp_path / "broken.pt"
    actor_state.pop("mlp.0.weight")
    torch.save({"actor_state_dict": actor_state}, broken)
    with pytest.raises(RuntimeError, match="mlp.0.weight"):
        FrozenStage7RecoveryActor.from_checkpoint(broken)


@pytest.mark.parametrize(
    "cfg",
    [
        StabilityRecoveryBlendCfg(blend_in_time_s=0.10),
        StabilityRecoveryBlendCfg(blend_out_time_s=0.40),
        StabilityRecoveryBlendCfg(recovery_forward_command=-0.01),
    ],
)
def test_invalid_configuration_fails_closed(cfg: StabilityRecoveryBlendCfg) -> None:
    with pytest.raises(ValueError):
        cfg.validate()


def test_module_has_deployable_inputs_only() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not imported_roots & {"isaaclab", "isaacsim", "omni", "pxr", "numpy"}
    update = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "update"
    )
    argument_names = {argument.arg for argument in update.args.args}
    assert argument_names == {
        "self",
        "policy_observation",
        "baseline_action",
        "stability_state",
    }


def test_evaluator_integration_is_explicitly_default_off() -> None:
    source = EVAL_PATH.read_text(encoding="utf-8")
    # Filled by the evaluator integration patch; these assertions prevent a
    # later refactor from silently enabling the recovery actor by default.
    assert '"--stability_recovery_checkpoint"' in source
    assert "stability_recovery_blend: StabilityRecoveryBlend | None = None" in source
    assert "recovery_output = stability_recovery_blend.update(" in source
