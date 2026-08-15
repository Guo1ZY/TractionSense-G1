from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "unitree_rl_lab"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from unitree_rl_lab.traction import (  # noqa: E402
    CanonicalObservationBuilder,
    DualFootForceInput,
    IsaacForceAdapter,
    OfflineRecordedForceAdapter,
    ProprioceptiveState,
    TractionPolicyRuntime,
)
from unitree_rl_lab.traction.deployment import (  # noqa: E402
    DEFAULT_JOINT_POSITION,
    NOMINAL_ROBOT_MASS_KG,
)
from unitree_rl_lab.traction.schema import (  # noqa: E402
    TEMPORAL_STUDENT_FRAME_SCHEMA,
)


def _state(timestamp: float = 1.0) -> ProprioceptiveState:
    return ProprioceptiveState(
        timestamp=timestamp,
        base_angular_velocity=np.asarray([1.0, 2.0, 3.0]),
        projected_gravity=np.asarray([0.1, 0.2, -0.97]),
        joint_position=np.asarray(DEFAULT_JOINT_POSITION),
        joint_velocity=np.ones(29),
        previous_action=np.zeros(29),
        base_linear_velocity=np.asarray([0.2, 0.0, 0.0]),
    )


def test_observation_builder_applies_unique_order_scales_and_force_units() -> None:
    builder = CanonicalObservationBuilder()
    weight = NOMINAL_ROBOT_MASS_KG * 9.81
    force = IsaacForceAdapter.adapt(
        1.0,
        np.asarray([weight, 0.0, weight, 0.0, -weight, weight]),
    )
    history, flags = builder.append(_state(), force, np.asarray([0.8, 0.1, 0.2]))
    current = history.reshape(15, 106)[-1]
    assert not flags
    assert torch.allclose(current[0:3], torch.tensor([0.2, 0.4, 0.6]))
    assert torch.count_nonzero(current[6:35]) == 0
    assert torch.allclose(current[35:64], torch.full((29,), 0.05))
    assert torch.allclose(current[93:96], torch.tensor([0.8, 0.1, 0.2]))
    assert torch.allclose(
        current[96:102],
        torch.tensor([1.0, 0.0, 1.0, 0.0, -1.0, 1.0]),
    )
    assert torch.equal(current[102:104], torch.ones(2))
    assert torch.count_nonzero(history.reshape(15, 106)[:-1]) == 0


def test_builder_rejects_timestamp_rollback_and_marks_missing_foot() -> None:
    builder = CanonicalObservationBuilder()
    missing = DualFootForceInput(
        timestamp=1.0,
        left_force_xyz=np.zeros(3),
        right_force_xyz=np.zeros(3),
        left_valid=True,
        right_valid=False,
        left_age=0.0,
        right_age=1.0e6,
    )
    history, flags = builder.append(_state(), missing, np.zeros(3))
    assert "right_force_invalid" in flags
    assert history.reshape(15, 106)[-1, 105] == 1.0
    with pytest.raises(ValueError, match="did not increase"):
        builder.append(_state(), missing, np.zeros(3))


class _RiskPolicy:
    def __call__(self, history: torch.Tensor):
        current = history.reshape(1, 15, 106)[:, -1]
        action = torch.zeros((1, 29))
        action[:, :3] = current[:, 93:96]
        return (
            action,
            torch.full((1, 2), 0.9),
            torch.full((1, 1), 0.1),
            torch.ones((1, 1)),
        )


def test_runtime_uses_estimate_then_governed_command_for_fixed_policy() -> None:
    runtime = TractionPolicyRuntime(_RiskPolicy())
    force = IsaacForceAdapter.adapt(1.0, np.zeros(6))
    command = np.asarray([1.0, 0.4, 0.8])
    for index in range(17):
        output = runtime.step(
            _state(1.0 + index * 0.02),
            force,
            command,
        )
        if index < 14:
            assert output.governor.speed_scale.item() == 1.0
    assert output.governor.speed_scale.item() < 1.0
    assert torch.allclose(
        output.action[0, :3],
        output.governor.adjusted_command[0],
    )
    assert output.action.shape == (1, 29)
    assert output.joint_position_target.shape == (1, 29)


def test_runtime_governor_ablation_preserves_raw_command() -> None:
    runtime = TractionPolicyRuntime(_RiskPolicy(), governor_enabled=False)
    force = IsaacForceAdapter.adapt(1.0, np.zeros(6))
    command = np.asarray([1.0, 0.4, 0.8])
    output = runtime.step(_state(), force, command)
    assert output.governor.state.item() == 0
    assert output.governor.speed_scale.item() == 1.0
    assert torch.allclose(
        output.governor.adjusted_command[0],
        torch.from_numpy(command).float(),
    )
    assert torch.allclose(output.action[0, :3], torch.from_numpy(command).float())


def test_offline_adapter_preserves_force_valid_and_age(tmp_path: Path) -> None:
    path = tmp_path / "force.npz"
    np.savez(
        path,
        timestamp_s=np.asarray([[1.2]]),
        observed_force_normalized=np.ones((1, 6), dtype=np.float32),
        sensor_valid=np.asarray([[1.0, 0.0]], dtype=np.float32),
        sensor_age_s=np.asarray([[0.01, 0.20]], dtype=np.float32),
    )
    adapter = OfflineRecordedForceAdapter(
        path,
        force_key="observed_force_normalized",
        normalized_force=True,
    )
    sample = adapter.sample(0)
    assert np.allclose(sample.force_vector, NOMINAL_ROBOT_MASS_KG * 9.81)
    assert sample.left_valid and not sample.right_valid
    assert sample.right_age == pytest.approx(0.20)


def test_schema_frame_and_history_dimensions_remain_fixed() -> None:
    assert TEMPORAL_STUDENT_FRAME_SCHEMA.frame_dimension == 106
    assert TEMPORAL_STUDENT_FRAME_SCHEMA.history_frames == 15
    assert TEMPORAL_STUDENT_FRAME_SCHEMA.flat_dimension == 1590
