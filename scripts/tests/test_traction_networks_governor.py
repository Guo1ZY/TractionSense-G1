from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from tensordict import TensorDict


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "unitree_rl_lab"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from unitree_rl_lab.traction import (  # noqa: E402
    GatedTractionPolicy,
    DistillationLossCfg,
    PRIVILEGED_TRACTION_SCHEMA,
    TEMPORAL_STUDENT_FRAME_SCHEMA,
    TeacherTractionPolicy,
    TemporalStudentEncoderCfg,
    TemporalTactileProprioceptiveStudentEncoder,
    TractionAwareCommandGovernor,
    teacher_student_loss,
    temporal_history_to_legacy_proprio,
)
from unitree_rl_lab.traction.rsl_models import (  # noqa: E402
    CanonicalStudentRslModel,
    TEACHER_FLAT_DIM,
    TEACHER_FRAME_DIM,
    TEACHER_HISTORY_FRAMES,
    TractionTeacherCriticRslModel,
    TractionTeacherRslModel,
    teacher_history_to_legacy_observation,
)


def _history(batch: int = 4) -> torch.Tensor:
    history = torch.randn(
        (
            batch,
            TEMPORAL_STUDENT_FRAME_SCHEMA.history_frames,
            TEMPORAL_STUDENT_FRAME_SCHEMA.frame_dimension,
        )
    )
    history[..., 102:104] = 1.0
    history[..., 104:106] = 0.0
    return history


@pytest.mark.parametrize("variant", ("gru", "tcn"))
def test_student_variants_have_deployable_outputs(variant: str) -> None:
    encoder = TemporalTactileProprioceptiveStudentEncoder(
        TemporalStudentEncoderCfg(variant=variant)
    )
    output = encoder(_history())
    assert output.latent.shape == (4, 16)
    assert output.slip_probability.shape == (4, 2)
    assert output.traction_score.shape == (4, 1)
    assert output.sensor_confidence.shape == (4, 1)
    for value in (
        output.slip_probability,
        output.traction_score,
        output.sensor_confidence,
    ):
        assert torch.all((0.0 <= value) & (value <= 1.0))


def test_student_confidence_respects_invalid_and_stale_inputs() -> None:
    encoder = TemporalTactileProprioceptiveStudentEncoder()
    history = _history(2)
    history[0, -1, 102:104] = 0.0
    history[1, -1, 104:106] = 100.0
    confidence = encoder(history).sensor_confidence
    assert confidence[0, 0] == 0.0
    assert confidence[1, 0] < 1.0e-10


def test_teacher_privileged_schema_and_action_dimension() -> None:
    assert PRIVILEGED_TRACTION_SCHEMA.flat_dimension == 135
    teacher = TeacherTractionPolicy()
    output = teacher(
        torch.randn(3, 96),
        torch.randn(3, 3),
        torch.randn(3, PRIVILEGED_TRACTION_SCHEMA.flat_dimension),
    )
    assert output.action_mean.shape == (3, 29)
    assert output.latent.shape == (3, 16)


def test_zero_initialized_traction_branch_preserves_baseline_actor() -> None:
    policy = GatedTractionPolicy()
    baseline_observation = torch.randn(4, 480)
    history = _history()
    command = torch.randn(4, 3)
    with torch.no_grad():
        baseline_action = policy.baseline_actor(baseline_observation)
        output = policy(baseline_observation, history, command)
    assert torch.equal(output.action_mean, baseline_action)
    assert torch.all(output.traction_gate < 0.003)


def test_temporal_to_legacy_conversion_is_term_major() -> None:
    history = torch.arange(15 * 106, dtype=torch.float32).reshape(1, 15, 106)
    converted = temporal_history_to_legacy_proprio(history)
    assert converted.shape == (1, 480)
    recent = history[:, -5:]
    assert torch.equal(converted[:, 0:15], recent[..., 0:3].reshape(1, -1))
    assert torch.equal(converted[:, 30:45], recent[..., 93:96].reshape(1, -1))
    assert torch.equal(converted[:, 45:190], recent[..., 6:35].reshape(1, -1))


def test_distillation_losses_backpropagate_into_student() -> None:
    policy = GatedTractionPolicy()
    output = policy(torch.randn(8, 480), _history(8), torch.randn(8, 3))
    losses = teacher_student_loss(
        ppo_loss=output.action_mean.square().mean(),
        student=output,
        teacher_latent=torch.randn(8, 16),
        teacher_action=torch.randn(8, 29),
        slip_label=torch.randint(0, 2, (8, 2)),
        traction_target=torch.rand(8, 1),
        confidence_target=torch.ones(8, 1),
    )
    losses.total.backward()
    assert torch.isfinite(losses.total)
    assert policy.encoder.fusion[0].weight.grad is not None
    assert torch.isfinite(policy.encoder.fusion[0].weight.grad).all()
    assert losses.confidence > 0.0


def test_distillation_positive_weight_penalizes_missed_slip() -> None:
    policy = GatedTractionPolicy()
    output = policy(torch.randn(8, 480), _history(8), torch.randn(8, 3))
    labels = torch.zeros((8, 2))
    labels[0, 0] = 1.0
    common = {
        "ppo_loss": torch.zeros(()),
        "student": output,
        "teacher_latent": output.latent.detach(),
        "teacher_action": output.action_mean.detach(),
        "slip_label": labels,
        "traction_target": output.traction_score.detach(),
    }
    unweighted = teacher_student_loss(**common).slip
    weighted = teacher_student_loss(
        **common,
        cfg=DistillationLossCfg(slip_positive_weight=10.0),
    ).slip
    assert weighted > unweighted


def test_rsl_teacher_and_student_model_adapters() -> None:
    teacher_obs = TensorDict(
        {"policy": torch.randn(2, TEACHER_FLAT_DIM)},
        batch_size=[2],
    )
    teacher = TractionTeacherRslModel(
        teacher_obs,
        {"actor": ["policy"]},
        "actor",
        29,
        hidden_dims=[64, 32],
        activation="elu",
    )
    assert teacher(teacher_obs).shape == (2, 29)
    assert teacher.latest_traction_latent.shape == (2, 16)
    critic_obs = TensorDict(
        {"critic": torch.randn(2, TEACHER_FLAT_DIM)},
        batch_size=[2],
    )
    critic = TractionTeacherCriticRslModel(
        critic_obs,
        {"critic": ["critic"]},
        "critic",
        1,
        hidden_dims=[64, 32],
        activation="elu",
    )
    assert critic(critic_obs).shape == (2, 1)
    assert critic.latest_traction_latent.shape == (2, 16)

    student_obs = TensorDict(
        {
            "policy": torch.randn(
                2, TEMPORAL_STUDENT_FRAME_SCHEMA.flat_dimension
            )
        },
        batch_size=[2],
    )
    student_obs["policy"][:, -4:-2] = 1.0
    student_obs["policy"][:, -2:] = 0.0
    student = CanonicalStudentRslModel(
        student_obs,
        {"actor": ["policy"]},
        "actor",
        29,
        hidden_dims=[64, 32],
        activation="elu",
    )
    assert student(student_obs).shape == (2, 29)
    assert student.latest_slip_probability.shape == (2, 2)
    exported = student.as_onnx()
    action, slip, traction, confidence = exported(student_obs["policy"])
    assert action.shape == (2, 29)
    assert slip.shape == (2, 2)
    assert traction.shape == confidence.shape == (2, 1)


def test_teacher_history_conversion_matches_legacy_term_major_order() -> None:
    history = torch.arange(
        TEACHER_HISTORY_FRAMES * TEACHER_FRAME_DIM,
        dtype=torch.float32,
    ).reshape(1, TEACHER_HISTORY_FRAMES, TEACHER_FRAME_DIM)
    actor = teacher_history_to_legacy_observation(
        history,
        include_base_linear_velocity=False,
    )
    critic = teacher_history_to_legacy_observation(
        history,
        include_base_linear_velocity=True,
    )
    assert actor.shape == (1, 480)
    assert critic.shape == (1, 495)
    proprio = history[..., :96]
    assert torch.equal(actor[:, 0:15], proprio[..., 0:3].reshape(1, -1))
    assert torch.equal(actor[:, 30:45], proprio[..., 6:9].reshape(1, -1))
    assert torch.equal(actor[:, 45:190], proprio[..., 9:38].reshape(1, -1))
    assert torch.equal(critic[:, 15:], actor)


def _governor_inputs(num_envs: int = 1) -> dict[str, torch.Tensor]:
    return {
        "raw_command": torch.tensor([[1.0, 0.4, 0.8]]).repeat(num_envs, 1),
        "slip_probability": torch.zeros(num_envs, 2),
        "traction_score": torch.ones(num_envs, 1),
        "sensor_confidence": torch.ones(num_envs, 1),
        "slip_duration": torch.zeros(num_envs, 2),
        "current_velocity": torch.zeros(num_envs, 3),
    }


def test_governor_tracks_high_traction_command() -> None:
    governor = TractionAwareCommandGovernor(1)
    inputs = _governor_inputs()
    for _ in range(100):
        output = governor.update(**inputs)
    assert output.state.item() == 0
    assert output.speed_scale.item() == pytest.approx(1.0)
    assert torch.allclose(output.adjusted_command, inputs["raw_command"], atol=1.0e-5)


def test_governor_treats_healthy_nonunit_traction_margin_as_low_risk() -> None:
    governor = TractionAwareCommandGovernor(1)
    inputs = _governor_inputs()
    inputs["traction_score"].fill_(0.65)
    inputs["slip_probability"].fill_(0.15)
    for _ in range(100):
        output = governor.update(**inputs)
    assert output.state.item() == 0
    assert output.speed_scale.item() == pytest.approx(1.0)


def test_governor_limits_acceleration_lateral_yaw_and_persistent_slip() -> None:
    governor = TractionAwareCommandGovernor(1)
    inputs = _governor_inputs()
    normal = governor.update(**inputs)
    inputs["slip_probability"].fill_(0.9)
    inputs["traction_score"].fill_(0.1)
    for index in range(3):
        limited = governor.update(**inputs)
        if index < 2:
            assert limited.state.item() == 0
    assert limited.state.item() == 1
    assert limited.speed_scale.item() < normal.speed_scale.item()
    assert limited.acceleration_limit.item() < normal.acceleration_limit.item()
    assert limited.yaw_limit.item() < normal.yaw_limit.item()
    assert limited.push_off_scale.item() < normal.push_off_scale.item()

    inputs["slip_duration"].fill_(0.4)
    for _ in range(10):
        persistent = governor.update(**inputs)
    assert persistent.state.item() == 2
    assert persistent.speed_scale.item() <= 0.36
    assert abs(persistent.adjusted_command[0, 1]) < abs(inputs["raw_command"][0, 1])
    assert abs(persistent.adjusted_command[0, 2]) < abs(inputs["raw_command"][0, 2])


def test_governor_fast_down_slow_recovery_and_invalid_fallback() -> None:
    governor = TractionAwareCommandGovernor(1)
    inputs = _governor_inputs()
    inputs["slip_probability"].fill_(1.0)
    inputs["traction_score"].zero_()
    before = governor.speed_scale.item()
    for _ in range(3):
        down = governor.update(**inputs).speed_scale.item()
    down_change = before - down

    inputs["slip_probability"].zero_()
    inputs["traction_score"].fill_(1.0)
    # Clear minimum hold, then measure one recovery step.
    for _ in range(11):
        governor.update(**inputs)
    recovery_before = governor.speed_scale.item()
    recovery_after = governor.update(**inputs).speed_scale.item()
    assert recovery_after - recovery_before < down_change

    inputs["sensor_confidence"].zero_()
    fallback = governor.update(**inputs)
    assert fallback.state.item() == 3
    assert fallback.speed_scale.item() < recovery_after
