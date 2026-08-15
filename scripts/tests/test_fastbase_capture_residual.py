from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from unitree_rl_lab.traction.fastbase_capture_residual import (
    FastBaseHallCaptureHighEndRecoveryRslModel,
    FastBaseHallCaptureResidual,
    FastBaseHallCaptureRslModel,
    FastBaseHallCaptureStabilityResidual,
    FastBaseHallCaptureStabilityRslModel,
    RslActorMean,
    trainable_parameters,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_TEACHER = (
    REPO_ROOT / "artifacts" / "hall_speed_demo" / "speedboost112_frozen_teacher.pt"
)
from unitree_rl_lab.traction.frozen_speedboost_teacher import (
    INPUT_DIM,
    OUTPUT_DIM,
    VALID_SLICE,
    FrozenSpeedBoostTeacher,
)


def test_initial_residual_is_exactly_zero_and_teacher_is_frozen():
    model = FastBaseHallCaptureResidual(FrozenSpeedBoostTeacher(), residual_limit=1.2)
    observation = torch.randn(3, INPUT_DIM)
    with torch.no_grad():
        assert torch.equal(model.capture_delta(observation), torch.zeros(3, OUTPUT_DIM))
        assert torch.allclose(model(observation), model.base_action(observation))
    assert not any(parameter.requires_grad for parameter in model.teacher.parameters())
    assert len(list(trainable_parameters(model))) > 0


def test_assume_fresh_changes_only_teacher_trailing_channels():
    model = FastBaseHallCaptureResidual(
        FrozenSpeedBoostTeacher(), teacher_trailing_mode="assume_fresh"
    )
    observation = torch.randn(2, INPUT_DIM)
    adapted = model.teacher_observation(observation)
    assert torch.equal(adapted[:, :-2], observation[:, :-2])
    assert torch.equal(adapted[:, -2:], torch.zeros(2, 2))
    assert torch.equal(observation[:, -2:], observation.clone()[:, -2:])


def test_rsl_actor_shape_and_residual_bound():
    actor = RslActorMean()
    observation = torch.randn(4, INPUT_DIM)
    assert actor(observation).shape == (4, OUTPUT_DIM)
    model = FastBaseHallCaptureResidual(FrozenSpeedBoostTeacher(), residual_limit=0.25)
    with torch.no_grad():
        model.residual[-1].bias.fill_(100.0)
        model.gate[-1].bias.fill_(100.0)
        delta = model.capture_delta(observation)
    assert delta.shape == (4, OUTPUT_DIM)
    assert torch.max(torch.abs(delta)) <= 0.250001


def _rsl_actor(
    batch: int = 3,
    *,
    gate_logit_scale: float = 1.0,
    gate_logit_bias: float = 0.0,
) -> tuple[FastBaseHallCaptureRslModel, TensorDict]:
    observation = TensorDict(
        {"policy": torch.randn(batch, INPUT_DIM) * 0.02}, batch_size=[batch]
    )
    observation["policy"][:, VALID_SLICE] = 1.0
    actor = FastBaseHallCaptureRslModel(
        observation,
        {"actor": ["policy"]},
        "actor",
        OUTPUT_DIM,
        teacher_checkpoint=str(FROZEN_TEACHER),
        residual_limit=0.55,
        gate_power=1.0,
        gate_logit_scale=gate_logit_scale,
        gate_logit_bias=gate_logit_bias,
        teacher_trailing_mode="assume_fresh",
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.08,
            "std_type": "scalar",
        },
    )
    return actor, observation


def _rsl_stability_actor(
    batch: int = 3,
) -> tuple[FastBaseHallCaptureStabilityRslModel, TensorDict]:
    observation = TensorDict(
        {"policy": torch.zeros(batch, INPUT_DIM)}, batch_size=[batch]
    )
    observation["policy"][:, VALID_SLICE] = 1.0
    observation["policy"][:, 29] = -1.0
    actor = FastBaseHallCaptureStabilityRslModel(
        observation,
        {"actor": ["policy"]},
        "actor",
        OUTPUT_DIM,
        teacher_checkpoint=str(FROZEN_TEACHER),
        residual_limit=0.55,
        gate_power=1.0,
        gate_logit_scale=2.75,
        gate_logit_bias=-3.2,
        teacher_trailing_mode="assume_fresh",
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.06,
            "std_type": "scalar",
        },
    )
    return actor, observation


def _rsl_high_end_recovery_actor(
    batch: int = 3,
) -> tuple[FastBaseHallCaptureHighEndRecoveryRslModel, TensorDict]:
    observation = TensorDict(
        {"policy": torch.zeros(batch, INPUT_DIM)}, batch_size=[batch]
    )
    observation["policy"][:, VALID_SLICE] = 1.0
    observation["policy"][:, 29] = -1.0
    actor = FastBaseHallCaptureHighEndRecoveryRslModel(
        observation,
        {"actor": ["policy"]},
        "actor",
        OUTPUT_DIM,
        teacher_checkpoint=str(FROZEN_TEACHER),
        residual_limit=0.55,
        gate_power=1.0,
        gate_logit_scale=2.75,
        gate_logit_bias=-3.2,
        teacher_trailing_mode="assume_fresh",
        stability_limit=1.25,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.08,
            "std_type": "scalar",
        },
    )
    return actor, observation


@pytest.mark.skipif(not FROZEN_TEACHER.is_file(), reason="converted speedboost112 artifact missing")
def test_native_rsl_gaussian_initial_mean_is_exact_frozen_base() -> None:
    actor, observation = _rsl_actor()
    deterministic = actor(observation)
    stochastic = actor(observation, stochastic_output=True)
    assert deterministic.shape == stochastic.shape == (3, OUTPUT_DIM)
    torch.testing.assert_close(
        deterministic,
        actor.mlp.base_action(observation["policy"]),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(actor.output_mean, deterministic)
    assert not any(parameter.requires_grad for parameter in actor.mlp.teacher.parameters())
    assert all(
        parameter.requires_grad
        for module in (actor.mlp.residual, actor.mlp.gate)
        for parameter in module.parameters()
    )


@pytest.mark.skipif(not FROZEN_TEACHER.is_file(), reason="converted speedboost112 artifact missing")
def test_gate_auxiliary_gradient_updates_gate_but_never_teacher() -> None:
    actor, observation = _rsl_actor()
    probability = actor.raw_capture_probability(observation)
    assert probability.shape == (3, 1)
    target = torch.tensor([[0.0], [1.0], [1.0]])
    torch.nn.functional.binary_cross_entropy(probability, target).backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in actor.mlp.gate.parameters()
    )
    assert all(parameter.grad is None for parameter in actor.mlp.teacher.parameters())


@pytest.mark.skipif(not FROZEN_TEACHER.is_file(), reason="converted speedboost112 artifact missing")
def test_complete_foot_outage_removes_all_residual_authority() -> None:
    actor, observation = _rsl_actor(
        gate_logit_scale=2.75, gate_logit_bias=-3.2
    )
    with torch.no_grad():
        actor.mlp.residual[-1].bias.fill_(100.0)
        actor.mlp.gate[-1].bias.fill_(100.0)
        healthy_delta = actor.mlp.capture_delta(observation["policy"])
        observation["policy"][0, VALID_SLICE.start] = 0.0
        failed_delta = actor.mlp.capture_delta(observation["policy"])
    assert torch.count_nonzero(healthy_delta[0]).item() == OUTPUT_DIM
    torch.testing.assert_close(failed_delta[0], torch.zeros(OUTPUT_DIM))


@pytest.mark.skipif(not FROZEN_TEACHER.is_file(), reason="converted speedboost112 artifact missing")
def test_native_rsl_checkpoint_and_jit_export_round_trip(tmp_path: Path) -> None:
    actor, observation = _rsl_actor(
        gate_logit_scale=2.75, gate_logit_bias=-3.2
    )
    with torch.no_grad():
        actor.mlp.residual[-1].bias.fill_(0.07)
        expected = actor(observation)
    # Calibration is part of the checkpoint because it changes deployment
    # behavior even though it is not an optimized tensor.
    restored, _ = _rsl_actor(
        gate_logit_scale=2.75, gate_logit_bias=-3.2
    )
    restored.load_state_dict(actor.state_dict(), strict=True)
    torch.testing.assert_close(restored(observation), expected)

    exported = torch.jit.script(restored.as_jit())
    torch.testing.assert_close(exported(observation["policy"]), expected)
    assert exported(observation["policy"][:2]).shape == (2, OUTPUT_DIM)

    onnx_wrapper = restored.as_onnx(verbose=False).cpu().eval()
    torch.testing.assert_close(onnx_wrapper(observation["policy"]), expected)
    ort = pytest.importorskip("onnxruntime")
    onnx_path = tmp_path / "calibrated_fastbase.onnx"
    torch.onnx.export(
        onnx_wrapper,
        observation["policy"],
        onnx_path,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    actual = torch.from_numpy(
        session.run(
            ["action"], {"observation": observation["policy"].numpy()}
        )[0]
    )
    torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)


@pytest.mark.skipif(not FROZEN_TEACHER.is_file(), reason="converted speedboost112 artifact missing")
def test_stability_actor_imports_capture_checkpoint_with_exact_action() -> None:
    old_actor, observation = _rsl_actor(
        gate_logit_scale=2.75, gate_logit_bias=-3.2
    )
    with torch.no_grad():
        old_actor.mlp.residual[-1].bias.fill_(0.03)
        old_actor.mlp.gate[-1].bias.fill_(0.4)
        expected = old_actor(observation)

    actor, _ = _rsl_stability_actor(batch=observation.batch_size[0])
    result = actor.load_state_dict(old_actor.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert actor.mlp.loaded_legacy_stability
    assert torch.count_nonzero(actor.mlp.stability_residual[-1].weight) == 0
    assert torch.count_nonzero(actor.mlp.stability_residual[-1].bias) == 0
    with torch.no_grad():
        actual = actor(observation)
    # This is an ABI migration, not an approximate warm start.
    assert torch.equal(actual, expected)


def test_stability_authority_is_observation_only_smooth_and_turn_aware() -> None:
    model = FastBaseHallCaptureStabilityResidual(
        FrozenSpeedBoostTeacher(), stability_limit=0.25
    )
    observation = torch.zeros(4, INPUT_DIM)
    observation[:, VALID_SLICE] = 1.0
    observation[:, 29] = -1.0
    observation[1, 1863] = 0.55
    observation[2, 1863] = 0.55
    observation[2, [32, 35, 38, 41, 44]] = 0.20
    observation[3, 27] = 0.25
    authority = model.stability_authority(observation).flatten()
    assert authority[0].item() == 0.0
    assert authority[1].item() == pytest.approx(1.0)
    # Reset-relative heading is deliberately disabled during a commanded turn.
    assert authority[2].item() == 0.0
    assert authority[3].item() == pytest.approx(1.0)
    assert model.stability_features(observation).shape == (4, 482)


def test_stability_delta_is_exact_zero_initially_and_bounded_when_trained() -> None:
    model = FastBaseHallCaptureStabilityResidual(
        FrozenSpeedBoostTeacher(), stability_limit=0.25
    )
    observation = torch.zeros(2, INPUT_DIM)
    observation[:, VALID_SLICE] = 1.0
    observation[:, 29] = -1.0
    observation[:, 1863] = 1.0
    with torch.no_grad():
        assert torch.equal(
            model.stability_delta(observation), torch.zeros(2, OUTPUT_DIM)
        )
        model.stability_residual[-1].bias.fill_(100.0)
        delta = model.stability_delta(observation)
    assert delta.shape == (2, OUTPUT_DIM)
    assert torch.max(torch.abs(delta)) <= 0.250001
    # Unlike the Hall capture residual, a complete foot outage cannot remove
    # proprioceptive fall-recovery authority.
    observation[0, VALID_SLICE] = 0.0
    with torch.no_grad():
        assert torch.count_nonzero(model.stability_delta(observation)[0]) == OUTPUT_DIM


@pytest.mark.skipif(not FROZEN_TEACHER.is_file(), reason="converted speedboost112 artifact missing")
def test_high_end_recovery_actor_freezes_existing_capture_branches_only() -> None:
    actor, observation = _rsl_high_end_recovery_actor()
    assert actor.mlp.freeze_capture_branches is True
    assert all(not parameter.requires_grad for parameter in actor.mlp.gate.parameters())
    assert all(not parameter.requires_grad for parameter in actor.mlp.residual.parameters())
    assert all(
        parameter.requires_grad
        for parameter in actor.mlp.stability_residual.parameters()
    )
    with torch.inference_mode():
        assert torch.equal(
            actor.mlp.capture_delta(observation["policy"]),
            torch.zeros(observation.batch_size[0], OUTPUT_DIM),
        )
        assert torch.equal(
            actor(observation), actor.mlp.base_action(observation["policy"])
        )


def test_high_anchor_path_excludes_stability_residual_gradient() -> None:
    model = FastBaseHallCaptureStabilityResidual(
        FrozenSpeedBoostTeacher(), stability_limit=0.25
    )
    observation = torch.zeros(3, INPUT_DIM)
    observation[:, VALID_SLICE] = 1.0
    observation[:, 29] = -1.0
    observation[:, 1863] = 1.0
    before = model.anchor_action_without_stability(observation).detach().clone()
    with torch.no_grad():
        model.stability_residual[-1].bias.fill_(0.4)
        composite = model(observation)
        after = model.anchor_action_without_stability(observation)
    assert torch.equal(before, after)
    assert not torch.equal(composite, after)

    model.zero_grad(set_to_none=True)
    model.anchor_action_without_stability(observation).square().mean().backward()
    assert all(
        parameter.grad is None
        for parameter in model.stability_residual.parameters()
    )


def test_gate_logit_calibration_formula_identity_and_state_schema() -> None:
    identity = FastBaseHallCaptureResidual(
        FrozenSpeedBoostTeacher(),
        gate_logit_scale=1.0,
        gate_logit_bias=0.0,
    )
    calibrated = FastBaseHallCaptureResidual(
        FrozenSpeedBoostTeacher(),
        gate_logit_scale=2.75,
        gate_logit_bias=-3.2,
    )
    assert identity.state_dict().keys() == calibrated.state_dict().keys()
    assert "gate_logit_scale" in calibrated.state_dict()
    assert "gate_logit_bias" in calibrated.state_dict()

    observation = torch.randn(7, INPUT_DIM) * 0.02
    observation[:, VALID_SLICE] = 1.0
    with torch.inference_mode():
        raw = calibrated.raw_capture_probability(observation)
        expected = torch.sigmoid(2.75 * torch.logit(raw.clamp(1.0e-6, 1.0 - 1.0e-6)) - 3.2)
        torch.testing.assert_close(
            calibrated.capture_probability(observation), expected
        )
        # The identity branch is deliberate: all old tasks preserve exact
        # floating-point output, not merely approximate equality.
        raw_identity = identity.raw_capture_probability(observation)
        assert torch.equal(identity.capture_probability(observation), raw_identity)
        torch.testing.assert_close(raw, raw_identity)

    # Saved values, not destination constructor defaults, determine behavior.
    restored_identity = FastBaseHallCaptureResidual(
        FrozenSpeedBoostTeacher(), gate_logit_scale=1.0, gate_logit_bias=0.0
    )
    restored_identity.load_state_dict(calibrated.state_dict(), strict=True)
    assert float(restored_identity.gate_logit_scale) == 2.75
    assert float(restored_identity.gate_logit_bias) == pytest.approx(-3.2)
    with torch.inference_mode():
        torch.testing.assert_close(
            restored_identity.capture_probability(observation),
            calibrated.capture_probability(observation),
        )


def test_legacy_checkpoint_injects_explicit_runner_calibration_once() -> None:
    legacy = FastBaseHallCaptureResidual(FrozenSpeedBoostTeacher()).state_dict()
    legacy.pop("gate_logit_scale")
    legacy.pop("gate_logit_bias")
    migrated = FastBaseHallCaptureResidual(
        FrozenSpeedBoostTeacher(), gate_logit_scale=2.75, gate_logit_bias=-3.2
    )
    migrated.load_state_dict(legacy, strict=True)
    assert migrated.loaded_legacy_calibration
    assert float(migrated.gate_logit_scale) == 2.75
    assert float(migrated.gate_logit_bias) == pytest.approx(-3.2)
    assert "gate_logit_scale" in migrated.state_dict()


@pytest.mark.parametrize(
    ("scale", "bias"),
    [(0.0, 0.0), (-1.0, 0.0), (float("nan"), 0.0), (1.0, float("inf"))],
)
def test_gate_logit_calibration_rejects_invalid_values(scale: float, bias: float) -> None:
    with pytest.raises(ValueError, match="gate_logit"):
        FastBaseHallCaptureResidual(
            FrozenSpeedBoostTeacher(),
            gate_logit_scale=scale,
            gate_logit_bias=bias,
        )
