"""CPU contracts for LOW-only model6149 residual distillation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from unitree_rl_lab.traction.anchored_ppo import (
    CAPTURE_GATE_OPTIMIZER_ROLE,
    CAPTURE_RESIDUAL_OPTIMIZER_ROLE,
    OPTIMIZER_ROLE_KEY,
    SPATIAL_HIGH_START,
    SPATIAL_HIGH_END,
    SPATIAL_LOW,
    AnchoredPPO,
    AnchoredRolloutStorage,
    FrozenLowExpertResidualTargetBuilder,
    balanced_masked_stage_bce,
    capture_residual_has_zero_output,
    masked_low_expert_smooth_l1,
    stage_auxiliary_logits,
    stage_auxiliary_targets,
)
from unitree_rl_lab.traction.fastbase_capture_residual import (
    FastBaseHallCaptureRslModel,
)
from unitree_rl_lab.traction.frozen_low_expert import (
    LOW_EXPERT_COMMAND,
    TERM_MAJOR_COMMAND_SLICE,
    load_frozen_low_recovery_expert,
    rewrite_term_major_velocity_command,
)
from unitree_rl_lab.traction.frozen_speedboost_teacher import (
    INPUT_DIM,
    OUTPUT_DIM,
    VALID_SLICE,
)


ROOT = Path(__file__).resolve().parents[2]
FROZEN_SPEEDBOOST = (
    ROOT / "artifacts" / "hall_speed_demo" / "speedboost112_frozen_teacher.pt"
)
LOW_EXPERT = (
    ROOT
    / "logs"
    / "rsl_rl"
    / "unitree_g1_29dof_velocity_foot_traction_hall_handoff_recovery"
    / "2026-08-10_13-31-15_stage7a_handoff_mild_mu018_026"
    / "model_6149.pt"
)
LOW_EXPERT_SHA256 = (
    "2cef06ae9d189a4cb3ef22de4ce24be0d780f49007b0ee9ce2c897ff8a66f1ec"
)
FASTBASE_MODEL49 = (
    ROOT
    / "logs"
    / "rsl_rl"
    / "unitree_g1_29dof_velocity_foot_traction_hall_spatial_fastbase_capture"
    / "2026-08-10_20-02-07_fastbase_gate_warmup_medium_r3"
    / "model_49.pt"
)


def test_counterfactual_command_rewrite_is_exact_term_major_30_45() -> None:
    observation = torch.arange(2 * INPUT_DIM, dtype=torch.float32).reshape(
        2, INPUT_DIM
    )
    original = observation.clone()
    rewritten = rewrite_term_major_velocity_command(observation)
    assert torch.equal(observation, original)
    assert torch.equal(rewritten[:, :30], original[:, :30])
    assert torch.equal(rewritten[:, 45:], original[:, 45:])
    expected = torch.tensor(LOW_EXPERT_COMMAND).view(1, 1, 3).expand(2, 5, 3)
    torch.testing.assert_close(
        rewritten[:, TERM_MAJOR_COMMAND_SLICE].reshape(2, 5, 3), expected
    )


class _RecordingExpert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.inputs: list[torch.Tensor] = []

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.inputs.append(observation.detach().clone())
        return torch.full(
            (observation.shape[0], OUTPUT_DIM),
            0.5,
            device=observation.device,
            dtype=observation.dtype,
        )


class _RecordingBase:
    def __init__(self) -> None:
        self.calls = 0
        self.inputs: list[torch.Tensor] = []

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.inputs.append(observation.detach().clone())
        return torch.full(
            (observation.shape[0], OUTPUT_DIM),
            0.1,
            device=observation.device,
            dtype=observation.dtype,
        )


def test_low_expert_builder_is_low_valid_once_and_uses_original_base_command() -> None:
    observation = torch.zeros(4, INPUT_DIM)
    observation[:, TERM_MAJOR_COMMAND_SLICE] = (
        torch.tensor((0.8, 0.0, 0.0)).view(1, 1, 3).expand(4, 5, 3).reshape(4, -1)
    )
    observation[:, VALID_SLICE] = torch.tensor(
        ((1.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.6, 0.7))
    )
    stage = torch.tensor(
        (SPATIAL_LOW, SPATIAL_HIGH_START, SPATIAL_LOW, SPATIAL_LOW)
    )
    expert = _RecordingExpert()
    base = _RecordingBase()
    builder = FrozenLowExpertResidualTargetBuilder(
        expert, frozen_base_action=base, target_cap=0.20
    )
    batch = builder.build(observation, stage=stage)

    assert expert.calls == base.calls == builder.inference_calls == 1
    assert builder.cache_writes == 1
    assert builder.rows_inferred == 2
    assert torch.equal(batch.mask[:, 0], torch.tensor((True, False, False, True)))
    torch.testing.assert_close(
        batch.target[[0, 3]], torch.full((2, OUTPUT_DIM), 0.20)
    )
    torch.testing.assert_close(batch.target[[1, 2]], torch.zeros(2, OUTPUT_DIM))
    expert_command = expert.inputs[0][:, TERM_MAJOR_COMMAND_SLICE].reshape(2, 5, 3)
    torch.testing.assert_close(
        expert_command,
        torch.tensor(LOW_EXPERT_COMMAND).view(1, 1, 3).expand(2, 5, 3),
    )
    # The frozen deployment base must keep the original 0.8 command.  Rewriting
    # both branches would learn the wrong counterfactual residual.
    base_command = base.inputs[0][:, TERM_MAJOR_COMMAND_SLICE].reshape(2, 5, 3)
    torch.testing.assert_close(
        base_command,
        torch.tensor((0.8, 0.0, 0.0)).view(1, 1, 3).expand(2, 5, 3),
    )

    # No eligible row means no additional expert/base inference, but the
    # rollout cache is still written exactly once for this step.
    empty = builder.build(
        observation, stage=torch.full((4,), SPATIAL_HIGH_START)
    )
    assert expert.calls == base.calls == builder.inference_calls == 1
    assert builder.cache_writes == 2
    assert not empty.mask.any()


def test_masked_smooth_l1_beta_005_and_residual_only_gradient() -> None:
    prediction = torch.zeros(2, OUTPUT_DIM, requires_grad=True)
    target = torch.full((2, OUTPUT_DIM), 0.10)
    mask = torch.tensor(((True,), (False,)))
    result = masked_low_expert_smooth_l1(
        prediction, target, mask, beta=0.05
    )
    # |e| > beta: SmoothL1 = |e| - beta/2 = 0.075.
    assert result.total.item() == pytest.approx(0.075)
    assert result.valid_fraction.item() == pytest.approx(0.5)
    assert result.mean_abs_target.item() == pytest.approx(0.10)
    result.total.backward()
    assert torch.count_nonzero(prediction.grad[0]).item() == OUTPUT_DIM
    assert torch.count_nonzero(prediction.grad[1]).item() == 0


@pytest.mark.skipif(
    not LOW_EXPERT.is_file(), reason="frozen model6149 expert checkpoint missing"
)
def test_model6149_loader_is_strict_frozen_and_finite() -> None:
    expert = load_frozen_low_recovery_expert(
        LOW_EXPERT, expected_sha256=LOW_EXPERT_SHA256
    )
    assert not expert.training
    assert all(not parameter.requires_grad for parameter in expert.parameters())
    with torch.inference_mode():
        output = expert(torch.zeros(3, INPUT_DIM))
    assert output.shape == (3, OUTPUT_DIM)
    assert torch.isfinite(output).all()
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_frozen_low_recovery_expert(
            LOW_EXPERT, expected_sha256="0" * 64
        )


def _make_actor_and_observation(batch: int = 3):
    observation = TensorDict(
        {"policy": torch.randn(batch, INPUT_DIM) * 0.02}, batch_size=[batch]
    )
    observation["policy"][:, VALID_SLICE] = 1.0
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = FastBaseHallCaptureRslModel(
        observation,
        groups,
        "actor",
        OUTPUT_DIM,
        teacher_checkpoint=str(FROZEN_SPEEDBOOST),
        residual_limit=0.55,
        gate_logit_scale=2.75,
        gate_logit_bias=-3.2,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.08,
            "std_type": "scalar",
        },
    )
    return actor, observation, groups


@pytest.mark.skipif(
    not FROZEN_SPEEDBOOST.is_file() or not LOW_EXPERT.is_file(),
    reason="frozen Teacher/expert artifact missing",
)
def test_expert_is_algorithm_only_and_distillation_gradient_is_residual_only() -> None:
    actor, observation, groups = _make_actor_and_observation()
    with torch.inference_mode():
        action_before = actor(observation).clone()
    actor_keys_before = tuple(actor.state_dict())
    critic = MLPModel(observation, groups, "critic", 1, hidden_dims=[16, 8])
    storage = AnchoredRolloutStorage(
        "rl", 3, 2, observation, [OUTPUT_DIM], "cpu"
    )
    private_env = SimpleNamespace(
        spatial_course_stage_buf=torch.tensor(
            (SPATIAL_LOW, SPATIAL_HIGH_START, SPATIAL_LOW)
        ),
        _hall_foot_packet_cache={"age": torch.zeros(3, 2)},
        episode_length_buf=torch.full((3,), 5, dtype=torch.long),
    )
    algorithm = AnchoredPPO(
        actor,
        critic,
        storage,
        env=private_env,
        anchor_teacher_checkpoint=str(FROZEN_SPEEDBOOST),
        device="cpu",
        schedule="fixed",
        learning_rate=5.0e-6,
        capture_gate_warmup_updates=0,
        capture_residual_learning_rate=2.0e-5,
        capture_residual_max_grad_norm=0.1,
        low_expert_checkpoint=str(LOW_EXPERT),
        low_expert_expected_sha256=LOW_EXPERT_SHA256,
        low_expert_distillation_loss_coef=0.25,
        low_expert_target_cap=0.20,
        low_expert_smooth_l1_beta=0.05,
        low_expert_command=LOW_EXPERT_COMMAND,
    )
    assert algorithm.low_expert_builder is not None
    expert = algorithm.low_expert_builder.expert
    expert_ids = {id(parameter) for parameter in expert.parameters()}
    actor_ids = {id(parameter) for parameter in actor.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in algorithm.optimizer.param_groups
        for parameter in group["params"]
    }
    assert not expert_ids & actor_ids
    assert not expert_ids & optimizer_ids
    assert all(not parameter.requires_grad for parameter in expert.parameters())
    assert tuple(actor.state_dict()) == actor_keys_before
    with torch.inference_mode():
        torch.testing.assert_close(actor(observation), action_before)

    # One algorithm rollout step invokes model6149 once on the eligible LOW
    # subset and stores detached targets/masks on the private transition.
    assert algorithm.low_expert_builder.inference_calls == 0
    algorithm.act(observation)
    assert algorithm.low_expert_builder.inference_calls == 1
    assert algorithm.low_expert_builder.cache_writes == 1
    assert algorithm.low_expert_builder.rows_inferred == 2
    assert torch.equal(
        algorithm.transition.low_expert_residual_mask[:, 0],
        torch.tensor((True, False, True)),
    )
    assert not algorithm.transition.low_expert_residual_target.requires_grad

    prediction = algorithm._ungated_capture_residual(observation)
    target = torch.full_like(prediction, 0.10)
    mask = torch.tensor(((True,), (True,), (False,)))
    loss = masked_low_expert_smooth_l1(
        prediction, target, mask, beta=0.05
    ).total
    algorithm.optimizer.zero_grad()
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
        for parameter in algorithm.capture_residual_parameters
    )
    assert all(parameter.grad is None for parameter in algorithm.capture_gate_parameters)
    assert all(parameter.grad is None for parameter in actor.mlp.teacher.parameters())
    assert all(parameter.grad is None for parameter in expert.parameters())

    # Complete two synthetic rollout steps and one CPU PPO update.  model6149
    # must be called only during rollout (twice), never once per minibatch or
    # epoch; the cached loss then updates the residual and reports diagnostics.
    residual_before_update = {
        name: value.detach().clone()
        for name, value in algorithm.capture_residual.state_dict().items()
    }
    zeros = torch.zeros(3)
    algorithm.process_env_step(
        observation, zeros, torch.zeros(3, dtype=torch.long), {}
    )
    algorithm.act(observation)
    algorithm.process_env_step(
        observation, zeros, torch.zeros(3, dtype=torch.long), {}
    )
    assert algorithm.low_expert_builder.inference_calls == 2
    algorithm.compute_returns(observation)
    metrics = algorithm.update()
    assert algorithm.low_expert_builder.inference_calls == 2
    assert metrics["low_expert_distillation"] > 0.0
    assert metrics["low_expert_valid_fraction"] > 0.0
    assert metrics["low_expert_abs_target"] > 0.0
    assert metrics["low_expert_raw_gate"] > 0.0
    assert metrics["low_expert_calibrated_gate"] >= 0.0
    assert metrics["low_expert_effective_gate"] >= 0.0
    assert metrics["low_expert_ungated_residual_abs"] >= 0.0
    assert metrics["low_expert_effective_delta_abs"] >= 0.0
    assert metrics["high_effective_delta_abs"] >= 0.0
    assert metrics["residual_lr"] == pytest.approx(2.0e-5)
    assert any(
        not torch.equal(residual_before_update[name], value)
        for name, value in algorithm.capture_residual.state_dict().items()
    )

    payload = algorithm.save()
    assert not any("expert" in key for key in payload if key != "high_friction_anchor")
    assert not any("expert" in key for key in payload["actor_state_dict"])
    assert payload["high_friction_anchor"]["low_expert_distillation"][
        "expert_in_policy_export"
    ] is False
    residual_group = next(
        group
        for group in algorithm.optimizer.param_groups
        if group[OPTIMIZER_ROLE_KEY] == CAPTURE_RESIDUAL_OPTIMIZER_ROLE
    )
    assert residual_group["lr"] == pytest.approx(2.0e-5)
    assert algorithm.capture_residual_max_grad_norm == pytest.approx(0.1)

    restored_actor, restored_observation, restored_groups = (
        _make_actor_and_observation()
    )
    restored = AnchoredPPO(
        restored_actor,
        MLPModel(
            restored_observation,
            restored_groups,
            "critic",
            1,
            hidden_dims=[16, 8],
        ),
        AnchoredRolloutStorage(
            "rl", 3, 2, restored_observation, [OUTPUT_DIM], "cpu"
        ),
        env=private_env,
        anchor_teacher_checkpoint=str(FROZEN_SPEEDBOOST),
        device="cpu",
        schedule="fixed",
        learning_rate=5.0e-6,
        capture_gate_warmup_updates=0,
        capture_residual_learning_rate=2.0e-5,
        capture_residual_max_grad_norm=0.1,
        low_expert_checkpoint=str(LOW_EXPERT),
        low_expert_expected_sha256=LOW_EXPERT_SHA256,
        low_expert_distillation_loss_coef=0.25,
        low_expert_target_cap=0.20,
        low_expert_smooth_l1_beta=0.05,
        low_expert_command=LOW_EXPERT_COMMAND,
    )
    assert restored.load(payload, None, strict=True)
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(
            value, restored_actor.state_dict()[name], atol=0.0, rtol=0.0
        )
    assert restored.low_expert_builder is not None
    assert restored.low_expert_builder.inference_calls == 0
    assert all(
        not parameter.requires_grad
        for parameter in restored.low_expert_builder.expert.parameters()
    )

    exported = actor.as_onnx(verbose=False)
    assert not any("expert" in name.lower() for name, _ in exported.named_modules())
    torch.testing.assert_close(exported(observation["policy"]), actor(observation))


@pytest.mark.skipif(
    not FROZEN_SPEEDBOOST.is_file() or not LOW_EXPERT.is_file(),
    reason="frozen Teacher/expert artifact missing",
)
def test_strong_direction_offline_fit_drops_loss_without_gate_or_teacher_gradient() -> None:
    """Prove the strong runner's private residual update on fixed CPU data.

    This deliberately uses the real model6149 counterfactual target and the
    same residual Adam role/gradient clipping code as PPO.  Fifty small-batch
    steps are a conservative proxy for the many residual optimizer steps in
    25--50 online PPO updates; the purpose is to catch missing gradients,
    wrong optimizer identity and accidental gate/Teacher coupling before an
    Isaac smoke run.
    """

    torch.manual_seed(123)
    actor, observation, groups = _make_actor_and_observation(batch=16)
    observation["policy"][:, TERM_MAJOR_COMMAND_SLICE] = (
        torch.tensor((0.8, 0.0, 0.0))
        .view(1, 1, 3)
        .expand(16, 5, 3)
        .reshape(16, -1)
    )
    observation["policy"][:, VALID_SLICE] = 1.0
    private_env = SimpleNamespace(
        spatial_course_stage_buf=torch.full((16,), SPATIAL_LOW),
        _hall_foot_packet_cache={"age": torch.zeros(16, 2)},
        episode_length_buf=torch.full((16,), 5, dtype=torch.long),
    )
    algorithm = AnchoredPPO(
        actor,
        MLPModel(observation, groups, "critic", 1, hidden_dims=[16, 8]),
        AnchoredRolloutStorage("rl", 16, 2, observation, [OUTPUT_DIM], "cpu"),
        env=private_env,
        anchor_teacher_checkpoint=str(FROZEN_SPEEDBOOST),
        device="cpu",
        schedule="fixed",
        learning_rate=5.0e-6,
        # Deliberately non-unit so the gradient-isolation contract also proves
        # that the configured BCE coefficient is applied exactly once.
        stage_aux_loss_coef=0.37,
        capture_gate_warmup_updates=0,
        capture_gate_gradient_mode="stage_bce_only",
        capture_residual_learning_rate=1.0e-4,
        capture_residual_max_grad_norm=0.5,
        low_expert_checkpoint=str(LOW_EXPERT),
        low_expert_expected_sha256=LOW_EXPERT_SHA256,
        low_expert_distillation_loss_coef=1.0,
        low_expert_target_cap=0.20,
        low_expert_smooth_l1_beta=0.05,
        low_expert_command=LOW_EXPERT_COMMAND,
        low_expert_residual_gradient_mode="supervised_only",
    )
    assert algorithm.low_expert_builder is not None
    cached = algorithm.low_expert_builder.build(
        observation["policy"], stage=private_env.spatial_course_stage_buf
    )
    assert cached.mask.all()
    assert cached.target.abs().mean().item() > 0.10

    gate_before = [
        parameter.detach().clone()
        for parameter in algorithm.capture_gate_parameters
    ]
    teacher_before = [
        parameter.detach().clone()
        for parameter in actor.mlp.teacher.parameters()
    ]
    expert_before = [
        parameter.detach().clone()
        for parameter in algorithm.low_expert_builder.expert.parameters()
    ]

    # The fail-closed production backward must replace an intentionally
    # gate-opening PPO gradient with raw HIGH BCE, while independently replacing
    # the opposing PPO residual gradient with HIGH-anchor + LOW-expert.
    algorithm.optimizer.zero_grad(set_to_none=True)
    reference_gate_logits, reference_gate_source = stage_auxiliary_logits(
        actor, observation, None
    )
    assert reference_gate_source == "actor_raw_capture_gate"
    high_end_targets = stage_auxiliary_targets(
        torch.full((16,), SPATIAL_HIGH_END),
        torch.full((16,), 5, dtype=torch.long),
        reset_mask_steps=1,
        high_end_weight=4.0,
    )
    reference_stage_bce = balanced_masked_stage_bce(
        reference_gate_logits,
        high_end_targets.label,
        high_end_targets.mask,
        high_end_targets.weight,
    ).total
    (algorithm.stage_aux_loss_coef * reference_stage_bce).backward()
    reference_gate_gradients = [
        parameter.grad.detach().clone()
        for parameter in algorithm.capture_gate_parameters
    ]
    algorithm.optimizer.zero_grad(set_to_none=True)
    reference = masked_low_expert_smooth_l1(
        algorithm._ungated_capture_residual(observation),
        cached.target,
        cached.mask,
        beta=algorithm.low_expert_smooth_l1_beta,
    ).total
    reference.backward()
    reference_residual_gradients = [
        parameter.grad.detach().clone()
        for parameter in algorithm.capture_residual_parameters
    ]
    algorithm.optimizer.zero_grad(set_to_none=True)
    opposing_prediction = algorithm._ungated_capture_residual(observation)
    primary_loss = (
        (opposing_prediction - 0.40).square().mean()
        - actor.raw_capture_probability(observation).mean()
    )
    anchor_zero = algorithm._ungated_capture_residual(observation).sum() * 0.0
    stage_bce = balanced_masked_stage_bce(
        stage_auxiliary_logits(actor, observation, None)[0],
        high_end_targets.label,
        high_end_targets.mask,
        high_end_targets.weight,
    ).total
    isolated_expert = masked_low_expert_smooth_l1(
        algorithm._ungated_capture_residual(observation),
        cached.target,
        cached.mask,
        beta=algorithm.low_expert_smooth_l1_beta,
    ).total
    algorithm._backward_policy_and_low_expert(
        primary_loss,
        anchor_zero,
        stage_bce,
        isolated_expert,
        gate_supervision_has_rows=True,
        residual_supervision_has_rows=True,
    )
    for expected, parameter in zip(
        reference_residual_gradients, algorithm.capture_residual_parameters
    ):
        torch.testing.assert_close(parameter.grad, expected, atol=1.0e-8, rtol=1.0e-6)
    for expected, parameter in zip(
        reference_gate_gradients, algorithm.capture_gate_parameters
    ):
        torch.testing.assert_close(parameter.grad, expected, atol=1.0e-8, rtol=1.0e-6)
    assert any(gradient.abs().sum().item() > 0.0 for gradient in reference_gate_gradients)
    assert all(parameter.grad is None for parameter in actor.mlp.teacher.parameters())
    assert all(
        parameter.grad is None
        for parameter in algorithm.low_expert_builder.expert.parameters()
    )
    algorithm.optimizer.zero_grad(set_to_none=True)

    losses: list[float] = []
    for _ in range(50):
        distillation = masked_low_expert_smooth_l1(
            algorithm._ungated_capture_residual(observation),
            cached.target,
            cached.mask,
            beta=algorithm.low_expert_smooth_l1_beta,
        )
        losses.append(float(distillation.total.detach()))
        algorithm._backward_policy_and_low_expert(
            next(algorithm.critic.parameters()).sum() * 0.0,
            algorithm._ungated_capture_residual(observation).sum() * 0.0,
            actor.raw_capture_probability(observation).sum() * 0.0,
            distillation.total,
            gate_supervision_has_rows=False,
            residual_supervision_has_rows=True,
        )
        assert all(
            parameter.grad is None
            for parameter in algorithm.capture_gate_parameters
        )
        assert all(
            parameter.grad is None
            for parameter in actor.mlp.teacher.parameters()
        )
        assert all(
            parameter.grad is None
            for parameter in algorithm.low_expert_builder.expert.parameters()
        )
        _, residual_grad_norm = algorithm._clip_training_gradients()
        assert residual_grad_norm > 0.0
        algorithm.optimizer.step()

    with torch.no_grad():
        final_loss = masked_low_expert_smooth_l1(
            algorithm._ungated_capture_residual(observation),
            cached.target,
            cached.mask,
            beta=algorithm.low_expert_smooth_l1_beta,
        ).total.item()
    assert losses[24] < 0.85 * losses[0]
    assert final_loss < 0.40 * losses[0]
    assert all(
        torch.equal(before, after)
        for before, after in zip(gate_before, algorithm.capture_gate_parameters)
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(teacher_before, actor.mlp.teacher.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            expert_before, algorithm.low_expert_builder.expert.parameters()
        )
    )
    residual_group = next(
        group
        for group in algorithm.optimizer.param_groups
        if group[OPTIMIZER_ROLE_KEY] == CAPTURE_RESIDUAL_OPTIMIZER_ROLE
    )
    assert residual_group["lr"] == pytest.approx(1.0e-4)
    assert algorithm.capture_residual_max_grad_norm == pytest.approx(0.5)
    assert algorithm.low_expert_residual_gradient_mode == "supervised_only"
    assert algorithm.capture_gate_gradient_mode == "stage_bce_only"

    # Seed Adam state for the gate as well as the already-trained residual.
    # This matters because grad=None must suppress momentum updates too, not
    # merely suppress a first-order gradient on an uninitialized group.
    valid_gate_bce = balanced_masked_stage_bce(
        stage_auxiliary_logits(actor, observation, None)[0],
        high_end_targets.label,
        high_end_targets.mask,
        high_end_targets.weight,
    ).total
    algorithm._backward_policy_and_low_expert(
        next(algorithm.critic.parameters()).sum() * 0.0,
        algorithm._ungated_capture_residual(observation).sum() * 0.0,
        valid_gate_bce,
        algorithm._ungated_capture_residual(observation).sum() * 0.0,
        gate_supervision_has_rows=True,
        residual_supervision_has_rows=False,
    )
    assert all(
        parameter.grad is not None for parameter in algorithm.capture_gate_parameters
    )
    algorithm._clip_training_gradients()
    algorithm.optimizer.step()

    # Empty HIGH/LOW supervision must clear deliberately opposing PPO gradients
    # for both private branches.  Parameters and every Adam tensor must remain
    # bitwise unchanged after optimizer.step().
    private_parameters = (
        algorithm.capture_gate_parameters + algorithm.capture_residual_parameters
    )
    private_before_empty = [
        parameter.detach().clone() for parameter in private_parameters
    ]
    optimizer_before_empty = {
        parameter: {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in algorithm.optimizer.state[parameter].items()
        }
        for parameter in private_parameters
    }
    assert all(
        {"step", "exp_avg", "exp_avg_sq"}.issubset(
            optimizer_before_empty[parameter]
        )
        for parameter in private_parameters
    )
    empty_primary = (
        algorithm._ungated_capture_residual(observation).square().mean()
        - actor.raw_capture_probability(observation).mean()
    )
    empty_anchor = algorithm._ungated_capture_residual(observation).sum() * 0.0
    empty_expert = masked_low_expert_smooth_l1(
        algorithm._ungated_capture_residual(observation),
        cached.target,
        torch.zeros_like(cached.mask),
        beta=algorithm.low_expert_smooth_l1_beta,
    ).total
    algorithm._backward_policy_and_low_expert(
        empty_primary,
        empty_anchor,
        actor.raw_capture_probability(observation).sum() * 0.0,
        empty_expert,
        gate_supervision_has_rows=False,
        residual_supervision_has_rows=False,
    )
    assert all(parameter.grad is None for parameter in private_parameters)
    algorithm.optimizer.step()
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            private_before_empty, private_parameters
        )
    )
    for parameter in private_parameters:
        assert optimizer_before_empty[parameter].keys() == (
            algorithm.optimizer.state[parameter].keys()
        )
        for key, before in optimizer_before_empty[parameter].items():
            after = algorithm.optimizer.state[parameter][key]
            if torch.is_tensor(before):
                assert torch.equal(before, after)
            else:
                assert before == after
    with torch.inference_mode():
        assert actor.mlp.capture_delta(observation["policy"]).abs().max().item() < 1.0e-3

    payload = algorithm.save()
    payload["high_friction_anchor"]["low_expert_distillation"][
        "residual_gradient_mode"
    ] = "joint"
    with pytest.raises(ValueError, match="gradient mode changed"):
        algorithm.load(
            payload,
            {
                "actor": False,
                "critic": False,
                "optimizer": True,
                "iteration": False,
                "rnd": False,
            },
            strict=True,
        )

    gate_payload = algorithm.save()
    gate_payload["high_friction_anchor"]["capture_gate_warmup"][
        "gradient_mode"
    ] = "joint"
    with pytest.raises(ValueError, match="capture gate gradient mode changed"):
        algorithm.load(
            gate_payload,
            {
                "actor": False,
                "critic": False,
                "optimizer": True,
                "iteration": False,
                "rnd": False,
            },
            strict=True,
        )


@pytest.mark.skipif(
    not FROZEN_SPEEDBOOST.is_file()
    or not LOW_EXPERT.is_file()
    or not FASTBASE_MODEL49.is_file(),
    reason="released model49/Teacher/expert artifact missing",
)
def test_gate_bce_only_branch_starts_from_model49_with_fresh_optimizer() -> None:
    """The fail-closed branch must reuse model49 tensors, never its joint Adam."""

    policy = torch.zeros(2, INPUT_DIM)
    policy[:, VALID_SLICE] = 1.0
    observation = TensorDict(
        {"policy": policy, "critic": torch.zeros(2, 570)}, batch_size=[2]
    )
    groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = FastBaseHallCaptureRslModel(
        observation,
        groups,
        "actor",
        OUTPUT_DIM,
        teacher_checkpoint=str(FROZEN_SPEEDBOOST),
        residual_limit=0.55,
        gate_logit_scale=2.75,
        gate_logit_bias=-3.2,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.08,
            "std_type": "scalar",
        },
    )
    private_env = SimpleNamespace(
        spatial_course_stage_buf=torch.full((2,), SPATIAL_HIGH_START),
        _hall_foot_packet_cache={"age": torch.zeros(2, 2)},
        episode_length_buf=torch.full((2,), 5, dtype=torch.long),
    )
    algorithm = AnchoredPPO(
        actor,
        MLPModel(
            observation, groups, "critic", 1, hidden_dims=[512, 256, 128]
        ),
        AnchoredRolloutStorage("rl", 2, 2, observation, [OUTPUT_DIM], "cpu"),
        env=private_env,
        anchor_teacher_checkpoint=str(FROZEN_SPEEDBOOST),
        device="cpu",
        schedule="fixed",
        learning_rate=5.0e-6,
        stage_aux_loss_coef=1.0,
        stage_aux_high_end_weight=4.0,
        capture_gate_warmup_updates=50,
        capture_gate_learning_rate=1.0e-5,
        capture_gate_gradient_mode="stage_bce_only",
        capture_residual_learning_rate=1.0e-4,
        capture_residual_max_grad_norm=0.5,
        low_expert_checkpoint=str(LOW_EXPERT),
        low_expert_expected_sha256=LOW_EXPERT_SHA256,
        low_expert_distillation_loss_coef=1.0,
        low_expert_target_cap=0.20,
        low_expert_smooth_l1_beta=0.05,
        low_expert_residual_gradient_mode="supervised_only",
    )
    payload = torch.load(FASTBASE_MODEL49, map_location="cpu", weights_only=True)
    assert algorithm.load(
        payload,
        {
            "actor": True,
            "critic": True,
            "optimizer": False,
            "iteration": True,
            "rnd": False,
        },
        strict=True,
    )
    assert algorithm.capture_gate_updates_completed == 50
    assert not algorithm.capture_gate_warmup_active
    assert capture_residual_has_zero_output(algorithm.capture_residual)
    assert not algorithm.optimizer.state
    groups_by_role = {
        group[OPTIMIZER_ROLE_KEY]: group
        for group in algorithm.optimizer.param_groups
    }
    assert groups_by_role[CAPTURE_GATE_OPTIMIZER_ROLE]["lr"] == pytest.approx(
        1.0e-5
    )
    assert groups_by_role[CAPTURE_RESIDUAL_OPTIMIZER_ROLE]["lr"] == pytest.approx(
        1.0e-4
    )
    assert algorithm.capture_gate_gradient_mode == "stage_bce_only"
    assert algorithm.low_expert_residual_gradient_mode == "supervised_only"
    assert algorithm.stage_aux_high_end_weight == pytest.approx(4.0)
    with pytest.raises(ValueError, match="capture gate gradient mode changed"):
        algorithm.load(
            payload,
            {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": False,
            },
            strict=True,
        )


def test_explicit_expert_runner_does_not_mutate_existing_fast_lr_config() -> None:
    cfg_path = (
        ROOT
        / "source"
        / "unitree_rl_lab"
        / "unitree_rl_lab"
        / "tasks"
        / "locomotion"
        / "agents"
        / "rsl_rl_ppo_cfg.py"
    )
    task_path = (
        ROOT
        / "source"
        / "unitree_rl_lab"
        / "unitree_rl_lab"
        / "tasks"
        / "locomotion"
        / "robots"
        / "g1"
        / "29dof"
        / "__init__.py"
    )
    source = cfg_path.read_text(encoding="utf-8")
    fast = source.split(
        "class FootTractionHallSpatialFastBaseCapturePPORunnerCfg", 1
    )[1].split(
        "class FootTractionHallSpatialCalibratedFastBaseCapturePPORunnerCfg", 1
    )[0]
    expert = source.split(
        "class FootTractionHallSpatialCalibratedFastBaseExpertDistillPPORunnerCfg",
        1,
    )[1].split(
        "class FootTractionHallSpatialCalibratedFastBaseExpertStrongDirectionPPORunnerCfg",
        1,
    )[0]
    strong = source.split(
        "class FootTractionHallSpatialCalibratedFastBaseExpertStrongDirectionPPORunnerCfg",
        1,
    )[1].split(
        "class FootTractionHallSpatialCalibratedFastBaseExpertGateBceOnlyPPORunnerCfg",
        1,
    )[0]
    gate_bce = source.split(
        "class FootTractionHallSpatialCalibratedFastBaseExpertGateBceOnlyPPORunnerCfg",
        1,
    )[1].split("class FootTractionSlopeStairsTeacherPPORunnerCfg", 1)[0]
    assert "capture_residual_learning_rate=5.0e-5" in fast
    assert "capture_residual_max_grad_norm=0.5" in fast
    assert "capture_residual_learning_rate=2.0e-5" in expert
    assert "capture_residual_max_grad_norm=0.1" in expert
    assert "low_expert_distillation_loss_coef=0.25" in expert
    assert "low_expert_smooth_l1_beta=0.05" in expert
    assert "low_expert_target_cap=0.20" in expert
    assert 'low_expert_residual_gradient_mode="joint"' in expert
    assert "capture_residual_learning_rate=1.0e-4" in strong
    assert "capture_residual_max_grad_norm=0.5" in strong
    assert "low_expert_distillation_loss_coef=1.0" in strong
    assert 'low_expert_residual_gradient_mode="supervised_only"' in strong
    assert "stage_aux_loss_coef=1.0" in strong
    assert "stage_aux_high_end_weight=4.0" in strong
    assert 'capture_gate_gradient_mode="stage_bce_only"' in gate_bce
    assert 'low_expert_residual_gradient_mode="supervised_only"' in gate_bce
    assert "max_iterations = 12" in gate_bce
    assert "save_interval = 4" in gate_bce
    assert "stage_aux_high_end_weight=4.0" in gate_bce
    assert "capture_gate_learning_rate=1.0e-5" in gate_bce
    assert "capture_residual_learning_rate=1.0e-4" in gate_bce
    assert (
        "SpatialFrictionMediumDenseFastBaseCaptureCalibratedExpertDistill"
        in task_path.read_text(encoding="utf-8")
    )
    assert (
        "SpatialFrictionMediumDenseFastBaseCaptureCalibratedExpertStrongDirection"
        in task_path.read_text(encoding="utf-8")
    )
    assert (
        "SpatialFrictionMediumDenseFastBaseCaptureCalibratedExpertGateBceOnly"
        in task_path.read_text(encoding="utf-8")
    )
