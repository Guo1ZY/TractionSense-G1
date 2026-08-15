from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from unitree_rl_lab.traction.anchored_ppo import (
    CAPTURE_GATE_OPTIMIZER_ROLE,
    CAPTURE_RESIDUAL_OPTIMIZER_ROLE,
    OPTIMIZER_ROLE_KEY,
    PPO_OPTIMIZER_ROLE,
    AnchoredPPO,
    AnchoredRolloutStorage,
    FrozenTeacherAnchorTargetBuilder,
    StageAuxiliaryHead,
    actor_shared_latent_dim,
    actor_shared_trunk_latent,
    balanced_masked_stage_bce,
    bounded_teacher_targets,
    capture_residual_has_zero_output,
    high_friction_anchor_mask,
    masked_anchor_mse,
    stage_auxiliary_logits,
    stage_auxiliary_targets,
    validate_actor_observation_contract,
)
from unitree_rl_lab.traction.fastbase_capture_residual import (
    FastBaseHallCaptureRslModel,
)
from unitree_rl_lab.traction.frozen_speedboost_teacher import (
    INPUT_DIM,
    OUTPUT_DIM,
    TRAILING_SLICE,
)


class RecordingTeacher(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.calls = 0
        self.last_input: torch.Tensor | None = None

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.last_input = observation.detach().clone()
        # Large values prove both the action clamp and target-delta cap run.
        return torch.full(
            (observation.shape[0], OUTPUT_DIM),
            12.0,
            device=observation.device,
            dtype=observation.dtype,
        )


REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_TEACHER = (
    REPO_ROOT / "artifacts" / "hall_speed_demo" / "speedboost112_frozen_teacher.pt"
)


def test_high_start_and_high_end_anchor_but_low_never_does() -> None:
    stage = torch.tensor([0, 1, 2, 3, -1])
    mask = high_friction_anchor_mask(stage)
    assert mask[:, 0].tolist() == [True, False, True, False, False]

    finite = torch.tensor([True, True, False, True, True])
    mask = high_friction_anchor_mask(stage, finite_mask=finite)
    assert mask[:, 0].tolist() == [True, False, False, False, False]


def test_stage_auxiliary_labels_low_and_both_high_regions_without_truth_leakage() -> None:
    targets = stage_auxiliary_targets(
        torch.tensor([0, 1, 2, 3, -1, 1]),
        torch.tensor([8, 8, 8, 8, 8, 0]),
        reset_mask_steps=1,
        high_end_weight=4.0,
    )
    # HIGH_END must be high (zero), not confused with LOW just because it comes
    # after the low patch.
    assert targets.label[:, 0].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert targets.mask[:, 0].tolist() == [True, True, True, False, False, False]
    assert targets.weight[:, 0].tolist() == [1.0, 1.0, 4.0, 1.0, 1.0, 1.0]


def test_balanced_stage_loss_reports_high_and_low_and_masks_unknown_rows() -> None:
    logits = torch.tensor([[-2.0], [2.0], [2.0], [-2.0]], requires_grad=True)
    labels = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    mask = torch.tensor([[True], [True], [False], [False]])
    result = balanced_masked_stage_bce(logits, labels, mask)
    assert result.total.item() > 0.0
    assert result.high.item() > 0.0
    assert result.low.item() > 0.0
    assert result.accuracy.item() == pytest.approx(1.0)
    assert result.valid_fraction.item() == pytest.approx(0.5)
    assert result.low_fraction.item() == pytest.approx(0.5)
    result.total.backward()
    assert logits.grad is not None
    assert logits.grad[0].abs().item() > 0.0
    assert logits.grad[1].abs().item() > 0.0
    assert logits.grad[2].item() == 0.0
    assert logits.grad[3].item() == 0.0


def test_high_end_weight_prevents_return_samples_from_being_diluted() -> None:
    # Both rows are HIGH targets.  The second represents HIGH_END and should
    # contribute four times as much gradient as the equally wrong HIGH_START
    # row, without adding any deployable observation.
    logits = torch.tensor([[1.0], [1.0]], requires_grad=True)
    result = balanced_masked_stage_bce(
        logits,
        torch.zeros(2, 1),
        torch.ones(2, 1, dtype=torch.bool),
        torch.tensor([[1.0], [4.0]]),
    )
    result.total.backward()
    assert logits.grad is not None
    assert logits.grad[1].item() == pytest.approx(4.0 * logits.grad[0].item())


def test_teacher_targets_reject_nonfinite_and_cap_each_joint_delta() -> None:
    teacher = torch.tensor([[8.0, -8.0, 0.3], [float("nan"), 1.0, 2.0]])
    student = torch.tensor([[0.1, -0.1, 0.0], [0.2, 0.3, 0.4]])
    target, finite = bounded_teacher_targets(
        teacher, student, action_clamp=3.0, delta_cap=0.25
    )
    assert finite.tolist() == [True, False]
    assert torch.isfinite(target).all()
    assert torch.max(torch.abs(target[0] - student[0])) <= 0.25 + 1.0e-7
    torch.testing.assert_close(target[1], student[1])
    assert torch.max(torch.abs(target)) <= 3.0


def test_masked_anchor_loss_has_zero_low_stage_gradient() -> None:
    actor_mean = torch.tensor(
        [[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]], requires_grad=True
    )
    target = torch.zeros_like(actor_mean)
    mask = high_friction_anchor_mask(torch.tensor([0, 1, 2]))
    loss = masked_anchor_mse(actor_mean, target, mask)
    loss.backward()
    assert loss.item() > 0.0
    torch.testing.assert_close(actor_mean.grad[1], torch.zeros(2))
    assert torch.count_nonzero(actor_mean.grad[0]).item() == 2
    assert torch.count_nonzero(actor_mean.grad[2]).item() == 2


def test_builder_freezes_teacher_calls_once_caches_and_does_not_mutate_actor_obs() -> None:
    teacher = RecordingTeacher()
    builder = FrozenTeacherAnchorTargetBuilder(
        teacher, action_clamp=3.0, delta_cap=0.20
    )
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())

    observation = torch.randn(3, INPUT_DIM) * 0.01
    observation[:, TRAILING_SLICE] = torch.tensor(
        [[0.6, -0.4], [0.2, 0.3], [-0.8, 0.9]]
    )
    unchanged = observation.clone()
    sensor_age = torch.tensor([[0.1, 0.9], [0.0, 0.2], [0.7, 0.4]])
    student_mean = torch.zeros(3, OUTPUT_DIM)
    cache = builder.build(
        observation,
        sensor_age_lr=sensor_age,
        stage=torch.tensor([0, 1, 2]),
        student_mean=student_mean,
    )

    assert teacher.calls == builder.inference_calls == 1
    assert builder.cache_writes == 1
    assert cache.target.shape == (3, OUTPUT_DIM)
    assert cache.mask[:, 0].tolist() == [True, False, True]
    assert torch.max(torch.abs(cache.target - student_mean)) <= 0.20 + 1.0e-7
    torch.testing.assert_close(observation, unchanged, atol=0.0, rtol=0.0)
    assert teacher.last_input is not None
    torch.testing.assert_close(teacher.last_input[:, TRAILING_SLICE], sensor_age)


def _transition(
    observation: TensorDict,
    marker: float,
    *,
    num_envs: int,
    action_dim: int,
) -> RolloutStorage.Transition:
    transition = RolloutStorage.Transition()
    transition.observations = observation
    transition.actions = torch.zeros(num_envs, action_dim)
    transition.rewards = torch.zeros(num_envs)
    transition.dones = torch.zeros(num_envs, dtype=torch.long)
    transition.values = torch.zeros(num_envs, 1)
    transition.actions_log_prob = torch.zeros(num_envs)
    transition.distribution_params = (
        torch.zeros(num_envs, action_dim),
        torch.ones(num_envs, action_dim),
    )
    transition.anchor_target = torch.full((num_envs, action_dim), marker)
    transition.anchor_mask = torch.full((num_envs, 1), marker > 0.0, dtype=torch.bool)
    transition.stage_aux_label = torch.full((num_envs, 1), marker)
    transition.stage_aux_mask = torch.ones((num_envs, 1), dtype=torch.bool)
    transition.stage_aux_weight = torch.full((num_envs, 1), 1.0 + marker)
    transition.low_expert_residual_target = torch.full(
        (num_envs, action_dim), -marker
    )
    transition.low_expert_residual_mask = torch.full(
        (num_envs, 1), marker > 0.0, dtype=torch.bool
    )
    return transition


def test_rollout_storage_keeps_teacher_cache_aligned_with_observation() -> None:
    num_envs, steps, action_dim = 2, 2, 3
    initial = TensorDict(
        {"policy": torch.zeros(num_envs, 4)}, batch_size=[num_envs]
    )
    storage = AnchoredRolloutStorage(
        "rl", num_envs, steps, initial, [action_dim], "cpu"
    )
    assert list(storage.observations.keys()) == ["policy"]
    for marker in (0.0, 1.0):
        observation = TensorDict(
            {"policy": torch.full((num_envs, 4), marker)},
            batch_size=[num_envs],
        )
        storage.add_transition(
            _transition(
                observation, marker, num_envs=num_envs, action_dim=action_dim
            )
        )
    batch = next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))
    assert list(batch.observations.keys()) == ["policy"]
    observation_marker = batch.observations["policy"][:, 0]
    target_marker = batch.anchor_targets[:, 0]
    torch.testing.assert_close(observation_marker, target_marker)
    assert torch.equal(batch.anchor_masks[:, 0], target_marker.bool())
    torch.testing.assert_close(batch.stage_aux_labels[:, 0], target_marker)
    assert batch.stage_aux_masks.all()
    torch.testing.assert_close(batch.stage_aux_weights[:, 0], 1.0 + target_marker)
    torch.testing.assert_close(
        batch.low_expert_residual_targets[:, 0], -target_marker
    )
    assert torch.equal(
        batch.low_expert_residual_masks[:, 0], target_marker.bool()
    )


def test_fallback_auxiliary_head_backpropagates_into_shared_actor_trunk_only() -> None:
    observation = TensorDict(
        {"policy": torch.randn(4, INPUT_DIM)}, batch_size=[4]
    )
    actor = MLPModel(
        observation,
        {"actor": ["policy"]},
        "actor",
        OUTPUT_DIM,
        hidden_dims=[32, 16],
        activation="elu",
    )
    assert actor_shared_latent_dim(actor) == 16
    head = StageAuxiliaryHead(16, hidden_dim=8)
    logits = head(actor_shared_trunk_latent(actor, observation))
    loss = balanced_masked_stage_bce(
        logits,
        torch.tensor([[0.0], [1.0], [0.0], [1.0]]),
        torch.ones(4, 1, dtype=torch.bool),
    ).total
    loss.backward()

    first_layer = actor.mlp[0]
    action_head = actor.mlp[-1]
    assert first_layer.weight.grad is not None
    assert first_layer.weight.grad.abs().sum().item() > 0.0
    # The auxiliary branch taps before the action head; it reshapes the shared
    # Hall representation without directly supervising joint actions.
    assert action_head.weight.grad is None
    assert head.net[0].weight.grad is not None


def test_deployable_actor_export_contract_stays_1864_to_29_without_aux_head() -> None:
    observation = TensorDict(
        {"policy": torch.randn(2, INPUT_DIM)}, batch_size=[2]
    )
    actor = MLPModel(
        observation,
        {"actor": ["policy"]},
        "actor",
        OUTPUT_DIM,
        hidden_dims=[32, 16],
        activation="elu",
    )
    auxiliary = StageAuxiliaryHead(actor_shared_latent_dim(actor), hidden_dim=8)
    assert not any("stage_aux" in key for key in actor.state_dict())
    assert len(auxiliary.state_dict()) > 0

    export = actor.as_onnx(verbose=False)
    assert export.input_size == INPUT_DIM
    output = export(torch.zeros(3, INPUT_DIM))
    assert output.shape == (3, OUTPUT_DIM)


@pytest.mark.skipif(not FROZEN_TEACHER.is_file(), reason="converted Teacher missing")
def test_fastbase_auxiliary_supervises_the_deployable_capture_gate_directly() -> None:
    observation = TensorDict(
        {"policy": torch.randn(4, INPUT_DIM) * 0.02}, batch_size=[4]
    )
    actor = FastBaseHallCaptureRslModel(
        observation,
        {"actor": ["policy"]},
        "actor",
        OUTPUT_DIM,
        teacher_checkpoint=str(FROZEN_TEACHER),
        residual_limit=0.55,
        gate_logit_scale=2.75,
        gate_logit_bias=-3.2,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.08,
            "std_type": "scalar",
        },
    )
    logits, source = stage_auxiliary_logits(actor, observation, fallback_head=None)
    assert source == "actor_raw_capture_gate"
    assert logits.shape == (4, 1)
    raw = actor.raw_capture_probability(observation).clamp(1.0e-6, 1.0 - 1.0e-6)
    calibrated = actor.capture_probability(observation).clamp(1.0e-6, 1.0 - 1.0e-6)
    torch.testing.assert_close(logits, torch.logit(raw))
    assert not torch.allclose(logits, torch.logit(calibrated))
    loss = balanced_masked_stage_bce(
        logits,
        torch.tensor([[0.0], [1.0], [0.0], [1.0]]),
        torch.ones(4, 1, dtype=torch.bool),
    ).total
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
        for parameter in actor.mlp.gate.parameters()
    )
    assert all(parameter.grad is None for parameter in actor.mlp.teacher.parameters())


@pytest.mark.skipif(not FROZEN_TEACHER.is_file(), reason="converted Teacher missing")
def test_gate_only_warmup_is_exact_zero_frozen_and_checkpoint_resumable() -> None:
    observation = TensorDict(
        {"policy": torch.randn(3, INPUT_DIM) * 0.02}, batch_size=[3]
    )

    def make_algorithm(warmup_updates: int = 2) -> AnchoredPPO:
        groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = FastBaseHallCaptureRslModel(
            observation,
            groups,
            "actor",
            OUTPUT_DIM,
            teacher_checkpoint=str(FROZEN_TEACHER),
            residual_limit=0.55,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.08,
                "std_type": "scalar",
            },
        )
        critic = MLPModel(
            observation, groups, "critic", 1, hidden_dims=[16, 8]
        )
        storage = AnchoredRolloutStorage(
            "rl", 3, 2, observation, [OUTPUT_DIM], "cpu"
        )
        return AnchoredPPO(
            actor,
            critic,
            storage,
            env=SimpleNamespace(),
            anchor_teacher_checkpoint=str(FROZEN_TEACHER),
            device="cpu",
            schedule="fixed",
            learning_rate=5.0e-6,
            stage_aux_loss_coef=1.0,
            stage_aux_high_end_weight=4.0,
            capture_gate_warmup_updates=warmup_updates,
            capture_gate_warmup_learning_rate=1.0e-4,
            capture_gate_learning_rate=1.0e-5,
            capture_gate_max_grad_norm=1.0,
            capture_residual_learning_rate=5.0e-5,
            capture_residual_max_grad_norm=0.5,
        )

    algorithm = make_algorithm()
    residual = algorithm.capture_residual
    assert residual is not None
    assert algorithm.capture_gate_warmup_active
    assert capture_residual_has_zero_output(residual)
    assert all(not parameter.requires_grad for parameter in residual.parameters())
    roles = [
        group[OPTIMIZER_ROLE_KEY] for group in algorithm.optimizer.param_groups
    ]
    assert roles == [
        PPO_OPTIMIZER_ROLE,
        CAPTURE_GATE_OPTIMIZER_ROLE,
        CAPTURE_RESIDUAL_OPTIMIZER_ROLE,
    ]
    gate_group = algorithm._capture_gate_group()
    assert gate_group is not None
    assert gate_group["lr"] == pytest.approx(1.0e-4)
    gate_ids = {id(parameter) for parameter in algorithm.capture_gate_parameters}
    assert {id(parameter) for parameter in gate_group["params"]} == gate_ids
    residual_group = algorithm._capture_residual_group()
    assert residual_group is not None
    assert residual_group["lr"] == 0.0
    residual_ids = {
        id(parameter) for parameter in algorithm.capture_residual_parameters
    }
    assert {id(parameter) for parameter in residual_group["params"]} == residual_ids
    assert all(
        id(parameter) not in gate_ids | residual_ids
        for group in algorithm.optimizer.param_groups
        if group[OPTIMIZER_ROLE_KEY] == PPO_OPTIMIZER_ROLE
        for parameter in group["params"]
    )
    with torch.inference_mode():
        torch.testing.assert_close(
            algorithm.actor.mlp.capture_delta(observation["policy"]),
            torch.zeros(3, OUTPUT_DIM),
            atol=0.0,
            rtol=0.0,
        )

    gate_before = [
        parameter.detach().clone()
        for parameter in algorithm.actor.mlp.gate.parameters()
    ]
    residual_before = {
        name: value.detach().clone() for name, value in residual.state_dict().items()
    }
    logits, _ = stage_auxiliary_logits(
        algorithm.actor, observation, fallback_head=None
    )
    gate_loss = balanced_masked_stage_bce(
        logits,
        torch.tensor([[0.0], [1.0], [0.0]]),
        torch.ones(3, 1, dtype=torch.bool),
        torch.tensor([[1.0], [1.0], [4.0]]),
    ).total
    algorithm.optimizer.zero_grad()
    gate_loss.backward()
    algorithm.optimizer.step()
    assert any(
        not torch.equal(before, after)
        for before, after in zip(gate_before, algorithm.actor.mlp.gate.parameters())
    )
    for name, value in residual.state_dict().items():
        torch.testing.assert_close(value, residual_before[name], atol=0.0, rtol=0.0)
    assert capture_residual_has_zero_output(residual)

    algorithm._advance_capture_gate_warmup()
    assert algorithm.capture_gate_updates_completed == 1
    assert algorithm.capture_gate_warmup_active
    payload = algorithm.save()
    assert payload["capture_gate_warmup"]["completed_updates"] == 1
    assert payload["capture_gate_warmup"]["current_learning_rate"] == pytest.approx(
        1.0e-4
    )
    assert payload["capture_gate_warmup"][
        "residual_current_learning_rate"
    ] == 0.0
    residual_audit = payload["high_friction_anchor"][
        "capture_residual_optimizer"
    ]
    assert residual_audit == {
        "warmup_learning_rate": 0.0,
        "released_learning_rate": 5.0e-5,
        "current_learning_rate": 0.0,
        "max_grad_norm": 0.5,
        "independent_optimizer_group": True,
        "adaptive_ppo_lr_excluded": True,
        "release_counter_shared_with_gate_warmup": True,
        "optimizer_role": CAPTURE_RESIDUAL_OPTIMIZER_ROLE,
        "frozen_by_actor_contract": False,
        "all_parameters_require_grad_false": True,
    }
    saved_roles = [
        group[OPTIMIZER_ROLE_KEY]
        for group in payload["optimizer_state_dict"]["param_groups"]
    ]
    assert saved_roles == roles

    restored = make_algorithm()
    assert restored.load(payload, None, strict=True)
    assert restored.capture_gate_updates_completed == 1
    assert restored.capture_gate_warmup_active
    assert restored.capture_gate_current_learning_rate == pytest.approx(1.0e-4)
    assert restored.capture_residual_current_learning_rate == 0.0
    assert [
        group[OPTIMIZER_ROLE_KEY] for group in restored.optimizer.param_groups
    ] == roles
    assert all(
        not parameter.requires_grad
        for parameter in restored.capture_residual.parameters()
    )

    restored._advance_capture_gate_warmup()
    assert restored.capture_gate_updates_completed == 2
    assert not restored.capture_gate_warmup_active
    assert restored.capture_gate_current_learning_rate == pytest.approx(1.0e-5)
    assert restored.capture_residual_current_learning_rate == pytest.approx(5.0e-5)
    assert all(
        parameter.requires_grad
        for parameter in restored.capture_residual.parameters()
    )

    # Gate and residual clips are independent from the tiny PPO trust-region
    # clip.  The helper reports pre-clip norms, while actual gradients obey
    # each private bound.
    restored.optimizer.zero_grad()
    for parameter in restored.capture_gate_parameters:
        parameter.grad = torch.full_like(parameter, 10.0)
    for parameter in restored.capture_residual_parameters:
        parameter.grad = torch.full_like(parameter, 10.0)
    gate_norm, residual_norm, stability_norm = restored._clip_training_gradients()
    assert gate_norm > restored.capture_gate_max_grad_norm
    assert residual_norm > restored.capture_residual_max_grad_norm
    assert stability_norm == 0.0
    clipped_gate_norm = torch.linalg.vector_norm(
        torch.cat(
            [parameter.grad.reshape(-1) for parameter in restored.capture_gate_parameters]
        )
    )
    clipped_residual_norm = torch.linalg.vector_norm(
        torch.cat(
            [
                parameter.grad.reshape(-1)
                for parameter in restored.capture_residual_parameters
            ]
        )
    )
    # Accumulating a few hundred thousand float32 entries introduces a small
    # norm-reduction roundoff after clip_grad_norm_'s scaling.
    assert clipped_gate_norm <= restored.capture_gate_max_grad_norm + 1.0e-3
    assert clipped_residual_norm <= restored.capture_residual_max_grad_norm + 1.0e-3

    # A KL-adaptive PPO LR change must update only the ordinary PPO group.
    restored.learning_rate = 2.0e-6
    restored._sync_optimizer_learning_rates()
    for group in restored.optimizer.param_groups:
        if group[OPTIMIZER_ROLE_KEY] == CAPTURE_GATE_OPTIMIZER_ROLE:
            assert group["lr"] == pytest.approx(1.0e-5)
        elif group[OPTIMIZER_ROLE_KEY] == CAPTURE_RESIDUAL_OPTIMIZER_ROLE:
            assert group["lr"] == pytest.approx(5.0e-5)
        else:
            assert group["lr"] == pytest.approx(2.0e-6)

    # Create real Adam moments in the released residual role, then prove the
    # role, counter, LR and state survive a strict checkpoint roundtrip.
    restored.optimizer.zero_grad()
    features = restored.actor.mlp.capture_features(observation["policy"]).detach()
    restored.capture_residual(features).sum().backward()
    restored._clip_training_gradients()
    restored.optimizer.step()
    released_payload = restored.save()
    released_groups = {
        group[OPTIMIZER_ROLE_KEY]: group
        for group in released_payload["optimizer_state_dict"]["param_groups"]
    }
    released_residual_ids = set(
        released_groups[CAPTURE_RESIDUAL_OPTIMIZER_ROLE]["params"]
    )
    assert released_residual_ids & set(
        released_payload["optimizer_state_dict"]["state"]
    )
    resumed = make_algorithm()
    assert resumed.load(released_payload, None, strict=True)
    assert resumed.capture_gate_updates_completed == 2
    assert resumed.capture_residual_current_learning_rate == pytest.approx(5.0e-5)
    assert [
        group[OPTIMIZER_ROLE_KEY] for group in resumed.optimizer.param_groups
    ] == roles
    resumed_payload = resumed.save()
    assert len(resumed_payload["optimizer_state_dict"]["state"]) == len(
        released_payload["optimizer_state_dict"]["state"]
    )

    # model49-era checkpoints had only PPO+gate roles, with residual tensors
    # interleaved in PPO.  Recreate that serialized layout and ensure identity-
    # aware migration produces the new zero-LR residual role during warm-up.
    legacy = copy.deepcopy(payload)
    live_to_serialized: dict[int, int] = {}
    serialized_groups = legacy["optimizer_state_dict"]["param_groups"]
    for live_group, serialized_group in zip(
        algorithm.optimizer.param_groups, serialized_groups
    ):
        for parameter, serialized_id in zip(
            live_group["params"], serialized_group["params"]
        ):
            live_to_serialized[id(parameter)] = serialized_id
    groups_by_role = {
        group[OPTIMIZER_ROLE_KEY]: group for group in serialized_groups
    }
    legacy_ppo = copy.deepcopy(groups_by_role[PPO_OPTIMIZER_ROLE])
    legacy_ppo["params"] = [
        live_to_serialized[id(parameter)]
        for parameter in algorithm._legacy_gate_only_ppo_parameters
    ]
    legacy_gate = copy.deepcopy(groups_by_role[CAPTURE_GATE_OPTIMIZER_ROLE])
    legacy["optimizer_state_dict"]["param_groups"] = [legacy_ppo, legacy_gate]
    retained_ids = set(legacy_ppo["params"]) | set(legacy_gate["params"])
    legacy["optimizer_state_dict"]["state"] = {
        key: value
        for key, value in legacy["optimizer_state_dict"]["state"].items()
        if key in retained_ids
    }
    migrated = make_algorithm()
    assert migrated.load(legacy, None, strict=True)
    assert [
        group[OPTIMIZER_ROLE_KEY] for group in migrated.optimizer.param_groups
    ] == roles
    assert migrated.capture_residual_current_learning_rate == 0.0

    # Evaluation intentionally loads actor tensors only.  Its disabled
    # training phase must not conflict with the checkpointed warm-up counter.
    evaluator = make_algorithm(warmup_updates=0)
    assert not evaluator.load(
        payload,
        {
            "actor": True,
            "critic": False,
            "optimizer": False,
            "iteration": False,
            "rnd": False,
        },
        strict=True,
    )
    assert not evaluator.capture_gate_warmup_active


@pytest.mark.skipif(not FROZEN_TEACHER.is_file(), reason="converted Teacher missing")
def test_fallback_head_optimizer_checkpoint_and_legacy_resume_are_compatible() -> None:
    observation = TensorDict(
        {"policy": torch.zeros(2, INPUT_DIM)}, batch_size=[2]
    )

    def make_algorithm() -> AnchoredPPO:
        groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = MLPModel(
            observation,
            groups,
            "actor",
            OUTPUT_DIM,
            hidden_dims=[16, 8],
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.08,
                "std_type": "scalar",
            },
        )
        critic = MLPModel(
            observation, groups, "critic", 1, hidden_dims=[16, 8]
        )
        storage = AnchoredRolloutStorage(
            "rl", 2, 2, observation, [OUTPUT_DIM], "cpu"
        )
        return AnchoredPPO(
            actor,
            critic,
            storage,
            env=SimpleNamespace(),
            anchor_teacher_checkpoint=str(FROZEN_TEACHER),
            device="cpu",
            schedule="fixed",
            learning_rate=1.0e-4,
        )

    algorithm = make_algorithm()
    payload = algorithm.save()
    assert len(payload["optimizer_state_dict"]["param_groups"]) == 2
    assert "stage_auxiliary_state_dict" in payload
    assert not any("stage_aux" in key for key in payload["actor_state_dict"])
    assert make_algorithm().load(payload, None, strict=True)

    legacy = copy.deepcopy(payload)
    legacy.pop("stage_auxiliary_state_dict")
    legacy["optimizer_state_dict"]["param_groups"].pop()
    restored_legacy = make_algorithm()
    assert restored_legacy.load(legacy, None, strict=True)
    assert len(restored_legacy.optimizer.param_groups) == 2


def test_actor_contract_rejects_any_privileged_truth_group_or_schema_drift() -> None:
    valid = SimpleNamespace(obs_groups=["policy"], obs_dim=INPUT_DIM)
    validate_actor_observation_contract(valid)

    with pytest.raises(ValueError, match="truth-leakage"):
        validate_actor_observation_contract(
            SimpleNamespace(obs_groups=["policy", "critic"], obs_dim=INPUT_DIM)
        )
    with pytest.raises(ValueError, match="dimension"):
        validate_actor_observation_contract(
            SimpleNamespace(obs_groups=["policy"], obs_dim=INPUT_DIM + 1)
        )
