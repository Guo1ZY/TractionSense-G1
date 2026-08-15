from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import importlib.util

import pytest
import torch

from unitree_rl_lab.traction.holosoma_baseline import (
    HOLOSOMA_ACTION_SCALE_RAD,
    HOLOSOMA_DEFAULT_JOINT_POSITION,
    HOLOSOMA_FASTSAC_SHA256,
    HOLOSOMA_G1_29DOF_JOINT_ORDER,
    HOLOSOMA_OBSERVATION_SLICES,
    HoloSomaFastSacIsaacPolicy,
    HoloSomaFastSacTorchModule,
)
from unitree_rl_lab.traction.schema import G1_29DOF_JOINT_ORDER


MODEL = Path(
    "/home/mosense/guo/third_party/holosoma/src/holosoma_inference/"
    "holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx"
)


@pytest.mark.skipif(not MODEL.is_file(), reason="HoloSoma artifact not cloned")
def test_frozen_torch_reconstruction_is_dynamic_batch_and_finite() -> None:
    module, metadata = HoloSomaFastSacTorchModule.from_onnx(MODEL)
    generator = torch.Generator().manual_seed(20260812)
    observation = torch.randn(7, 100, generator=generator)
    with torch.inference_mode():
        output = module(observation)
    assert output.shape == (7, 29)
    assert torch.isfinite(output).all()
    assert metadata["dof_names"]


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None or not MODEL.is_file(),
    reason="ONNX Runtime or HoloSoma artifact unavailable",
)
def test_frozen_torch_reconstruction_matches_onnxruntime() -> None:
    import numpy as np
    import onnxruntime as ort

    module, _ = HoloSomaFastSacTorchModule.from_onnx(MODEL)
    generator = torch.Generator().manual_seed(20260812)
    observation = torch.randn(13, 100, generator=generator)
    with torch.inference_mode():
        torch_output = module(observation).numpy()
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    reference = np.concatenate(
        [
            session.run(
                ["action"],
                {"actor_obs": observation[index : index + 1].numpy()},
            )[0]
            for index in range(observation.shape[0])
        ],
        axis=0,
    )
    np.testing.assert_allclose(torch_output, reference, rtol=2.0e-5, atol=2.0e-6)


def _mock_runtime(num_envs: int = 3):
    robot_names = tuple(G1_29DOF_JOINT_ORDER)
    holo_lookup = {
        name: value
        for name, value in zip(
            HOLOSOMA_G1_29DOF_JOINT_ORDER, HOLOSOMA_DEFAULT_JOINT_POSITION
        )
    }
    default_robot = torch.tensor(
        [holo_lookup[name] for name in robot_names], dtype=torch.float32
    ).view(1, -1).repeat(num_envs, 1)
    data = SimpleNamespace(
        joint_pos=default_robot.clone(),
        joint_vel=torch.zeros(num_envs, 29),
        root_ang_vel_b=torch.zeros(num_envs, 3),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(
            num_envs, 1
        ),
    )
    robot = SimpleNamespace(joint_names=list(robot_names), data=data)
    action_term = SimpleNamespace(
        _joint_names=list(robot_names),
        _scale=0.25,
        _offset=default_robot.clone(),
    )
    return robot, action_term


@pytest.mark.skipif(not MODEL.is_file(), reason="HoloSoma artifact not cloned")
def test_real_fastsac_observation_and_joint_target_mapping() -> None:
    robot, action_term = _mock_runtime()
    policy = HoloSomaFastSacIsaacPolicy(
        MODEL,
        robot=robot,
        action_term=action_term,
        command_x_m_s=0.8,
        policy_dt_s=0.02,
    )
    action = policy(None)
    assert action.shape == (3, 29)
    assert torch.isfinite(action).all()
    obs = policy.last_observation
    assert obs.shape == (3, 100)

    def term(name: str) -> torch.Tensor:
        start, stop = HOLOSOMA_OBSERVATION_SLICES[name]
        return obs[:, start:stop]

    assert torch.count_nonzero(term("actions")) == 0
    assert torch.count_nonzero(term("base_ang_vel")) == 0
    assert torch.equal(term("command_ang_vel"), torch.zeros(3, 1))
    assert torch.equal(
        term("command_lin_vel"), torch.tensor([[0.8, 0.0]]).repeat(3, 1)
    )
    assert torch.allclose(term("dof_pos"), torch.zeros(3, 29), atol=1.0e-7)
    assert torch.count_nonzero(term("dof_vel")) == 0
    assert torch.equal(
        term("projected_gravity"), torch.tensor([[0.0, 0.0, -1.0]]).repeat(3, 1)
    )

    # Prove the environment action resolves to exactly HoloSoma's target q,
    # despite the different Isaac-vs-Holo joint order and nominal offsets.
    q_target_action_order = action * 0.25 + action_term._offset
    holo_lookup = {
        name: index for index, name in enumerate(HOLOSOMA_G1_29DOF_JOINT_ORDER)
    }
    raw_holo_action_order = policy.last_policy_action.index_select(
        1,
        torch.tensor(
            [holo_lookup[name] for name in G1_29DOF_JOINT_ORDER],
            dtype=torch.long,
        ),
    )
    default_holo_action_order = torch.tensor(
        [
            HOLOSOMA_DEFAULT_JOINT_POSITION[holo_lookup[name]]
            for name in G1_29DOF_JOINT_ORDER
        ]
    ).view(1, -1)
    expected_target = (
        default_holo_action_order
        + HOLOSOMA_ACTION_SCALE_RAD * raw_holo_action_order
    )
    assert torch.allclose(q_target_action_order, expected_target, atol=1.0e-6)


@pytest.mark.skipif(not MODEL.is_file(), reason="HoloSoma artifact not cloned")
def test_phase_and_previous_action_reset_are_per_environment() -> None:
    robot, action_term = _mock_runtime(num_envs=2)
    policy = HoloSomaFastSacIsaacPolicy(
        MODEL,
        robot=robot,
        action_term=action_term,
        command_x_m_s=0.8,
        policy_dt_s=0.02,
    )
    policy(None)
    previous_env1 = policy.last_policy_action[1].clone()
    policy.reset(torch.tensor([True, False]))
    policy(None)
    actions_start, actions_stop = HOLOSOMA_OBSERVATION_SLICES["actions"]
    action_input = policy.last_observation[:, actions_start:actions_stop]
    assert torch.count_nonzero(action_input[0]) == 0
    assert torch.equal(action_input[1], previous_env1)
    expected_dt = torch.tensor(2.0 * torch.pi * 0.02)
    assert torch.allclose(policy.phase[0, 0], expected_dt, atol=1.0e-6)
    assert torch.allclose(policy.phase[1, 0], 2.0 * expected_dt, atol=1.0e-6)


@pytest.mark.skipif(not MODEL.is_file(), reason="HoloSoma artifact not cloned")
def test_manifest_is_proprio_only_and_artifact_is_pinned() -> None:
    robot, action_term = _mock_runtime(num_envs=1)
    policy = HoloSomaFastSacIsaacPolicy(
        MODEL,
        robot=robot,
        action_term=action_term,
        command_x_m_s=0.8,
        policy_dt_s=0.02,
    )
    manifest = policy.manifest()
    assert manifest["source_model_sha256"] == HOLOSOMA_FASTSAC_SHA256
    assert manifest["source_observation_dimension"] == 100
    assert manifest["source_action_dimension"] == 29
    assert manifest["uses_hall"] is False
    assert manifest["uses_force_contact_friction_mu_slip_or_stage"] is False


def test_bad_joint_abi_and_rate_fail_closed(tmp_path: Path) -> None:
    if not MODEL.is_file():
        pytest.skip("HoloSoma artifact not cloned")
    robot, action_term = _mock_runtime(num_envs=1)
    with pytest.raises(RuntimeError, match="requires 50 Hz"):
        HoloSomaFastSacIsaacPolicy(
            MODEL,
            robot=robot,
            action_term=action_term,
            command_x_m_s=0.8,
            policy_dt_s=0.01,
        )
    robot.joint_names[-1] = "wrong_joint"
    with pytest.raises(RuntimeError, match="joint ABI mismatch"):
        HoloSomaFastSacIsaacPolicy(
            MODEL,
            robot=robot,
            action_term=action_term,
            command_x_m_s=0.8,
            policy_dt_s=0.02,
        )
