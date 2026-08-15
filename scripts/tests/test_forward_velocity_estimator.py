from __future__ import annotations

import numpy as np
import pytest
import torch

from unitree_rl_lab.traction.forward_velocity_estimator import (
    ForwardVelocityEstimator,
    NormalizedForwardVelocityEstimator,
    build_forward_velocity_estimator,
)


def test_normalized_estimator_has_batch_shape_and_bound() -> None:
    core = ForwardVelocityEstimator(8, (16, 8), output_clip=0.4)
    runtime = NormalizedForwardVelocityEstimator(
        core,
        mean=np.zeros(8, dtype=np.float32),
        scale=np.ones(8, dtype=np.float32),
    )
    output = runtime(torch.randn(5, 8) * 100.0)
    assert output.shape == (5,)
    assert torch.all(torch.abs(output) <= 0.4)


def test_feature_projection_matches_selected_input_columns() -> None:
    core = ForwardVelocityEstimator(3, (4,), output_clip=10.0)
    with torch.no_grad():
        for parameter in core.parameters():
            parameter.zero_()
        core.network[-1].bias.fill_(0.25)
    runtime = NormalizedForwardVelocityEstimator(
        core,
        mean=np.zeros(3, dtype=np.float32),
        scale=np.ones(3, dtype=np.float32),
        feature_indices=np.asarray([1, 4, 6], dtype=np.int64),
    )
    value = runtime(torch.randn(2, 8))
    assert torch.allclose(value, torch.full((2,), 0.25))


def test_checkpoint_builder_restores_exact_runtime() -> None:
    core = ForwardVelocityEstimator(4, (8,), output_clip=1.5)
    payload = {
        "input_dim": 4,
        "hidden_dims": (8,),
        "output_clip": 1.5,
        "model": core.state_dict(),
        "mean": np.zeros(4, dtype=np.float32),
        "scale": np.ones(4, dtype=np.float32),
        "feature_indices": np.arange(4, dtype=np.int64),
    }
    restored = build_forward_velocity_estimator(payload)
    sample = torch.randn(3, 4)
    assert torch.allclose(restored(sample), core(sample))


def test_bad_normalization_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        NormalizedForwardVelocityEstimator(
            ForwardVelocityEstimator(2, (4,)),
            mean=np.zeros(2, dtype=np.float32),
            scale=np.asarray([1.0, 0.0], dtype=np.float32),
        )
