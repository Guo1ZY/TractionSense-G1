from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from unitree_rl_lab.utils.partial_checkpoint import load_partial_into_runner


def _linear(value: float) -> nn.Linear:
    module = nn.Linear(3, 2)
    with torch.no_grad():
        module.weight.fill_(value)
        module.bias.fill_(value)
    return module


def test_critic_only_partial_warmstart_keeps_every_actor_tensor_fresh(tmp_path) -> None:
    source_actor = _linear(7.0)
    source_critic = _linear(11.0)
    checkpoint = tmp_path / "source.pt"
    torch.save(
        {
            "actor_state_dict": source_actor.state_dict(),
            "critic_state_dict": source_critic.state_dict(),
        },
        checkpoint,
    )

    actor = _linear(-3.0)
    critic = _linear(-5.0)
    actor_before = {
        name: value.detach().clone() for name, value in actor.state_dict().items()
    }
    runner = SimpleNamespace(alg=SimpleNamespace(actor=actor, critic=critic))
    stats = load_partial_into_runner(
        runner,
        str(checkpoint),
        load_actor=False,
        load_critic=True,
        verbose=False,
    )

    assert set(stats) == {"path", "critic"}
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value, actor_before[name], atol=0.0, rtol=0.0)
    for name, value in critic.state_dict().items():
        torch.testing.assert_close(
            value, source_critic.state_dict()[name], atol=0.0, rtol=0.0
        )


def test_partial_warmstart_rejects_loading_neither_branch(tmp_path) -> None:
    runner = SimpleNamespace(
        alg=SimpleNamespace(actor=_linear(0.0), critic=_linear(0.0))
    )
    with pytest.raises(ValueError, match="actor, critic, or both"):
        load_partial_into_runner(
            runner,
            str(tmp_path / "unused.pt"),
            load_actor=False,
            load_critic=False,
            verbose=False,
        )
