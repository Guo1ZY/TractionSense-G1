"""Causal filter history-isolation regression tests."""

from __future__ import annotations

import torch

from unitree_rl_lab.traction_torque.torque_filter import JointStateFilter


def test_first_frame_and_reset_do_not_leak_acceleration() -> None:
    filt = JointStateFilter(2, 29)
    qd = torch.full((2, 29), 5.0)
    qdd, _ = filt.update(qd, torch.zeros_like(qd))
    assert qdd.count_nonzero().item() == 0
    filt.update(qd + 1.0, torch.ones_like(qd))
    filt.reset(torch.tensor([1]))
    qdd, tau = filt.update(qd + 2.0, torch.ones_like(qd))
    assert qdd[1].count_nonzero().item() == 0
    assert torch.isfinite(qdd).all() and torch.isfinite(tau).all()
