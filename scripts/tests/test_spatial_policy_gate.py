"""CPU-only contract tests for the same-seed Hall policy selection gate."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "scripts" / "traction" / "eval_spatial_policy_gate.py"


def test_gate_is_independent_orchestrator_with_true_hardened_default():
    source = GATE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "subprocess" in imports
    assert "isaaclab" not in source.lower().split("def main", 1)[0]
    for token in (
        '"--candidate"',
        '"--seed"',
        '"--nominal_hall"',
        'command.append("--hardened_hall")',
        '"effective_config_valid"',
        '"zero_fall"',
        '"deceleration_by_0_5s"',
        '"high_recovery_time"',
        '"hall_health_by_seed"',
    ):
        assert token in source


def test_gate_rejects_legacy_rollouts_without_causal_metrics():
    source = GATE.read_text(encoding="utf-8")
    assert 'rollout.get("first_episode_only", False)' in source
    assert '"first-episode-causal-response-v1"' in source
    assert '"requested_hardened mismatch"' in source
    assert '"hardened mechanical randomization did not vary"' in source
