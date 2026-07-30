#!/usr/bin/env python3
"""Source-level safety invariants for the Isaac friction matrix evaluator."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "scripts/rsl_rl/eval_friction_matrix.py"


def test_warmup_falls_are_counted_before_measurement_continue() -> None:
    source = EVALUATOR.read_text()
    after_step = source.split("obs, rew, dones, extras = env.step(actions)", 1)[1]
    fall_count = after_step.index("falls_total +=")
    warmup_continue = after_step.index("if step < args_cli.warmup_steps:")

    assert fall_count < warmup_continue
    assert source.count("falls_total +=") == 1


def test_fall_diagnostics_identify_warmup_or_measurement_phase() -> None:
    source = EVALUATOR.read_text()

    assert '"phase",' in source
    assert 'phase = "warmup" if step < args_cli.warmup_steps else "measure"' in source
    assert '"phase": phase' in source
