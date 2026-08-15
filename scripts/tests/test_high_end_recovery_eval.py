from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/rsl_rl/eval_high_end_recovery_bank.py"


def test_validation_evaluator_is_first_episode_censored_and_hall_only() -> None:
    source = SCRIPT.read_text()
    assert 'default="validation_high_end_state_bank"' in source
    assert "load_high_end_state_bank(" in source
    assert 'entry_point_key="env_cfg_entry_point"' in source
    assert 'type(base).__name__ != "HighEndRecoveryRLEnv"' in source
    assert "_high_end_recovery_last_audit" in source
    assert "pre_active = active.clone()" in source
    assert "active &= ~done.bool()" in source
    assert "post-reset sample excluded" in source
    assert '"uses_force_contact_mu_slip_or_stage": False' in source


def test_validation_evaluator_loads_actor_only_and_reports_recovery_retention() -> None:
    source = SCRIPT.read_text()
    for text in (
        '"actor": True',
        '"critic": False',
        '"optimizer": False',
        '"iteration": False',
        '"near_failure_recovery"',
        '"nominal_retention"',
        '"recovered_and_held_fraction"',
        '"action_saturation_fraction"',
        '"heading_rms_rad"',
        '"body_vy_rms_m_s"',
    ):
        assert text in source
