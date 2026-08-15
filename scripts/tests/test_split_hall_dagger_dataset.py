from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
TRACTION_SCRIPTS = ROOT / "scripts" / "traction"
if str(TRACTION_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TRACTION_SCRIPTS))

import split_hall_dagger_dataset as splitter


def test_motion_dataset_manifest_does_not_relabel_tail_as_packet_age(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "motion.npz"
    output = tmp_path / "split"
    observation = np.zeros((4, 1864), dtype=np.float32)
    observation[:, 1862] = np.asarray((0.1, 0.2, 0.3, 0.4))
    observation[:, 1863] = np.asarray((-0.4, -0.2, 0.2, 0.4))
    np.savez_compressed(
        source,
        obs=observation,
        teacher_obs=np.zeros((4, 641), dtype=np.float32),
        mu=np.asarray((0.2, 0.2, 0.8, 0.8), dtype=np.float32),
        cmd_vx=np.asarray((0.2, 0.2, 0.8, 0.8), dtype=np.float32),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split_hall_dagger_dataset.py",
            "--input",
            str(source),
            "--output-dir",
            str(output),
            "--trailing-feature-mode",
            "motion_feedback",
        ],
    )

    assert splitter.main() == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["trailing_feature_mode"] == "motion_feedback"
    assert manifest["trailing_feature_summary"]["body_vy_mean"] == pytest.approx(
        0.25
    )
    assert manifest["trailing_feature_summary"][
        "relative_heading_mean"
    ] == pytest.approx(0.0, abs=1.0e-7)
    assert "age_mean_lr" not in manifest
