from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "traction"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pretrain_rsl_hall_actor import select_deploy_targets


def test_pretraining_uses_baseline_target_when_either_hall_foot_is_invalid() -> None:
    teacher = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    baseline = np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    target, used_teacher = select_deploy_targets(
        teacher,
        baseline,
        np.asarray([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(target[0], teacher[0])
    np.testing.assert_allclose(target[1:], baseline[1:])
    np.testing.assert_array_equal(used_teacher, [True, False, False])
