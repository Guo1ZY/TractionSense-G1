#!/usr/bin/env python3
"""Launch the independent privileged torque Teacher PPO task."""

from __future__ import annotations

import os
from pathlib import Path
import sys


if __name__ == "__main__":
    train = Path(__file__).resolve().parents[1] / "rsl_rl/train.py"
    os.execv(sys.executable, [sys.executable, str(train), "--task", "Unitree-G1-29dof-Velocity-TorqueTractionTeacher", *sys.argv[1:]])

