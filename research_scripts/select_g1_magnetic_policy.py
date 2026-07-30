#!/usr/bin/env python3
"""Validate and optionally select the final magnetic G1 policy slot."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


REPO_ROOT = Path(
    os.environ.get("TRACTIONSENSE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
DEPLOY_ROOT = REPO_ROOT / "deploy/robots/g1_29dof"
CONFIG = DEPLOY_ROOT / "config/config.yaml"
BACKUP = DEPLOY_ROOT / "config/config.before_magnetic_harness.yaml"
FINAL_POLICY = "config/policy/velocity/traction_magnetic_speedboost112_guard"


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--apply", action="store_true")
    group.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    if args.restore:
        if not BACKUP.is_file():
            raise FileNotFoundError(f"backup does not exist: {BACKUP}")
        if args.restore:
            atomic_write(CONFIG, BACKUP.read_text(encoding="utf-8"))
            print(f"restored {CONFIG} from {BACKUP}")
        return 0

    onnx = DEPLOY_ROOT / FINAL_POLICY / "exported/policy.onnx"
    deploy = DEPLOY_ROOT / FINAL_POLICY / "params/deploy.yaml"
    if not onnx.is_file() or not deploy.is_file():
        raise FileNotFoundError(f"final policy slot is incomplete: {FINAL_POLICY}")
    text = CONFIG.read_text(encoding="utf-8")
    lines = [
        line for line in text.splitlines()
        if line.lstrip().startswith("policy_dir: config/policy/velocity/")
    ]
    if len(lines) != 1:
        raise RuntimeError(f"expected one velocity policy_dir, found {len(lines)}")
    current = lines[0].split("policy_dir:", 1)[1].strip()
    print(f"current: {current}")
    print(f"target : {FINAL_POLICY}")
    if not args.apply:
        print("dry run only; pass --apply after BLE calibration and harness setup")
        return 0
    if not BACKUP.exists():
        BACKUP.write_text(text, encoding="utf-8")
        print(f"backup : {BACKUP}")
    updated = text.replace(
        f"policy_dir: {current}",
        f"policy_dir: {FINAL_POLICY}",
        1,
    )
    atomic_write(CONFIG, updated)
    print(f"selected final magnetic policy in {CONFIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
