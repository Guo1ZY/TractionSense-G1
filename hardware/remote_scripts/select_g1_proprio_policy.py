#!/usr/bin/env python3
"""Validate and optionally select the final torque-only G1 policy slot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


DEPLOY_ROOT = Path("/home/mosense/guo/unitree_rl_lab/deploy/robots/g1_29dof")
CONFIG = DEPLOY_ROOT / "config/config.yaml"
BACKUP = DEPLOY_ROOT / "config/config.before_proprio_tau_harness.yaml"
FINAL_POLICY = "config/policy/velocity/traction_proprio_tau_dagger2"
OFFICIAL_ONNX = DEPLOY_ROOT / "config/policy/velocity/v0/exported/policy.onnx"
OFFICIAL_SHA256 = "610c27e463a8f666aa50a06346678c00b4df3859f10b54bcc1f817c28251406f"
FOOT_TERMS = (
    "foot_contact:",
    "foot_normal_force:",
    "foot_tangent_force:",
    "foot_friction_ratio:",
    "foot_load_ratio:",
    "foot_magnetic_array:",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def validate_slot() -> None:
    slot = DEPLOY_ROOT / FINAL_POLICY
    required = (
        slot / "exported/policy.onnx",
        slot / "exported/friction_estimator.onnx",
        slot / "params/deploy.yaml",
        slot / "install_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete torque-only slot: {missing}")
    manifest = json.loads(required[3].read_text(encoding="utf-8"))
    if manifest.get("input_dim") != 915:
        raise RuntimeError(f"unexpected policy input: {manifest.get('input_dim')}")
    if manifest.get("foot_sensor_required") or manifest.get("magnetic_sensor_required"):
        raise RuntimeError("selected slot is not pure proprioception")
    deploy = required[2].read_text(encoding="utf-8")
    if "joint_effort:" not in deploy or "history_length: 15" not in deploy:
        raise RuntimeError("deploy schema lacks the 15-frame joint_effort history")
    found_foot = [term for term in FOOT_TERMS if term in deploy]
    if found_foot:
        raise RuntimeError(f"unexpected external-sensor observations: {found_foot}")
    official_hash = sha256(OFFICIAL_ONNX)
    if official_hash != OFFICIAL_SHA256:
        raise RuntimeError(f"official v0 hash changed: {official_hash}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--apply", action="store_true")
    group.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    if args.restore:
        if not BACKUP.is_file():
            raise FileNotFoundError(f"backup does not exist: {BACKUP}")
        atomic_write(CONFIG, BACKUP.read_text(encoding="utf-8"))
        print(f"restored {CONFIG} from {BACKUP}")
        return 0

    validate_slot()
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
    print("schema : 915-D, 480-D stock history + 15x29 tau_est, no external sensor")
    print(f"v0 hash: {OFFICIAL_SHA256} (unchanged)")
    if not args.apply:
        print("DRY_RUN PASS; pass --apply only with harness and hardware E-stop ready")
        return 0
    if not BACKUP.exists():
        BACKUP.write_text(text, encoding="utf-8")
        print(f"backup : {BACKUP}")
    updated = text.replace(
        f"policy_dir: {current}", f"policy_dir: {FINAL_POLICY}", 1
    )
    atomic_write(CONFIG, updated)
    print(f"selected torque-only policy in {CONFIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
