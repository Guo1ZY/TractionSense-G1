#!/usr/bin/env python3
"""Log per-foot Hall F0M1 asymmetry for MuJoCo walk diagnostics.

Reads the F0M1 bridge file written by unitree_mujoco (G1_MUJOCO_MAGNETIC_BRIDGE=1),
computes per-foot RMS/mean of the 15x3 magnetic array, and appends one JSON
line per aggregation window to a log file.  g1_ctrl is not touched.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import time

MAGIC_F0M1 = 0x46304D31
HEADER = struct.Struct("<IIQffffff")
VALUES = 2 * 15 * 3
PACKET = struct.Struct("<IIQffffff" + "f" * VALUES)


def read_packet(path: str):
    try:
        with open(path, "rb") as f:
            buf = f.read(PACKET.size)
    except OSError:
        return None
    if len(buf) != PACKET.size:
        return None
    fields = PACKET.unpack(buf)
    if fields[0] != MAGIC_F0M1:
        return None
    magic, seq, stamp_ns = fields[0], fields[1], fields[2]
    valid = fields[3:5]
    age = fields[5:7]
    period = fields[7:9]
    mag = fields[9:]
    left = mag[:45]
    right = mag[45:90]
    return {
        "stamp_ns": stamp_ns,
        "seq": seq,
        "valid": list(valid),
        "age_s": list(age),
        "period_s": list(period),
        "left": left,
        "right": right,
    }


def rms(vals) -> float:
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/tmp/g1_foot_rl_obs.bin")
    ap.add_argument("--out", default="/tmp/g1_hall_asym.jsonl")
    ap.add_argument("--window-s", type=float, default=1.0)
    ap.add_argument("--hz", type=float, default=100.0)
    args = ap.parse_args()

    period_s = 1.0 / args.hz
    window = []
    last_flush = time.monotonic()
    last_seq = -1
    with open(args.out, "a") as out:
        while True:
            pkt = read_packet(args.path)
            if pkt is not None and pkt["seq"] != last_seq:
                last_seq = pkt["seq"]
                lr = rms(pkt["left"])
                rr = rms(pkt["right"])
                window.append((pkt["stamp_ns"], lr, rr))
            now = time.monotonic()
            if now - last_flush >= args.window_s and window:
                lrs = [w[1] for w in window]
                rrs = [w[2] for w in window]
                l_mean = sum(lrs) / len(lrs)
                r_mean = sum(rrs) / len(rrs)
                l_rms = rms(lrs)
                r_rms = rms(rrs)
                diff = [a - b for a, b in zip(lrs, rrs)]
                record = {
                    "t0_ns": window[0][0],
                    "t1_ns": window[-1][0],
                    "n": len(window),
                    "left_mean": l_mean,
                    "right_mean": r_mean,
                    "left_rms": l_rms,
                    "right_rms": r_rms,
                    "diff_mean": sum(diff) / len(diff),
                    "diff_rms": rms(diff),
                    "asym_mean": (l_mean - r_mean) / (0.5 * (l_mean + r_mean) + 1e-6),
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                print(json.dumps(record))
                window = []
                last_flush = now
            time.sleep(period_s)


if __name__ == "__main__":
    main()
