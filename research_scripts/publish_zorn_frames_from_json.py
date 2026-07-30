#!/usr/bin/env python3
"""Publish zorn-compatible /g1/{left,right}_foot/frame from a real ContactView JSON snapshot.

Real2Sim data path: zorn-recorded frame JSON → ROS2 topics (same as live zorn) → foot_ros_bridge.

Usage (host, ROS2 Jazzy):
  source /opt/ros/jazzy/setup.bash
  export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  python3 publish_zorn_frames_from_json.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

DEFAULT_JSON = str(
    Path(
        os.environ.get(
            "ZORN_FOOT_SENSOR_ROOT",
            Path.home() / "docker/zorn/workspace/foot_sensor",
        )
    )
    / "latest_contactview_frame_live.json"
)


def build_frame35(side: dict) -> list[float]:
    """Match zorn build_frame35 layout (35 floats)."""
    def f(key, default=0.0):
        v = side.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    def vec3(key):
        v = side.get(key, [0.0, 0.0, 0.0])
        if not isinstance(v, (list, tuple)) or len(v) < 3:
            return [0.0, 0.0, 0.0]
        return [float(v[0]), float(v[1]), float(v[2])]

    sensor15 = side.get("sensor15") or side.get("sensor15_normal_force")
    if not isinstance(sensor15, (list, tuple)) or len(sensor15) != 15:
        # RBF-like flat distribution of Fn (placeholder if missing)
        fn = max(f("normal_force_mag"), 0.0)
        sensor15 = [fn / 15.0] * 15
    else:
        sensor15 = [float(x) for x in sensor15[:15]]

    frame = (
        [
            f("normal_force_mag"),
            f("tangent_force_mag"),
            f("total_force_mag"),
        ]
        + vec3("cop_local")
        + vec3("force_local_total")
        + vec3("normal_force_local")
        + vec3("tangent_force_local")
        + vec3("torque_local")
        + [f("contact_count"), f("friction_count")]
        + sensor15
    )
    if len(frame) != 35:
        raise RuntimeError(f"frame35 len={len(frame)}")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=Path(DEFAULT_JSON))
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--walk-mod", action="store_true", help="modulate Fn as alternate stance")
    args = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray

    data = json.loads(args.json.read_text())
    left0 = build_frame35(data["left"])
    right0 = build_frame35(data["right"])
    print(f"[zorn-replay] loaded {args.json}")
    print(f"  L Fn={left0[0]:.1f} Ft={left0[1]:.1f}  R Fn={right0[0]:.1f} Ft={right0[1]:.1f}")

    rclpy.init()
    node = Node("zorn_frame_replay")
    pub_l = node.create_publisher(Float32MultiArray, "/g1/left_foot/frame", 10)
    pub_r = node.create_publisher(Float32MultiArray, "/g1/right_foot/frame", 10)
    pub_ls = node.create_publisher(Float32MultiArray, "/g1/left_foot/sensor15", 10)
    pub_rs = node.create_publisher(Float32MultiArray, "/g1/right_foot/sensor15", 10)

    dt = 1.0 / max(args.hz, 1.0)
    t0 = time.time()
    n = 0
    try:
        while rclpy.ok():
            left = list(left0)
            right = list(right0)
            if args.walk_mod:
                phase = (time.time() - t0) * 1.2
                # alternate load L/R like gait
                w_l = 0.55 + 0.45 * math.sin(phase)
                w_r = 0.55 + 0.45 * math.sin(phase + math.pi)
                left[0] = left0[0] * max(w_l, 0.05)
                right[0] = right0[0] * max(w_r, 0.05)
                left[1] = left0[1] * max(w_l, 0.05)
                right[1] = right0[1] * max(w_r, 0.05)
                left[2] = math.hypot(left[0], left[1])
                right[2] = math.hypot(right[0], right[1])
            ml = Float32MultiArray()
            ml.data = [float(x) for x in left]
            mr = Float32MultiArray()
            mr.data = [float(x) for x in right]
            sl = Float32MultiArray()
            sl.data = [float(x) for x in left[20:35]]
            sr = Float32MultiArray()
            sr.data = [float(x) for x in right[20:35]]
            pub_l.publish(ml)
            pub_r.publish(mr)
            pub_ls.publish(sl)
            pub_rs.publish(sr)
            n += 1
            if n % int(max(args.hz, 1)) == 0:
                node.get_logger().info(
                    f"pub L Fn={left[0]:.0f} R Fn={right[0]:.0f}  n={n}"
                )
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    print("[zorn-replay] stopped")


if __name__ == "__main__":
    main()
