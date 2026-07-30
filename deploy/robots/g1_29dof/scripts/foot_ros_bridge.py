#!/usr/bin/env python3
"""Zorn foot ROS2 → g1_ctrl foot observation bridge.

Subscribes to zorn topics (Isaac / docker):
  /g1/left_foot/frame   std_msgs/Float32MultiArray[35]
  /g1/right_foot/frame  std_msgs/Float32MultiArray[35]

Maps frame35 → RL terms (aligned with unitree_rl_lab mdp/foot_sensor.py):
  contact_L/R   soft sigmoid from normal force mag
  normal_L/R    frame[0] * FORCE_SCALE   (default 0.01)
  tangent_L/R   frame[1] * FORCE_SCALE

Writes binary packet for C++ g1_ctrl (see foot_bridge.h):
  default path: /tmp/g1_foot_rl_obs.bin
  override:     env G1_FOOT_BRIDGE_PATH  or  --out PATH

Also writes a JSON sidecar for debugging: <out>.json

Usage (host, ROS2 Jazzy):
  source /opt/ros/jazzy/setup.bash
  export ROS_DOMAIN_ID=0
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  python3 foot_ros_bridge.py

  # dry-run / print only
  python3 foot_ros_bridge.py --print-hz 2

Requires: rclpy, std_msgs (ROS2). Does NOT need Isaac / unitree python.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

# Binary layout must match foot_bridge.h
MAGIC = 0x46305431  # F0T1
PACKET_FMT = "<IIQffffff"  # magic, seq, stamp_ns, cL,cR, nL,nR, tL,tR
PACKET_SIZE = struct.calcsize(PACKET_FMT)
assert PACKET_SIZE == 40

LEFT_FRAME_TOPIC = "/g1/left_foot/frame"
RIGHT_FRAME_TOPIC = "/g1/right_foot/frame"

DEFAULT_FORCE_SCALE = 0.01
DEFAULT_CONTACT_THRESHOLD = 5.0  # Newtons on raw normal mag
DEFAULT_SOFT_SCALE = 2.0


def soft_contact(normal_mag: float, threshold: float, soft_scale: float) -> float:
    """Match training soft contact: sigmoid((|F|-thr)*scale)."""
    x = (float(normal_mag) - threshold) * soft_scale
    # stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def frame_to_terms(
    frame,
    force_scale: float,
    threshold: float,
    soft_scale: float,
) -> tuple[float, float, float]:
    """Return (contact, normal_scaled, tangent_scaled) from frame35."""
    if frame is None or len(frame) < 2:
        return 0.0, 0.0, 0.0
    fn = float(frame[0])
    ft = float(frame[1])
    c = soft_contact(fn, threshold, soft_scale)
    # clip like training ObsTerm clip
    n = max(0.0, min(5.0, fn * force_scale))
    t = max(0.0, min(5.0, ft * force_scale))
    c = max(0.0, min(1.0, c))
    return c, n, t


def pack_packet(
    seq: int,
    c_l: float,
    c_r: float,
    n_l: float,
    n_r: float,
    t_l: float,
    t_r: float,
) -> bytes:
    stamp_ns = time.time_ns()
    return struct.pack(
        PACKET_FMT,
        MAGIC,
        seq & 0xFFFFFFFF,
        stamp_ns & 0xFFFFFFFFFFFFFFFF,
        float(c_l),
        float(c_r),
        float(n_l),
        float(n_r),
        float(t_l),
        float(t_r),
    )


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


class FootBridgeNode:
    def __init__(
        self,
        out_path: Path,
        force_scale: float,
        threshold: float,
        soft_scale: float,
        print_hz: float,
        write_json: bool,
    ):
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Float32MultiArray

        self.rclpy = rclpy
        self.Float32MultiArray = Float32MultiArray
        self.out_path = out_path
        self.json_path = out_path.with_suffix(out_path.suffix + ".json")
        self.force_scale = force_scale
        self.threshold = threshold
        self.soft_scale = soft_scale
        self.print_period = 1.0 / print_hz if print_hz > 0 else 0.0
        self.write_json = write_json

        self.left_frame = None
        self.right_frame = None
        self.seq = 0
        self.last_print = 0.0
        self.last_left_t = 0.0
        self.last_right_t = 0.0

        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = Node("g1_foot_ros_rl_bridge")
        self.node.create_subscription(Float32MultiArray, LEFT_FRAME_TOPIC, self._on_left, 10)
        self.node.create_subscription(Float32MultiArray, RIGHT_FRAME_TOPIC, self._on_right, 10)
        # publish rate independent of topic (re-write latest)
        self.timer = self.node.create_timer(0.01, self._on_timer)  # 100 Hz

        self.node.get_logger().info(f"Bridge out: {self.out_path}")
        self.node.get_logger().info(f"Listening: {LEFT_FRAME_TOPIC}, {RIGHT_FRAME_TOPIC}")
        self.node.get_logger().info(
            f"scale={force_scale} thr={threshold} soft={soft_scale} packet={PACKET_SIZE}B"
        )

    def _on_left(self, msg):
        self.left_frame = list(msg.data)
        self.last_left_t = time.time()

    def _on_right(self, msg):
        self.right_frame = list(msg.data)
        self.last_right_t = time.time()

    def _on_timer(self):
        c_l, n_l, t_l = frame_to_terms(
            self.left_frame, self.force_scale, self.threshold, self.soft_scale
        )
        c_r, n_r, t_r = frame_to_terms(
            self.right_frame, self.force_scale, self.threshold, self.soft_scale
        )
        self.seq += 1
        pkt = pack_packet(self.seq, c_l, c_r, n_l, n_r, t_l, t_r)
        try:
            atomic_write_bytes(self.out_path, pkt)
        except OSError as e:
            self.node.get_logger().warn(f"write bin failed: {e}")
            return

        now = time.time()
        age_l = now - self.last_left_t if self.last_left_t else 1e9
        age_r = now - self.last_right_t if self.last_right_t else 1e9

        if self.write_json:
            atomic_write_json(
                self.json_path,
                {
                    "seq": self.seq,
                    "path": str(self.out_path),
                    "contact": [c_l, c_r],
                    "normal_scaled": [n_l, n_r],
                    "tangent_scaled": [t_l, t_r],
                    "raw_normal": [
                        float(self.left_frame[0]) if self.left_frame else 0.0,
                        float(self.right_frame[0]) if self.right_frame else 0.0,
                    ],
                    "raw_tangent": [
                        float(self.left_frame[1]) if self.left_frame else 0.0,
                        float(self.right_frame[1]) if self.right_frame else 0.0,
                    ],
                    "age_sec": {"left": age_l, "right": age_r},
                    "force_scale": self.force_scale,
                },
            )

        if self.print_period > 0 and (now - self.last_print) >= self.print_period:
            self.last_print = now
            print(
                f"[foot_bridge] seq={self.seq} "
                f"C=[{c_l:.2f},{c_r:.2f}] N=[{n_l:.3f},{n_r:.3f}] T=[{t_l:.3f},{t_r:.3f}] "
                f"ageL={age_l:.2f}s ageR={age_r:.2f}s → {self.out_path}",
                flush=True,
            )

    def spin(self):
        try:
            self.rclpy.spin(self.node)
        except KeyboardInterrupt:
            pass
        finally:
            self.node.destroy_node()
            if self.rclpy.ok():
                self.rclpy.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Zorn foot ROS2 → g1_ctrl foot obs bridge")
    parser.add_argument(
        "--out",
        type=str,
        default=os.environ.get("G1_FOOT_BRIDGE_PATH", "/tmp/g1_foot_rl_obs.bin"),
        help="Binary IPC path (default /tmp/g1_foot_rl_obs.bin)",
    )
    parser.add_argument("--force-scale", type=float, default=DEFAULT_FORCE_SCALE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONTACT_THRESHOLD)
    parser.add_argument("--soft-scale", type=float, default=DEFAULT_SOFT_SCALE)
    parser.add_argument("--print-hz", type=float, default=1.0, help="0 to disable console print")
    parser.add_argument("--no-json", action="store_true", help="Do not write .json sidecar")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="No ROS: write synthetic walking-like forces for g1_ctrl dry-run",
    )
    args = parser.parse_args(argv)

    out_path = Path(args.out)

    if args.demo:
        print(f"[foot_bridge] DEMO mode → {out_path}")
        seq = 0
        try:
            while True:
                t = time.time()
                # alternate L/R load
                phase = (math.sin(t * 2.0 * math.pi * 0.8) + 1.0) * 0.5
                fn_l = 200.0 * phase
                fn_r = 200.0 * (1.0 - phase)
                ft_l = 20.0 * phase
                ft_r = 20.0 * (1.0 - phase)
                c_l, n_l, t_l = frame_to_terms(
                    [fn_l, ft_l], args.force_scale, args.threshold, args.soft_scale
                )
                c_r, n_r, t_r = frame_to_terms(
                    [fn_r, ft_r], args.force_scale, args.threshold, args.soft_scale
                )
                seq += 1
                atomic_write_bytes(out_path, pack_packet(seq, c_l, c_r, n_l, n_r, t_l, t_r))
                if seq % 50 == 0:
                    print(f"  demo seq={seq} C=[{c_l:.2f},{c_r:.2f}] N=[{n_l:.2f},{n_r:.2f}]")
                time.sleep(0.02)
        except KeyboardInterrupt:
            print("demo stopped")
        return 0

    try:
        import rclpy  # noqa: F401
        from std_msgs.msg import Float32MultiArray  # noqa: F401
    except ImportError as e:
        print(
            "ERROR: rclpy/std_msgs not available. Source ROS2 Jazzy first:\n"
            "  source /opt/ros/jazzy/setup.bash\n"
            "  export ROS_DOMAIN_ID=0\n"
            "  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp\n"
            "Or use --demo for synthetic forces without ROS.\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        return 1

    node = FootBridgeNode(
        out_path=out_path,
        force_scale=args.force_scale,
        threshold=args.threshold,
        soft_scale=args.soft_scale,
        print_hz=args.print_hz,
        write_json=not args.no_json,
    )
    node.spin()
    return 0


if __name__ == "__main__":
    sys.exit(main())
