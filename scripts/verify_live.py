"""Live verification: score the running ROS graph against ground truth.

Run while the stack is launched (or use --duration to let this script
observe for N seconds and then report). Subscribes to
``/perception/detections_3d`` and compares every published object
position with the nearest same-class object in the core's world model,
writing ``out/metrics_live.json``.

This is the live counterpart of run_offline_eval.py: the same scoring
gate (2 m), but the numbers come from messages that actually travelled
through the ROS graph — image topic, detection node, TF lookups and all.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "mini_perception_stack"))

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from vision_msgs.msg import Detection3DArray  # noqa: E402

from mini_perception_stack import perception_core as core  # noqa: E402

MATCH_GATE_M = 2.0


class LiveVerifier(Node):
    def __init__(self) -> None:
        super().__init__("live_verifier")
        self.errors: list[float] = []
        self.messages = 0
        self.objects = 0
        self.unmatched = 0
        self.per_class: dict[str, int] = {}
        self._sub = self.create_subscription(
            Detection3DArray, "/perception/detections_3d", self._on_msg, 10
        )

    def _on_msg(self, msg: Detection3DArray) -> None:
        self.messages += 1
        for det in msg.detections:
            if not det.results:
                continue
            label = det.results[0].hypothesis.class_id
            x = det.bbox.center.position.x
            y = det.bbox.center.position.y
            self.objects += 1
            self.per_class[label] = self.per_class.get(label, 0) + 1
            best = min(
                (
                    float(np.hypot(x - ox, y - oy))
                    for lbl, ox, oy in core.WORLD_OBJECTS
                    if lbl == label
                ),
                default=None,
            )
            if best is None or best > MATCH_GATE_M:
                self.unmatched += 1
            else:
                self.errors.append(best)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()

    rclpy.init()
    node = LiveVerifier()
    deadline = time.monotonic() + args.duration
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)

    errors = np.array(node.errors)
    metrics = {
        "duration_s": args.duration,
        "detection3d_messages": node.messages,
        "objects_received": node.objects,
        "objects_matched_within_gate": int(len(errors)),
        "objects_unmatched": node.unmatched,
        "per_class_received": node.per_class,
        "position_rmse_m": (
            float(np.sqrt(np.mean(errors**2))) if len(errors) else None
        ),
        "position_mean_error_m": float(np.mean(errors)) if len(errors) else None,
        "position_max_error_m": float(np.max(errors)) if len(errors) else None,
    }
    out_path = PROJECT_DIR / "out" / "metrics_live.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
