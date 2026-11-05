"""Detection node: camera image -> 2D detections.

Subscribes to ``/camera/image_raw`` and publishes
``/perception/detections_2d`` (vision_msgs/Detection2DArray). The
detector itself lives in the ROS-free core (nearest-palette color
segmentation), so this node is only message plumbing: image buffer ->
NumPy array -> core.detect_color -> vision_msgs.

The output detection format ({box, label, score}) is deliberately the
same one the Project 1 tracker consumed, so the two projects compose.
A learned detector (e.g. a YOLO node) could replace this node without
changing anything downstream: it would publish the same topic type.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from . import perception_core as core


class DetectionNode(Node):
    def __init__(self) -> None:
        super().__init__("detection")
        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._sub = self.create_subscription(
            Image, "/camera/image_raw", self._on_image, sensor_qos
        )
        self._pub = self.create_publisher(
            Detection2DArray, "/perception/detections_2d", 10
        )
        self._frames = 0
        self.get_logger().info("detection node ready (color segmentation)")

    def _on_image(self, msg: Image) -> None:
        if msg.encoding != "rgb8":
            self.get_logger().warn(
                f"expected rgb8 image, got {msg.encoding}", throttle_duration_sec=5.0
            )
            return
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )
        detections = core.detect_color(frame)

        out = Detection2DArray()
        out.header = msg.header
        for det in detections:
            x1, y1, x2, y2 = [float(v) for v in det["box"]]
            detection = Detection2D()
            detection.header = msg.header
            detection.bbox.center.position.x = (x1 + x2) / 2.0
            detection.bbox.center.position.y = (y1 + y2) / 2.0
            detection.bbox.size_x = x2 - x1
            detection.bbox.size_y = y2 - y1
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = det["label"]
            hypothesis.hypothesis.score = float(det["score"])
            detection.results.append(hypothesis)
            out.detections.append(detection)
        self._pub.publish(out)

        self._frames += 1
        if self._frames % 50 == 0:
            self.get_logger().info(
                f"processed {self._frames} frames, latest had "
                f"{len(detections)} detections"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
