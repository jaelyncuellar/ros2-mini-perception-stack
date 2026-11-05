"""Object localizer node: 2D detections + TF -> 3D objects in map frame.

For each detection on ``/perception/detections_2d``:

1. Reconstruct the 3D position in the camera optical frame with the
   core's monocular model (depth from the known real height of the
   detected class, Z = fy * H / box_height_px; back-project the
   ground-contact ray).
2. Look up the ``map <- camera_optical_frame`` transform in TF for the
   detection's timestamp and apply it (rotation by quaternion +
   translation, implemented directly so the node only needs tf2_ros for
   the buffer; production code would typically use tf2_geometry_msgs).
3. Publish the result two ways: ``/perception/detections_3d``
   (vision_msgs/Detection3DArray) for other software, and
   ``/perception/objects_3d`` (visualization_msgs/MarkerArray) for RViz.

Objects whose bounding box is clipped by the image border are published
like any other; the offline evaluation quantifies the extra error that
clipping causes (their apparent height is truncated, so their estimated
depth overshoots).
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2DArray, Detection3D, Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

from . import perception_core as core

MARKER_COLORS = {
    "cone": (1.0, 0.45, 0.1),
    "crate": (0.2, 0.4, 1.0),
    "pedestrian": (1.0, 0.8, 0.15),
}


def rotate_by_quaternion(point: np.ndarray, q) -> np.ndarray:
    """Rotate a vector by a geometry_msgs Quaternion (xyzw)."""
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    # v' = v + 2 * q_vec x (q_vec x v + qw * v)
    q_vec = np.array([qx, qy, qz])
    uv = np.cross(q_vec, point)
    uuv = np.cross(q_vec, uv)
    return point + 2.0 * (uv * qw + uuv)


class ObjectLocalizerNode(Node):
    def __init__(self) -> None:
        super().__init__("object_localizer")
        self._fx = core.FX
        self._fy = core.FY
        self._cx = core.CX
        self._cy = core.CY
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._info_sub = self.create_subscription(
            CameraInfo, "/camera/camera_info", self._on_info, 10
        )
        self._det_sub = self.create_subscription(
            Detection2DArray, "/perception/detections_2d", self._on_detections, 10
        )
        self._pub_3d = self.create_publisher(
            Detection3DArray, "/perception/detections_3d", 10
        )
        self._pub_markers = self.create_publisher(
            MarkerArray, "/perception/objects_3d", 10
        )
        self.get_logger().info("object localizer ready (monocular + TF)")

    def _on_info(self, msg: CameraInfo) -> None:
        if len(msg.k) == 9 and msg.k[0] > 0:
            self._fx = float(msg.k[0])
            self._fy = float(msg.k[4])
            self._cx = float(msg.k[2])
            self._cy = float(msg.k[5])

    def _on_detections(self, msg: Detection2DArray) -> None:
        if not msg.detections:
            return
        stamp = Time.from_msg(msg.header.stamp)
        try:
            transform = self._tf_buffer.lookup_transform(
                "map",
                "camera_optical_frame",
                stamp,
                timeout=Duration(seconds=0.1),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"TF lookup failed: {exc}", throttle_duration_sec=5.0
            )
            return
        tr = transform.transform.translation
        rot = transform.transform.rotation
        translation = np.array([tr.x, tr.y, tr.z])

        out = Detection3DArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = "map"
        markers = MarkerArray()

        for i, det in enumerate(msg.detections):
            if not det.results:
                continue
            label = det.results[0].hypothesis.class_id
            if label not in core.CLASSES:
                continue
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            box = np.array(
                [
                    cx - det.bbox.size_x / 2.0,
                    cy - det.bbox.size_y / 2.0,
                    cx + det.bbox.size_x / 2.0,
                    cy + det.bbox.size_y / 2.0,
                ]
            )
            point_optical = core.localize_box(
                box, label, fx=self._fx, fy=self._fy, cx=self._cx, cy=self._cy
            )
            point_map = rotate_by_quaternion(point_optical, rot) + translation

            det3d = Detection3D()
            det3d.header = out.header
            det3d.bbox.center.position.x = float(point_map[0])
            det3d.bbox.center.position.y = float(point_map[1])
            det3d.bbox.center.position.z = 0.0
            det3d.bbox.center.orientation.w = 1.0
            det3d.bbox.size.x = 0.4
            det3d.bbox.size.y = 0.4
            det3d.bbox.size.z = core.CLASSES[label][1]
            det3d.results.append(det.results[0])
            out.detections.append(det3d)

            marker = Marker()
            marker.header = out.header
            marker.ns = "perception"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(point_map[0])
            marker.pose.position.y = float(point_map[1])
            marker.pose.position.z = 0.25
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.35
            marker.scale.y = 0.35
            marker.scale.z = 0.35
            r, g, b = MARKER_COLORS.get(label, (1.0, 1.0, 1.0))
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 0.95
            marker.lifetime = Duration(seconds=0.4).to_msg()
            markers.markers.append(marker)

        if out.detections:
            self._pub_3d.publish(out)
            self._pub_markers.publish(markers)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
