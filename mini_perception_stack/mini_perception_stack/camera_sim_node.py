"""Synthetic camera + robot simulator node.

Stands in for a real robot and camera so the perception stack can run
anywhere, including CI and headless machines:

* publishes ``/camera/image_raw`` (sensor_msgs/Image, rgb8) rendered by
  the ROS-free core from a known 3D world,
* publishes ``/camera/camera_info`` with the matching pinhole intrinsics,
* drives the robot along its loop and publishes ``/odom`` plus the TF
  tree map -> odom -> base_link -> camera_link -> camera_optical_frame,
* publishes ``/ground_truth/markers`` (the true object positions) for
  RViz comparison and for the live verification script. The perception
  nodes never subscribe to ground truth.

On real hardware this node is the only one that gets replaced: a camera
driver publishes the same two topics and robot_localization / the robot
driver provides the same TF tree.
"""

from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from . import perception_core as core


def yaw_to_quaternion(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


class CameraSimNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_sim")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("time_scale", 1.0)
        rate = float(self.get_parameter("publish_rate_hz").value)
        self._time_scale = float(self.get_parameter("time_scale").value)
        self._start_wall = time.monotonic()

        self._image_pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self._info_pub = self.create_publisher(
            CameraInfo, "/camera/camera_info", 10
        )
        self._odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self._truth_pub = self.create_publisher(
            MarkerArray, "/ground_truth/markers", 10
        )
        self._tf_broadcaster = TransformBroadcaster(self)
        self._publish_static_tf()
        self._timer = self.create_timer(1.0 / rate, self._on_timer)
        self.get_logger().info(
            f"camera_sim publishing at {rate} Hz (time_scale={self._time_scale})"
        )

    def _sim_time(self) -> float:
        return (time.monotonic() - self._start_wall) * self._time_scale

    def _publish_static_tf(self) -> None:
        static = StaticTransformBroadcaster(self)
        transforms = []

        base_to_cam = TransformStamped()
        base_to_cam.header.frame_id = "base_link"
        base_to_cam.child_frame_id = "camera_link"
        base_to_cam.transform.translation.x = core.CAM_OFFSET_X_M
        base_to_cam.transform.translation.z = core.CAM_HEIGHT_M
        base_to_cam.transform.rotation.w = 1.0
        transforms.append(base_to_cam)

        # REP-103 optical frame: z forward, x right, y down.
        cam_to_optical = TransformStamped()
        cam_to_optical.header.frame_id = "camera_link"
        cam_to_optical.child_frame_id = "camera_optical_frame"
        cam_to_optical.transform.rotation = Quaternion(
            x=-0.5, y=0.5, z=-0.5, w=0.5
        )
        transforms.append(cam_to_optical)

        static.sendTransform(transforms)

    def _on_timer(self) -> None:
        stamp = self.get_clock().now().to_msg()
        t = self._sim_time()
        x, y, yaw = core.robot_pose(t)
        frame, _visible = core.render_frame((x, y, yaw))

        image = Image()
        image.header.stamp = stamp
        image.header.frame_id = "camera_optical_frame"
        image.height = core.IMG_H
        image.width = core.IMG_W
        image.encoding = "rgb8"
        image.is_bigendian = 0
        image.step = core.IMG_W * 3
        image.data = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
        self._image_pub.publish(image)

        info = CameraInfo()
        info.header = image.header
        info.height = core.IMG_H
        info.width = core.IMG_W
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [
            core.FX, 0.0, core.CX,
            0.0, core.FY, core.CY,
            0.0, 0.0, 1.0,
        ]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [
            core.FX, 0.0, core.CX, 0.0,
            0.0, core.FY, core.CY, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        self._info_pub.publish(info)

        # Odometry + dynamic TF. map and odom coincide in this sim (no
        # drift); base_link follows the scripted loop.
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation = yaw_to_quaternion(yaw)
        self._odom_pub.publish(odom)

        map_to_odom = TransformStamped()
        map_to_odom.header.stamp = stamp
        map_to_odom.header.frame_id = "map"
        map_to_odom.child_frame_id = "odom"
        map_to_odom.transform.rotation.w = 1.0

        odom_to_base = TransformStamped()
        odom_to_base.header.stamp = stamp
        odom_to_base.header.frame_id = "odom"
        odom_to_base.child_frame_id = "base_link"
        odom_to_base.transform.translation.x = x
        odom_to_base.transform.translation.y = y
        odom_to_base.transform.rotation = yaw_to_quaternion(yaw)
        self._tf_broadcaster.sendTransform([map_to_odom, odom_to_base])

        markers = MarkerArray()
        for idx, (label, ox, oy) in enumerate(core.WORLD_OBJECTS):
            color, height_m, width_m = core.CLASSES[label]
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = "map"
            marker.ns = "ground_truth"
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = ox
            marker.pose.position.y = oy
            marker.pose.position.z = height_m / 2.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = width_m
            marker.scale.y = width_m
            marker.scale.z = height_m
            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 0.55
            markers.markers.append(marker)
        self._truth_pub.publish(markers)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
