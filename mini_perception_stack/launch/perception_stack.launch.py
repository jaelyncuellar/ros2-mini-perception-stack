"""Launch the mini perception stack.

    ros2 launch mini_perception_stack perception_stack.launch.py
    ros2 launch mini_perception_stack perception_stack.launch.py use_rviz:=true
    ros2 launch mini_perception_stack perception_stack.launch.py time_scale:=2.0

Nodes: camera_sim -> detection -> object_localizer. RViz is off by
default so the stack runs headless (CI, SSH, containers); enable it on
a desktop to get the camera view, TF tree, ground-truth cubes, and the
localized object spheres from config/perception.rviz.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("mini_perception_stack")
    rviz_config = os.path.join(package_share, "config", "perception.rviz")

    use_rviz = LaunchConfiguration("use_rviz")
    time_scale = LaunchConfiguration("time_scale")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_rviz",
                default_value="false",
                description="Start RViz with the bundled config (desktop only).",
            ),
            DeclareLaunchArgument(
                "time_scale",
                default_value="1.0",
                description="Simulation speed multiplier for the camera sim robot.",
            ),
            Node(
                package="mini_perception_stack",
                executable="camera_sim_node",
                name="camera_sim",
                output="screen",
                parameters=[{"time_scale": time_scale}],
            ),
            Node(
                package="mini_perception_stack",
                executable="detection_node",
                name="detection",
                output="screen",
            ),
            Node(
                package="mini_perception_stack",
                executable="object_localizer_node",
                name="object_localizer",
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config],
                condition=IfCondition(use_rviz),
            ),
        ]
    )
