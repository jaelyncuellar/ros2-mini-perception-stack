import os
from glob import glob

from setuptools import find_packages, setup

package_name = "mini_perception_stack"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.rviz"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jaelyn Cuellar",
    maintainer_email="jc@muse.local",
    description="Mini ROS 2 perception stack: camera -> detection -> TF localization -> RViz.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "camera_sim_node = mini_perception_stack.camera_sim_node:main",
            "detection_node = mini_perception_stack.detection_node:main",
            "object_localizer_node = mini_perception_stack.object_localizer_node:main",
        ],
    },
)
