"""Unit tests for the ROS-free perception core.

Run from the project root (no ROS installation needed):

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "mini_perception_stack"),
)
from mini_perception_stack import perception_core as core  # noqa: E402


class TestProjectionRoundTrip(unittest.TestCase):
    """Project a world object, then localize it back: the estimate must
    land near the true map position for a clean, centred box."""

    def _roundtrip(self, pose, label, ox, oy):
        frame, visible = core.render_frame(pose)
        rec = next(
            (v for v in visible if v["map_xy"] == (ox, oy) and v["label"] == label),
            None,
        )
        self.assertIsNotNone(rec, f"{label} at {(ox, oy)} not visible from {pose}")
        est = core.estimate_map_position(np.array(rec["box"]), label, pose)
        return float(np.hypot(est[0] - ox, est[1] - oy))

    def test_roundtrip_at_several_poses(self):
        # Renderer ground-truth boxes invert exactly through the
        # localizer (same pinhole model both ways), so tolerance is
        # tight. Detection-box error is measured by the offline eval.
        cases = [
            (core.robot_pose(0.0), "pedestrian", 12.0, 7.0),
            (core.robot_pose(10.0), "cone", -2.0, 10.5),
            (core.robot_pose(50.0), "crate", -7.0, -10.5),
            (core.robot_pose(60.0), "cone", 3.0, -10.5),
        ]
        for pose, label, ox, oy in cases:
            err = self._roundtrip(pose, label, ox, oy)
            self.assertLess(err, 0.05, f"{label} round-trip error {err:.2f} m")


class TestDetection(unittest.TestCase):
    def test_detector_finds_rendered_objects(self):
        pose = core.robot_pose(10.0)
        frame, visible = core.render_frame(pose)
        detections = core.detect_color(frame)
        self.assertGreaterEqual(len(detections), 2)
        self.assertLessEqual(abs(len(detections) - len(visible)), 2)

    def test_blank_frame_has_no_detections(self):
        blank = np.full((core.IMG_H, core.IMG_W, 3), 40, dtype=np.uint8)
        self.assertEqual(core.detect_color(blank), [])


class TestTransforms(unittest.TestCase):
    def test_world_optical_base_map_inverse(self):
        pose = core.robot_pose(33.0)
        point_map = np.array([6.0, 3.0])
        optical = core.world_to_optical(point_map, 0.0, pose)
        back = core.base_to_map(core.optical_to_base(optical), pose)
        self.assertAlmostEqual(back[0], point_map[0], places=6)
        self.assertAlmostEqual(back[1], point_map[1], places=6)
        self.assertAlmostEqual(back[2], 0.0, places=6)

    def test_object_straight_ahead_is_centred(self):
        # Robot at t=0 sits at (13, 0) facing +y (north). A point at
        # (13, 8) is dead ahead: it must project near the image centre
        # column.
        pose = core.robot_pose(0.0)
        optical = core.world_to_optical(np.array([13.0, 8.0]), 0.0, pose)
        px = core.project(optical)
        self.assertIsNotNone(px)
        self.assertAlmostEqual(px[0], core.CX, delta=40.0)


if __name__ == "__main__":
    unittest.main()
