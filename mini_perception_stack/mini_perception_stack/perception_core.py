"""ROS-free perception core for the mini perception stack.

Everything in this module is plain NumPy/Pillow so it can be unit-tested
and evaluated offline, without a ROS installation. The ROS 2 nodes in
this package are thin wrappers around these functions:

* ``render_frame``     - synthetic camera: projects the known 3D world
  into a pinhole camera image as the robot drives its loop.
* ``detect_color``     - the detector: nearest-palette color segmentation
  producing {box, label, score} dicts (the same detection format the
  Project 1 tracker consumed).
* ``localize_box``     - monocular 3D estimate: depth from the known real
  height of the detected class, Z = fy * H_real / height_px, then the
  pixel ray back-projection into the camera (optical) frame.
* ``base_to_map``      - 2D rigid transform from the robot base frame to
  the map frame (what tf2 does at runtime in the live ROS graph).

Frames: map is a fixed east/north plane. base_link is x-forward,
y-left, z-up. The camera optical frame is z-forward, x-right, y-down,
mounted CAM_HEIGHT_M above the ground at CAM_OFFSET_X_M ahead of the
base origin, looking along the robot's forward axis.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# --- camera -------------------------------------------------------------
IMG_W, IMG_H = 640, 480
FX = FY = 520.0
CX, CY = IMG_W / 2.0, IMG_H / 2.0
CAM_HEIGHT_M = 1.2
CAM_OFFSET_X_M = 0.25
MAX_RANGE_M = 25.0
MIN_DEPTH_M = 0.8

# label -> (RGB in the rendered world, real height m, real width m)
CLASSES = {
    "cone": ((240, 110, 30), 0.50, 0.35),
    "crate": ((50, 90, 220), 0.80, 0.80),
    "pedestrian": ((240, 200, 40), 1.70, 0.45),
}

# The world: static objects on/near the robot's elliptical loop.
# (label, x_map, y_map)
WORLD_OBJECTS = [
    ("cone", 6.0, 3.0),
    ("cone", -4.0, 7.5),
    ("cone", -11.0, -3.0),
    ("cone", 10.5, -6.0),
    ("cone", 1.5, -8.0),
    ("crate", 15.5, 2.0),
    ("crate", -15.0, 4.5),
    ("crate", -7.0, -10.5),
    ("crate", 5.0, 11.0),
    ("pedestrian", 12.0, 7.0),
    ("pedestrian", -13.5, -6.5),
    ("pedestrian", 0.5, 0.5),
    ("cone", 13.5, 4.5),
    ("cone", 7.0, 9.5),
    ("cone", -2.0, 10.5),
    ("cone", -9.0, 6.0),
    ("cone", -14.0, -1.0),
    ("cone", -6.0, -9.5),
    ("cone", 3.0, -10.5),
    ("cone", 11.0, -4.0),
    ("crate", 17.0, -3.0),
    ("crate", -17.0, -2.0),
    ("crate", 0.0, 13.0),
    ("pedestrian", 16.0, 6.0),
    ("pedestrian", -10.0, 10.0),
    ("pedestrian", 8.0, -11.0),
]

# Robot path: ellipse, one lap in LAP_TIME_S seconds.
ELLIPSE_A, ELLIPSE_B = 13.0, 8.5
LAP_TIME_S = 100.0


def robot_pose(time_s: float) -> tuple[float, float, float]:
    """(x, y, yaw) of base_link in map at time t along the ellipse."""
    theta = 2.0 * np.pi * (time_s / LAP_TIME_S)
    x = ELLIPSE_A * np.cos(theta)
    y = ELLIPSE_B * np.sin(theta)
    dx = -ELLIPSE_A * np.sin(theta)
    dy = ELLIPSE_B * np.cos(theta)
    return float(x), float(y), float(np.arctan2(dy, dx))


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _camera_origin_map(pose: tuple[float, float, float]) -> np.ndarray:
    x, y, yaw = pose
    return np.array(
        [x + CAM_OFFSET_X_M * np.cos(yaw), y + CAM_OFFSET_X_M * np.sin(yaw)]
    )


def world_to_optical(
    point_xy: np.ndarray, z_world: float, pose: tuple[float, float, float]
) -> np.ndarray:
    """Map-frame point (x, y, z) -> camera optical frame (x right, y down, z fwd)."""
    _x, _y, yaw = pose
    rel = np.asarray(point_xy, dtype=float) - _camera_origin_map(pose)
    # Rotate into base frame (x fwd, y left).
    c, s = np.cos(-yaw), np.sin(-yaw)
    x_base = c * rel[0] - s * rel[1]
    y_base = s * rel[0] + c * rel[1]
    # Optical: z = forward, x = -left, y = -(up relative to camera).
    return np.array([-y_base, -(z_world - CAM_HEIGHT_M), x_base])


def project(optical: np.ndarray) -> tuple[float, float] | None:
    """Optical point -> pixel (u, v), or None if behind/too close."""
    x_o, y_o, z_o = optical
    if z_o < MIN_DEPTH_M:
        return None
    return FX * x_o / z_o + CX, FY * y_o / z_o + CY


def render_frame(
    pose: tuple[float, float, float],
) -> tuple[np.ndarray, list[dict]]:
    """Render the world from the robot's camera. Returns (image, visible truth).

    Each visible-truth record: {object_index, label, box, range_m,
    map_xy} where box is the exact projected rectangle [x1, y1, x2, y2].
    """
    img = Image.new("RGB", (IMG_W, IMG_H), (28, 32, 40))
    draw = ImageDraw.Draw(img)
    horizon = int(CY)
    draw.rectangle([0, horizon, IMG_W, IMG_H], fill=(56, 56, 56))
    draw.line([0, horizon, IMG_W, horizon], fill=(90, 90, 90), width=2)

    records = []
    for idx, (label, ox, oy) in enumerate(WORLD_OBJECTS):
        color, height_m, width_m = CLASSES[label]
        base_opt = world_to_optical(np.array([ox, oy]), 0.0, pose)
        depth = float(base_opt[2])
        if depth < MIN_DEPTH_M or depth > MAX_RANGE_M:
            continue
        base_px = project(base_opt)
        top_px = project(world_to_optical(np.array([ox, oy]), height_m, pose))
        if base_px is None or top_px is None:
            continue
        u_c, v_base = base_px
        v_top = top_px[1]
        width_px = FX * width_m / depth
        box = [
            u_c - width_px / 2.0,
            v_top,
            u_c + width_px / 2.0,
            v_base,
        ]
        if box[2] < 0 or box[0] > IMG_W:
            continue
        records.append(
            {
                "object_index": idx,
                "label": label,
                "box": box,
                "range_m": float(np.hypot(*(np.array([ox, oy]) - np.array(pose[:2])))),
                "map_xy": (ox, oy),
                "depth_m": depth,
            }
        )

    # Painter's algorithm: far objects first. Boxes are clamped to the
    # image; a box that clamps to zero area is not actually visible and
    # its record is dropped.
    visible_records = []
    for rec in sorted(records, key=lambda r: -r["depth_m"]):
        color = CLASSES[rec["label"]][0]
        x1, y1, x2, y2 = [int(round(v)) for v in rec["box"]]
        x1c, x2c = max(0, x1), min(IMG_W - 1, x2)
        y1c, y2c = max(0, y1), min(IMG_H - 1, y2)
        if x2c <= x1c or y2c <= y1c:
            continue
        draw.rectangle([x1c, y1c, x2c, y2c], fill=color, outline=color, width=2)
        visible_records.append(rec)
    return np.asarray(img), visible_records


def detect_color(frame: np.ndarray, min_area: int = 120) -> list[dict]:
    """Detect objects by nearest-palette-color segmentation.

    Every pixel is assigned to its nearest class color (Manhattan
    distance in RGB); pixels too far from every palette color are
    background. Per class, pixels are clustered by x-runs, as in
    Project 1, so two same-class objects in one frame stay separate.
    """
    img = frame.astype(int)
    labels = list(CLASSES)
    dists = np.stack(
        [
            np.abs(img[:, :, 0] - CLASSES[label][0][0])
            + np.abs(img[:, :, 1] - CLASSES[label][0][1])
            + np.abs(img[:, :, 2] - CLASSES[label][0][2])
            for label in labels
        ]
    )
    nearest = np.argmin(dists, axis=0)
    min_dist = np.min(dists, axis=0)

    detections = []
    for li, label in enumerate(labels):
        mask = (nearest == li) & (min_dist < 110)
        ys, xs = np.nonzero(mask)
        if len(xs) < min_area:
            continue
        order = np.argsort(xs)
        xs_s, ys_s = xs[order], ys[order]
        splits = np.where(np.diff(xs_s) > 30)[0]
        for grp in np.split(np.arange(len(xs_s)), splits + 1):
            if len(grp) < min_area:
                continue
            gx, gy = xs_s[grp], ys_s[grp]
            detections.append(
                {
                    "box": np.array(
                        [gx.min(), gy.min(), gx.max(), gy.max()], dtype=float
                    ),
                    "label": label,
                    "score": 0.99,
                }
            )
    return detections


def localize_box(
    box: np.ndarray,
    label: str,
    fx: float = FX,
    fy: float = FY,
    cx: float = CX,
    cy: float = CY,
) -> np.ndarray:
    """Box + class -> 3D point in the camera optical frame (metres).

    Depth comes from the known real height of the class:
    Z = fy * H_real / box_height_px. The object's ground-contact point
    (bottom-centre of the box) defines the back-projected ray.
    Intrinsics default to this module's camera; the ROS node passes the
    values from the live CameraInfo topic instead.
    """
    height_m = CLASSES[label][1]
    x1, y1, x2, y2 = [float(v) for v in box]
    height_px = max(1.0, y2 - y1)
    z_o = fy * height_m / height_px
    u_c = 0.5 * (x1 + x2)
    x_o = (u_c - cx) * z_o / fx
    y_o = (y2 - cy) * z_o / fy  # bottom of box: the ground contact ray
    return np.array([x_o, y_o, z_o])


def optical_to_base(optical: np.ndarray) -> np.ndarray:
    """Optical point -> point in base_link axes, relative to the camera."""
    x_o, y_o, z_o = optical
    return np.array([z_o, -x_o, -y_o + CAM_HEIGHT_M])


def base_to_map(
    point_base: np.ndarray, pose: tuple[float, float, float]
) -> np.ndarray:
    """Base-frame point (relative to camera) -> map-frame (x, y, z)."""
    x, y, yaw = pose
    origin = _camera_origin_map(pose)
    c, s = np.cos(yaw), np.sin(yaw)
    rotated = np.array(
        [
            c * point_base[0] - s * point_base[1],
            s * point_base[0] + c * point_base[1],
        ]
    )
    return np.array([origin[0] + rotated[0], origin[1] + rotated[1], point_base[2]])


def estimate_map_position(
    box: np.ndarray, label: str, pose: tuple[float, float, float]
) -> np.ndarray:
    """Full offline chain: detection box -> estimated map-frame position."""
    return base_to_map(optical_to_base(localize_box(box, label)), pose)
