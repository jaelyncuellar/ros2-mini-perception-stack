"""Offline evaluation of the perception core (no ROS required).

Drives the simulated robot around one full lap, runs the exact code the
ROS nodes run (render -> detect -> localize -> transform to map), and
scores the estimated object positions against ground truth:

* every detection is matched to the nearest same-class world object;
  matches farther than MATCH_GATE_M are counted as false positives;
* visible objects with no detection are counted as missed;
* errors are reported separately for clean detections and for boxes
  clipped by the image border (a truncated box height corrupts the
  monocular depth estimate — a real, well-known failure mode).

Outputs in ``out/``: metrics.json, metrics.txt, topdown.png (map-frame
estimates vs. truth), and a sample camera frame (sample_frame.png).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(
    0, str(PROJECT_DIR / "mini_perception_stack")
)
from mini_perception_stack import perception_core as core  # noqa: E402

MATCH_GATE_M = 2.0


def is_clipped(box: np.ndarray) -> bool:
    return bool(
        box[0] <= 2
        or box[1] <= 2
        or box[2] >= core.IMG_W - 3
        or box[3] >= core.IMG_H - 3
    )


def run_evaluation(duration_s: float = core.LAP_TIME_S, step_s: float = 0.5) -> dict:
    errors_clean: list[float] = []
    errors_clipped: list[float] = []
    depth_errors: list[float] = []
    false_positives = 0
    visible_total = 0
    detected_total = 0
    matched_truth: set[tuple[int, int]] = set()  # (frame_no, object_index)
    estimates_xy: list[tuple[float, float]] = []
    sample_frame = None

    frame_count = int(duration_s / step_s)
    for frame_no in range(frame_count):
        t = frame_no * step_s
        pose = core.robot_pose(t)
        frame, visible = core.render_frame(pose)
        if frame_no == 10:
            sample_frame = frame
        detections = core.detect_color(frame)
        visible_total += len(visible)
        detected_total += len(detections)

        visible_by_index = {v["object_index"]: v for v in visible}
        for det in detections:
            est = core.estimate_map_position(det["box"], det["label"], pose)
            estimates_xy.append((float(est[0]), float(est[1])))
            # Match to the nearest visible truth object of the same class.
            best_idx, best_err = None, None
            for v in visible:
                if v["label"] != det["label"]:
                    continue
                err = float(np.hypot(est[0] - v["map_xy"][0], est[1] - v["map_xy"][1]))
                if best_err is None or err < best_err:
                    best_idx, best_err = v["object_index"], err
            if best_idx is None or best_err > MATCH_GATE_M:
                false_positives += 1
                continue
            matched_truth.add((frame_no, best_idx))
            truth_depth = visible_by_index[best_idx]["depth_m"]
            est_depth = float(core.localize_box(det["box"], det["label"])[2])
            depth_errors.append(abs(est_depth - truth_depth))
            if is_clipped(det["box"]):
                errors_clipped.append(best_err)
            else:
                errors_clean.append(best_err)

    missed = visible_total - len(matched_truth)

    def rmse(values: list[float]) -> float | None:
        if not values:
            return None
        return float(np.sqrt(np.mean(np.array(values) ** 2)))

    metrics = {
        "frames": frame_count,
        "step_s": step_s,
        "visible_object_frames": visible_total,
        "detections": detected_total,
        "matched": len(matched_truth),
        "missed": missed,
        "false_positives": false_positives,
        "position_rmse_clean_m": rmse(errors_clean),
        "position_mean_error_clean_m": (
            float(np.mean(errors_clean)) if errors_clean else None
        ),
        "position_max_error_clean_m": (
            float(np.max(errors_clean)) if errors_clean else None
        ),
        "clean_detections_scored": len(errors_clean),
        "position_rmse_clipped_m": rmse(errors_clipped),
        "clipped_detections_scored": len(errors_clipped),
        "depth_mae_m": float(np.mean(depth_errors)) if depth_errors else None,
    }
    return {"metrics": metrics, "estimates_xy": estimates_xy, "sample_frame": sample_frame}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=core.LAP_TIME_S)
    parser.add_argument("--step", type=float, default=0.5)
    args = parser.parse_args()

    result = run_evaluation(args.duration, args.step)
    metrics = result["metrics"]
    out_dir = PROJECT_DIR / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    lines = [
        "Project 3 - offline perception evaluation (ROS-free core, synthetic world)",
        f"frames: {metrics['frames']} at {metrics['step_s']} s steps "
        f"(one full {core.LAP_TIME_S:.0f} s lap)",
        f"visible object-frames: {metrics['visible_object_frames']}, "
        f"detections: {metrics['detections']}, matched: {metrics['matched']}, "
        f"missed: {metrics['missed']}, false positives: {metrics['false_positives']}",
        "",
        f"map-frame position RMSE (clean boxes):   {metrics['position_rmse_clean_m']:.2f} m "
        f"over {metrics['clean_detections_scored']} detections",
        f"map-frame mean error (clean boxes):      {metrics['position_mean_error_clean_m']:.2f} m",
        f"map-frame max error (clean boxes):       {metrics['position_max_error_clean_m']:.2f} m",
        f"map-frame position RMSE (clipped boxes): {metrics['position_rmse_clipped_m']:.2f} m "
        f"over {metrics['clipped_detections_scored']} detections",
        f"monocular depth mean abs error:          {metrics['depth_mae_m']:.2f} m",
    ]
    (out_dir / "metrics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    if result["sample_frame"] is not None:
        from PIL import Image

        Image.fromarray(result["sample_frame"]).save(out_dir / "sample_frame.png")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    truth_x = [o[1] for o in core.WORLD_OBJECTS]
    truth_y = [o[2] for o in core.WORLD_OBJECTS]
    labels = [o[0] for o in core.WORLD_OBJECTS]
    colors = {"cone": "#f06e1e", "crate": "#325adc", "pedestrian": "#d8b400"}
    if result["estimates_xy"]:
        ex, ey = zip(*result["estimates_xy"])
        ax.scatter(ex, ey, s=9, color="#0f766e", alpha=0.55,
                   label="localized estimates (map frame)", zorder=3)
    # Truth as hollow squares so the estimate dots inside them stay visible.
    for x, y, label in zip(truth_x, truth_y, labels):
        ax.scatter([x], [y], s=150, marker="s", facecolors="none",
                   edgecolors=colors[label], linewidths=1.8, zorder=4)
    path_theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(core.ELLIPSE_A * np.cos(path_theta), core.ELLIPSE_B * np.sin(path_theta),
            color="#666666", lw=1.2, ls="--", label="robot path")
    ax.set_xlabel("x (m, map)")
    ax.set_ylabel("y (m, map)")
    ax.set_title("Project 3 offline eval - ground truth (squares) vs. estimates (dots)")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "topdown.png", dpi=150)
    print(f"\noutputs in {out_dir}: metrics.txt/json, topdown.png, sample_frame.png")


if __name__ == "__main__":
    main()
