#!/usr/bin/env python3
"""Track YOLO detections with a FRoG-MOT-inspired tracker.

This is a pragmatic project-compatible variant inspired by FRoG-MOT's
combination of IoU association and motion-state association. It is not an
exact reproduction of the WACV 2024 paper equations.

Input detections:
frame,x1,y1,x2,y2,confidence,class

Output tracks use MOT-like rows:
frame,id,x,y,w,h,confidence,class,visibility
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

from scipy.optimize import linear_sum_assignment


Box = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass
class Detection:
    frame: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def xyxy(self) -> Box:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> Point:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return max(1.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(1.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return max(1.0, self.width * self.height)

    @property
    def aspect_ratio(self) -> float:
        return clamp(self.width / self.height, 0.1, 10.0)


class FrogTrack:
    def __init__(self, track_id: int, detection: Detection) -> None:
        self.track_id = track_id
        self.box = detection.xyxy
        self.predicted_box = detection.xyxy
        self.center = detection.center
        self.predicted_center = detection.center
        self.velocity = (0.0, 0.0)
        self.acceleration = (0.0, 0.0)
        self.aspect_ratio = detection.aspect_ratio
        self.aspect_delta = 0.0
        self.area = detection.area
        self.last_detection = detection
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0

    def predict(self) -> Box:
        self.age += 1
        self.time_since_update += 1
        if self.time_since_update > 0:
            self.hit_streak = 0

        predicted_center = (
            self.center[0] + self.velocity[0] + 0.5 * self.acceleration[0],
            self.center[1] + self.velocity[1] + 0.5 * self.acceleration[1],
        )
        predicted_aspect = clamp(self.aspect_ratio + self.aspect_delta, 0.1, 10.0)
        self.predicted_center = predicted_center
        self.predicted_box = box_from_center_area_aspect(
            predicted_center,
            self.area,
            predicted_aspect,
        )
        return self.predicted_box

    def update(
        self,
        detection: Detection,
        normal_alpha: float,
        outlier_alpha: float,
        outlier_component: str | None = None,
    ) -> None:
        measured_center = detection.center
        measured_velocity = subtract_points(measured_center, self.center)
        measured_acceleration = subtract_points(measured_velocity, self.velocity)
        measured_aspect_delta = detection.aspect_ratio - self.aspect_ratio

        self.velocity = blend_point(
            self.velocity,
            measured_velocity,
            alpha_for_component("velocity", outlier_component, normal_alpha, outlier_alpha),
        )
        self.acceleration = blend_point(
            self.acceleration,
            measured_acceleration,
            alpha_for_component("acceleration", outlier_component, normal_alpha, outlier_alpha),
        )
        self.aspect_delta = blend_value(
            self.aspect_delta,
            measured_aspect_delta,
            alpha_for_component("aspect", outlier_component, normal_alpha, outlier_alpha),
        )

        self.box = detection.xyxy
        self.predicted_box = detection.xyxy
        self.center = measured_center
        self.predicted_center = measured_center
        self.aspect_ratio = detection.aspect_ratio
        self.area = detection.area
        self.last_detection = detection
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def output_box(self) -> Box:
        if self.time_since_update == 0:
            return self.box
        return self.predicted_box


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def subtract_points(point_a: Point, point_b: Point) -> Point:
    return point_a[0] - point_b[0], point_a[1] - point_b[1]


def point_norm(point: Point) -> float:
    return sqrt(point[0] * point[0] + point[1] * point[1])


def blend_value(old: float, new: float, alpha: float) -> float:
    return alpha * old + (1.0 - alpha) * new


def blend_point(old: Point, new: Point, alpha: float) -> Point:
    return blend_value(old[0], new[0], alpha), blend_value(old[1], new[1], alpha)


def alpha_for_component(
    component: str,
    outlier_component: str | None,
    normal_alpha: float,
    outlier_alpha: float,
) -> float:
    return outlier_alpha if component == outlier_component else normal_alpha


def box_from_center_area_aspect(center: Point, area: float, aspect_ratio: float) -> Box:
    width = sqrt(max(1.0, area) * clamp(aspect_ratio, 0.1, 10.0))
    height = max(1.0, area / max(1.0, width))
    cx, cy = center
    return cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0


def load_detections(path: Path, min_conf: float) -> dict[int, list[Detection]]:
    detections_by_frame: dict[int, list[Detection]] = defaultdict(list)
    with path.open() as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            confidence = float(row[5])
            if confidence < min_conf:
                continue
            detection = Detection(
                frame=int(float(row[0])),
                x1=float(row[1]),
                y1=float(row[2]),
                x2=float(row[3]),
                y2=float(row[4]),
                confidence=confidence,
                class_id=int(float(row[6])),
            )
            detections_by_frame[detection.frame].append(detection)
    return detections_by_frame


def iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def assign_iou(
    tracks: list[FrogTrack],
    detections: list[Detection],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, str | None]], set[int], set[int]]:
    if not tracks or not detections:
        return [], set(range(len(tracks))), set(range(len(detections)))

    cost_matrix = [
        [-iou(track.predicted_box, detection.xyxy) for detection in detections]
        for track in tracks
    ]
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches: list[tuple[int, int, str | None]] = []
    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()
    for row_index, col_index in zip(row_indices, col_indices):
        overlap = -cost_matrix[row_index][col_index]
        if overlap < iou_threshold:
            continue
        matches.append((row_index, col_index, None))
        matched_tracks.add(row_index)
        matched_detections.add(col_index)

    return (
        matches,
        set(range(len(tracks))) - matched_tracks,
        set(range(len(detections))) - matched_detections,
    )


def motion_state_cost(
    track: FrogTrack,
    detection: Detection,
    motion_scale: float,
) -> tuple[float, str | None]:
    measured_center = detection.center
    measured_velocity = subtract_points(measured_center, track.center)
    measured_acceleration = subtract_points(measured_velocity, track.velocity)
    measured_aspect_delta = detection.aspect_ratio - track.aspect_ratio

    size_scale = max(1.0, sqrt(max(track.area, detection.area)))
    position_error = point_norm(subtract_points(measured_center, track.predicted_center))
    position_cost = clamp(position_error / (size_scale * motion_scale), 0.0, 1.0)

    velocity_cost = normalized_vector_cost(
        subtract_points(measured_velocity, track.velocity),
        point_norm(track.velocity),
        size_scale,
    )
    acceleration_cost = normalized_vector_cost(
        subtract_points(measured_acceleration, track.acceleration),
        point_norm(track.acceleration),
        size_scale,
    )
    aspect_cost = clamp(
        abs(measured_aspect_delta - track.aspect_delta) / (abs(track.aspect_delta) + 0.25),
        0.0,
        1.0,
    )

    total_cost = (
        0.45 * position_cost
        + 0.25 * velocity_cost
        + 0.20 * acceleration_cost
        + 0.10 * aspect_cost
    )

    components = {
        "velocity": velocity_cost,
        "acceleration": acceleration_cost,
        "aspect": aspect_cost,
    }
    worst_component, worst_cost = max(components.items(), key=lambda item: item[1])
    outlier_component = worst_component if worst_cost >= 0.8 else None
    return total_cost, outlier_component


def normalized_vector_cost(delta: Point, current_magnitude: float, size_scale: float) -> float:
    denominator = max(1.0, current_magnitude + 0.25 * size_scale)
    return clamp(point_norm(delta) / denominator, 0.0, 1.0)


def assign_motion_state(
    tracks: list[FrogTrack],
    detections: list[Detection],
    unmatched_tracks: set[int],
    unmatched_detections: set[int],
    threshold: float,
    motion_scale: float,
) -> tuple[list[tuple[int, int, str | None]], set[int], set[int]]:
    if not unmatched_tracks or not unmatched_detections:
        return [], unmatched_tracks, unmatched_detections

    track_indices = sorted(unmatched_tracks)
    detection_indices = sorted(unmatched_detections)
    costs: list[list[float]] = []
    outlier_components: dict[tuple[int, int], str | None] = {}

    for track_index in track_indices:
        row = []
        for detection_index in detection_indices:
            cost, outlier_component = motion_state_cost(
                tracks[track_index],
                detections[detection_index],
                motion_scale,
            )
            row.append(cost)
            outlier_components[(track_index, detection_index)] = outlier_component
        costs.append(row)

    row_indices, col_indices = linear_sum_assignment(costs)
    matches: list[tuple[int, int, str | None]] = []
    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()

    for row_index, col_index in zip(row_indices, col_indices):
        cost = costs[row_index][col_index]
        if cost > threshold:
            continue
        track_index = track_indices[row_index]
        detection_index = detection_indices[col_index]
        matches.append(
            (
                track_index,
                detection_index,
                outlier_components[(track_index, detection_index)],
            )
        )
        matched_tracks.add(track_index)
        matched_detections.add(detection_index)

    return (
        matches,
        unmatched_tracks - matched_tracks,
        unmatched_detections - matched_detections,
    )


def run_frog_mot(
    detections_by_frame: dict[int, list[Detection]],
    primary_iou_threshold: float,
    secondary_cost_threshold: float,
    motion_scale: float,
    normal_alpha: float,
    outlier_alpha: float,
    max_age: int,
    min_hits: int,
) -> list[list[float | int]]:
    if not detections_by_frame:
        return []

    tracks: list[FrogTrack] = []
    next_track_id = 1
    rows: list[list[float | int]] = []
    first_frame = min(detections_by_frame)
    last_frame = max(detections_by_frame)

    for frame in range(first_frame, last_frame + 1):
        detections = detections_by_frame.get(frame, [])
        for track in tracks:
            track.predict()

        iou_matches, unmatched_tracks, unmatched_detections = assign_iou(
            tracks,
            detections,
            primary_iou_threshold,
        )
        motion_matches, _unmatched_tracks, unmatched_detections = assign_motion_state(
            tracks,
            detections,
            unmatched_tracks,
            unmatched_detections,
            secondary_cost_threshold,
            motion_scale,
        )

        for track_index, detection_index, outlier_component in iou_matches + motion_matches:
            tracks[track_index].update(
                detections[detection_index],
                normal_alpha=normal_alpha,
                outlier_alpha=outlier_alpha,
                outlier_component=outlier_component,
            )

        for detection_index in sorted(unmatched_detections):
            tracks.append(FrogTrack(next_track_id, detections[detection_index]))
            next_track_id += 1

        tracks = [track for track in tracks if track.time_since_update <= max_age]

        for track in sorted(tracks, key=lambda item: item.track_id):
            if track.time_since_update != 0:
                continue
            if track.hits < min_hits:
                continue
            x1, y1, x2, y2 = track.output_box()
            rows.append(
                [
                    frame,
                    track.track_id,
                    round(x1, 3),
                    round(y1, 3),
                    round(x2 - x1, 3),
                    round(y2 - y1, 3),
                    round(track.last_detection.confidence, 6),
                    track.last_detection.class_id,
                    -1,
                ]
            )

    return rows


def write_tracks(path: Path, rows: list[list[float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def track_file(
    detections_path: Path,
    output_path: Path,
    min_conf: float,
    primary_iou_threshold: float,
    secondary_cost_threshold: float,
    motion_scale: float,
    normal_alpha: float,
    outlier_alpha: float,
    max_age: int,
    min_hits: int,
) -> int:
    detections_by_frame = load_detections(detections_path, min_conf)
    rows = run_frog_mot(
        detections_by_frame=detections_by_frame,
        primary_iou_threshold=primary_iou_threshold,
        secondary_cost_threshold=secondary_cost_threshold,
        motion_scale=motion_scale,
        normal_alpha=normal_alpha,
        outlier_alpha=outlier_alpha,
        max_age=max_age,
        min_hits=min_hits,
    )
    write_tracks(output_path, rows)
    return len(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_detections_dir = repo_root / "outputs" / "detections" / "yolo_conf035"
    default_output_dir = repo_root / "outputs" / "tracks" / "frogmot_conf035"

    parser = argparse.ArgumentParser()
    parser.add_argument("--detections-dir", default=str(default_detections_dir))
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--video-id", action="append", help="Track one video id. Repeatable.")
    parser.add_argument("--min-conf", type=float, default=0.0)
    parser.add_argument("--primary-iou-threshold", type=float, default=0.25)
    parser.add_argument("--secondary-cost-threshold", type=float, default=0.65)
    parser.add_argument("--motion-scale", type=float, default=3.0)
    parser.add_argument("--normal-alpha", type=float, default=0.60)
    parser.add_argument("--outlier-alpha", type=float, default=0.85)
    parser.add_argument("--max-age", type=int, default=15)
    parser.add_argument("--min-hits", type=int, default=1)
    args = parser.parse_args()

    detections_dir = Path(args.detections_dir)
    output_dir = Path(args.output_dir)

    if args.video_id:
        detection_files = [detections_dir / f"{video_id}.txt" for video_id in args.video_id]
    else:
        detection_files = sorted(detections_dir.glob("*.txt"))

    for detections_path in detection_files:
        if not detections_path.exists():
            print(f"Skipping missing detections: {detections_path}")
            continue
        output_path = output_dir / detections_path.name
        rows = track_file(
            detections_path=detections_path,
            output_path=output_path,
            min_conf=args.min_conf,
            primary_iou_threshold=args.primary_iou_threshold,
            secondary_cost_threshold=args.secondary_cost_threshold,
            motion_scale=args.motion_scale,
            normal_alpha=args.normal_alpha,
            outlier_alpha=args.outlier_alpha,
            max_age=args.max_age,
            min_hits=args.min_hits,
        )
        print(f"Wrote {rows} track rows to {output_path}")


if __name__ == "__main__":
    main()
