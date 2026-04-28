#!/usr/bin/env python3
"""Track YOLO detections with a SORT-style Kalman/IoU tracker.

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
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


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
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1

    @property
    def measurement(self) -> np.ndarray:
        cx = (self.x1 + self.x2) / 2.0
        cy = (self.y1 + self.y2) / 2.0
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        return np.array([[cx], [cy], [width], [height]], dtype=np.float32)


class SortTrack:
    def __init__(self, track_id: int, detection: Detection) -> None:
        self.track_id = track_id
        self.kalman = cv2.KalmanFilter(8, 4)
        self.kalman.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kalman.errorCovPost = np.eye(8, dtype=np.float32)

        measurement = detection.measurement
        self.kalman.statePost = np.array(
            [
                [measurement[0, 0]],
                [measurement[1, 0]],
                [measurement[2, 0]],
                [measurement[3, 0]],
                [0],
                [0],
                [0],
                [0],
            ],
            dtype=np.float32,
        )
        self.predicted_box = detection.xyxy
        self.last_detection = detection
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0

    def predict(self) -> tuple[float, float, float, float]:
        prediction = self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.predicted_box = state_to_xyxy(prediction)
        return self.predicted_box

    def update(self, detection: Detection) -> None:
        self.kalman.correct(detection.measurement)
        self.last_detection = detection
        self.predicted_box = detection.xyxy
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def output_box(self) -> tuple[float, float, float, float]:
        if self.time_since_update == 0:
            return self.last_detection.xyxy
        return self.predicted_box


def state_to_xyxy(state: np.ndarray) -> tuple[float, float, float, float]:
    cx = float(state[0, 0])
    cy = float(state[1, 0])
    width = max(1.0, float(state[2, 0]))
    height = max(1.0, float(state[3, 0]))
    return (
        cx - width / 2.0,
        cy - height / 2.0,
        cx + width / 2.0,
        cy + height / 2.0,
    )


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


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
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


def assign_detections(
    tracks: list[SortTrack],
    detections: list[Detection],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    if not tracks or not detections:
        return [], set(range(len(tracks))), set(range(len(detections)))

    cost_matrix = [
        [-iou(track.predicted_box, detection.xyxy) for detection in detections]
        for track in tracks
    ]
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()

    for row_index, col_index in zip(row_indices, col_indices):
        overlap = -cost_matrix[row_index][col_index]
        if overlap >= iou_threshold:
            matches.append((row_index, col_index))
            matched_tracks.add(row_index)
            matched_detections.add(col_index)

    unmatched_tracks = set(range(len(tracks))) - matched_tracks
    unmatched_detections = set(range(len(detections))) - matched_detections
    return matches, unmatched_tracks, unmatched_detections


def run_sort(
    detections_by_frame: dict[int, list[Detection]],
    iou_threshold: float,
    max_age: int,
    min_hits: int,
) -> list[list[float | int]]:
    if not detections_by_frame:
        return []

    tracks: list[SortTrack] = []
    next_track_id = 1
    rows: list[list[float | int]] = []
    first_frame = min(detections_by_frame)
    last_frame = max(detections_by_frame)

    for frame in range(first_frame, last_frame + 1):
        detections = detections_by_frame.get(frame, [])
        for track in tracks:
            track.predict()

        matches, _unmatched_tracks, unmatched_detections = assign_detections(
            tracks,
            detections,
            iou_threshold,
        )

        for track_index, detection_index in matches:
            tracks[track_index].update(detections[detection_index])

        for detection_index in sorted(unmatched_detections):
            tracks.append(SortTrack(next_track_id, detections[detection_index]))
            next_track_id += 1

        active_tracks: list[SortTrack] = []
        for track in tracks:
            if track.time_since_update <= max_age:
                active_tracks.append(track)
        tracks = active_tracks

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
    iou_threshold: float,
    max_age: int,
    min_hits: int,
) -> int:
    detections_by_frame = load_detections(detections_path, min_conf)
    rows = run_sort(detections_by_frame, iou_threshold, max_age, min_hits)
    write_tracks(output_path, rows)
    return len(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_detections_dir = repo_root / "outputs" / "detections" / "yolo"
    default_output_dir = repo_root / "outputs" / "tracks" / "sort"

    parser = argparse.ArgumentParser()
    parser.add_argument("--detections-dir", default=str(default_detections_dir))
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--video-id", action="append", help="Track one video id. Repeatable.")
    parser.add_argument("--min-conf", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-age", type=int, default=10)
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
            iou_threshold=args.iou_threshold,
            max_age=args.max_age,
            min_hits=args.min_hits,
        )
        print(f"Wrote {rows} track rows to {output_path}")


if __name__ == "__main__":
    main()
