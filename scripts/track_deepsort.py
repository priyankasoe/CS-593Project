#!/usr/bin/env python3
"""Track YOLO detections with a DeepSORT-lite tracker.

This is a dependency-light approximation of DeepSORT:
- Kalman prediction over bounding-box center/size, as in SORT.
- Hungarian assignment.
- Matching cost combines IoU motion overlap and HSV appearance similarity.

Input detections:
frame,x1,y1,x2,y2,confidence,class

Output tracks use MOT-like rows:
frame,id,x,y,w,h,confidence,class,visibility
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    appearance: np.ndarray

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def measurement(self) -> np.ndarray:
        cx = (self.x1 + self.x2) / 2.0
        cy = (self.y1 + self.y2) / 2.0
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        return np.array([[cx], [cy], [width], [height]], dtype=np.float32)


class DeepSortLiteTrack:
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
        self.appearance = detection.appearance.copy()
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

    def update(self, detection: Detection, appearance_alpha: float) -> None:
        self.kalman.correct(detection.measurement)
        self.last_detection = detection
        self.predicted_box = detection.xyxy
        self.appearance = (
            appearance_alpha * self.appearance
            + (1.0 - appearance_alpha) * detection.appearance
        )
        norm = np.linalg.norm(self.appearance)
        if norm > 0:
            self.appearance = self.appearance / norm
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


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    metadata = json.loads(path.read_text())
    return {video["id"]: video for video in metadata["videos"]}


def load_detection_rows(path: Path, min_conf: float) -> dict[int, list[tuple[float, float, float, float, float, int]]]:
    detections_by_frame: dict[int, list[tuple[float, float, float, float, float, int]]] = defaultdict(list)
    with path.open() as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            confidence = float(row[5])
            if confidence < min_conf:
                continue
            frame = int(float(row[0]))
            detections_by_frame[frame].append(
                (
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    confidence,
                    int(float(row[6])),
                )
            )
    return detections_by_frame


def crop_histogram(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    bins: tuple[int, int] = (16, 16),
) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1_i = max(0, min(width - 1, int(round(x1))))
    y1_i = max(0, min(height - 1, int(round(y1))))
    x2_i = max(0, min(width, int(round(x2))))
    y2_i = max(0, min(height, int(round(y2))))

    if x2_i <= x1_i or y2_i <= y1_i:
        return np.zeros((bins[0] * bins[1],), dtype=np.float32)

    crop = frame[y1_i:y2_i, x1_i:x2_i]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(bins), [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist = hist / norm
    return hist


def build_detections_for_frame(
    frame_number: int,
    frame: np.ndarray,
    detection_rows: list[tuple[float, float, float, float, float, int]],
) -> list[Detection]:
    detections: list[Detection] = []
    for x1, y1, x2, y2, confidence, class_id in detection_rows:
        detections.append(
            Detection(
                frame=frame_number,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=confidence,
                class_id=class_id,
                appearance=crop_histogram(frame, (x1, y1, x2, y2)),
            )
        )
    return detections


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


def appearance_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    if hist_a.size == 0 or hist_b.size == 0:
        return 1.0
    similarity = float(np.dot(hist_a, hist_b))
    return max(0.0, min(1.0, 1.0 - similarity))


def combined_cost(
    track: DeepSortLiteTrack,
    detection: Detection,
    appearance_weight: float,
) -> tuple[float, float, float]:
    overlap = iou(track.predicted_box, detection.xyxy)
    motion_cost = 1.0 - overlap
    app_cost = appearance_distance(track.appearance, detection.appearance)
    total = (1.0 - appearance_weight) * motion_cost + appearance_weight * app_cost
    return total, overlap, app_cost


def assign_detections(
    tracks: list[DeepSortLiteTrack],
    detections: list[Detection],
    appearance_weight: float,
    max_cost: float,
    min_iou: float,
    max_appearance_distance: float,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    if not tracks or not detections:
        return [], set(range(len(tracks))), set(range(len(detections)))

    cost_matrix: list[list[float]] = []
    overlap_matrix: list[list[float]] = []
    appearance_matrix: list[list[float]] = []
    for track in tracks:
        cost_row: list[float] = []
        overlap_row: list[float] = []
        appearance_row: list[float] = []
        for detection in detections:
            cost, overlap, app_cost = combined_cost(track, detection, appearance_weight)
            cost_row.append(cost)
            overlap_row.append(overlap)
            appearance_row.append(app_cost)
        cost_matrix.append(cost_row)
        overlap_matrix.append(overlap_row)
        appearance_matrix.append(appearance_row)

    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()

    for row_index, col_index in zip(row_indices, col_indices):
        cost = cost_matrix[row_index][col_index]
        overlap = overlap_matrix[row_index][col_index]
        app_cost = appearance_matrix[row_index][col_index]
        if cost <= max_cost and overlap >= min_iou and app_cost <= max_appearance_distance:
            matches.append((row_index, col_index))
            matched_tracks.add(row_index)
            matched_detections.add(col_index)

    unmatched_tracks = set(range(len(tracks))) - matched_tracks
    unmatched_detections = set(range(len(detections))) - matched_detections
    return matches, unmatched_tracks, unmatched_detections


def run_deepsort_lite(
    video_path: Path,
    detections_by_frame: dict[int, list[tuple[float, float, float, float, float, int]]],
    appearance_weight: float,
    appearance_alpha: float,
    max_cost: float,
    min_iou: float,
    max_appearance_distance: float,
    max_age: int,
    min_hits: int,
) -> list[list[float | int]]:
    if not detections_by_frame:
        return []

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    tracks: list[DeepSortLiteTrack] = []
    next_track_id = 1
    rows: list[list[float | int]] = []
    first_frame = min(detections_by_frame)
    last_frame = max(detections_by_frame)
    zero_based_frame = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_number = zero_based_frame + 1
        zero_based_frame += 1

        if frame_number < first_frame:
            continue
        if frame_number > last_frame:
            break

        for track in tracks:
            track.predict()

        detection_rows = detections_by_frame.get(frame_number, [])
        detections = build_detections_for_frame(frame_number, frame, detection_rows)
        matches, _unmatched_tracks, unmatched_detections = assign_detections(
            tracks=tracks,
            detections=detections,
            appearance_weight=appearance_weight,
            max_cost=max_cost,
            min_iou=min_iou,
            max_appearance_distance=max_appearance_distance,
        )

        for track_index, detection_index in matches:
            tracks[track_index].update(detections[detection_index], appearance_alpha)

        for detection_index in sorted(unmatched_detections):
            tracks.append(DeepSortLiteTrack(next_track_id, detections[detection_index]))
            next_track_id += 1

        tracks = [
            track for track in tracks
            if track.time_since_update <= max_age
        ]

        for track in sorted(tracks, key=lambda item: item.track_id):
            if track.time_since_update != 0:
                continue
            if track.hits < min_hits:
                continue
            x1, y1, x2, y2 = track.output_box()
            rows.append(
                [
                    frame_number,
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

    capture.release()
    return rows


def write_tracks(path: Path, rows: list[list[float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def track_video(
    video_id: str,
    video: dict[str, Any],
    detections_dir: Path,
    output_dir: Path,
    repo_root: Path,
    args: argparse.Namespace,
) -> int:
    detections_path = detections_dir / f"{video_id}.txt"
    if not detections_path.exists():
        print(f"Skipping {video_id}: missing detections {detections_path}")
        return 0

    detections_by_frame = load_detection_rows(detections_path, args.min_conf)
    rows = run_deepsort_lite(
        video_path=repo_root / video["video_path"],
        detections_by_frame=detections_by_frame,
        appearance_weight=args.appearance_weight,
        appearance_alpha=args.appearance_alpha,
        max_cost=args.max_cost,
        min_iou=args.min_iou,
        max_appearance_distance=args.max_appearance_distance,
        max_age=args.max_age,
        min_hits=args.min_hits,
    )
    output_path = output_dir / f"{video_id}.txt"
    write_tracks(output_path, rows)
    print(f"Wrote {len(rows)} track rows to {output_path}")
    return len(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_metadata = repo_root / "data" / "metadata" / "videos.json"
    default_detections_dir = repo_root / "outputs" / "detections" / "yolo"
    default_output_dir = repo_root / "outputs" / "tracks" / "deepsort"

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=str(default_metadata))
    parser.add_argument("--detections-dir", default=str(default_detections_dir))
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--video-id", action="append", help="Track one video id. Repeatable.")
    parser.add_argument("--min-conf", type=float, default=0.25)
    parser.add_argument("--appearance-weight", type=float, default=0.35)
    parser.add_argument("--appearance-alpha", type=float, default=0.8)
    parser.add_argument("--max-cost", type=float, default=0.75)
    parser.add_argument("--min-iou", type=float, default=0.05)
    parser.add_argument("--max-appearance-distance", type=float, default=0.75)
    parser.add_argument("--max-age", type=int, default=10)
    parser.add_argument("--min-hits", type=int, default=1)
    args = parser.parse_args()

    videos = load_metadata(Path(args.metadata))
    detections_dir = Path(args.detections_dir)
    output_dir = Path(args.output_dir)

    if args.video_id:
        video_ids = args.video_id
    else:
        video_ids = sorted(path.stem for path in detections_dir.glob("*.txt"))

    for video_id in video_ids:
        if video_id not in videos:
            print(f"Skipping unknown video id: {video_id}")
            continue
        track_video(
            video_id=video_id,
            video=videos[video_id],
            detections_dir=detections_dir,
            output_dir=output_dir,
            repo_root=repo_root,
            args=args,
        )


if __name__ == "__main__":
    main()
