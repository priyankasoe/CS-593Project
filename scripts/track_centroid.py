#!/usr/bin/env python3
"""Track YOLO detections with a centroid-distance baseline.

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
    def centroid(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1


@dataclass
class Track:
    track_id: int
    detection: Detection
    missing: int = 0

    @property
    def centroid(self) -> tuple[float, float]:
        return self.detection.centroid


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


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def assign_detections(
    tracks: dict[int, Track],
    detections: list[Detection],
    max_distance: float,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    if not tracks or not detections:
        return [], set(tracks), set(range(len(detections)))

    track_ids = list(tracks)
    cost_matrix = [
        [
            distance(tracks[track_id].centroid, detection.centroid)
            for detection in detections
        ]
        for track_id in track_ids
    ]

    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()

    for row_index, col_index in zip(row_indices, col_indices):
        track_id = track_ids[row_index]
        if cost_matrix[row_index][col_index] <= max_distance:
            matches.append((track_id, col_index))
            matched_tracks.add(track_id)
            matched_detections.add(col_index)

    unmatched_tracks = set(track_ids) - matched_tracks
    unmatched_detections = set(range(len(detections))) - matched_detections
    return matches, unmatched_tracks, unmatched_detections


def run_tracker(
    detections_by_frame: dict[int, list[Detection]],
    max_distance: float,
    max_missing: int,
) -> list[list[float | int]]:
    active_tracks: dict[int, Track] = {}
    next_track_id = 1
    output_rows: list[list[float | int]] = []

    if not detections_by_frame:
        return output_rows

    first_frame = min(detections_by_frame)
    last_frame = max(detections_by_frame)

    for frame in range(first_frame, last_frame + 1):
        detections = detections_by_frame.get(frame, [])
        matches, unmatched_tracks, unmatched_detections = assign_detections(
            active_tracks,
            detections,
            max_distance,
        )

        for track_id, detection_index in matches:
            active_tracks[track_id].detection = detections[detection_index]
            active_tracks[track_id].missing = 0

        for track_id in unmatched_tracks:
            active_tracks[track_id].missing += 1

        for detection_index in sorted(unmatched_detections):
            active_tracks[next_track_id] = Track(
                track_id=next_track_id,
                detection=detections[detection_index],
            )
            next_track_id += 1

        stale_tracks = [
            track_id
            for track_id, track in active_tracks.items()
            if track.missing > max_missing
        ]
        for track_id in stale_tracks:
            del active_tracks[track_id]

        for track_id, track in sorted(active_tracks.items()):
            if track.missing != 0 or track.detection.frame != frame:
                continue
            x, y, width, height = track.detection.xywh
            output_rows.append(
                [
                    frame,
                    track_id,
                    round(x, 3),
                    round(y, 3),
                    round(width, 3),
                    round(height, 3),
                    round(track.detection.confidence, 6),
                    track.detection.class_id,
                    -1,
                ]
            )

    return output_rows


def write_tracks(path: Path, rows: list[list[float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def track_file(
    detections_path: Path,
    output_path: Path,
    min_conf: float,
    max_distance: float,
    max_missing: int,
) -> int:
    detections_by_frame = load_detections(detections_path, min_conf)
    rows = run_tracker(detections_by_frame, max_distance, max_missing)
    write_tracks(output_path, rows)
    return len(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_detections_dir = repo_root / "outputs" / "detections" / "yolo"
    default_output_dir = repo_root / "outputs" / "tracks" / "centroid"

    parser = argparse.ArgumentParser()
    parser.add_argument("--detections-dir", default=str(default_detections_dir))
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--video-id", action="append", help="Track one video id. Repeatable.")
    parser.add_argument("--min-conf", type=float, default=0.25)
    parser.add_argument("--max-distance", type=float, default=80.0)
    parser.add_argument("--max-missing", type=int, default=10)
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
            max_distance=args.max_distance,
            max_missing=args.max_missing,
        )
        print(f"Wrote {rows} track rows to {output_path}")


if __name__ == "__main__":
    main()
