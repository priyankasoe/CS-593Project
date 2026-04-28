#!/usr/bin/env python3
"""Track YOLO detections with the deep-sort-realtime package.

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
from pathlib import Path
from typing import Any

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort


DetectionRow = tuple[float, float, float, float, float, int]


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    metadata = json.loads(path.read_text())
    return {video["id"]: video for video in metadata["videos"]}


def load_detection_rows(path: Path, min_conf: float) -> dict[int, list[DetectionRow]]:
    detections_by_frame: dict[int, list[DetectionRow]] = defaultdict(list)
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


def to_deepsort_detections(rows: list[DetectionRow]) -> list[tuple[list[float], float, int]]:
    detections: list[tuple[list[float], float, int]] = []
    for x1, y1, x2, y2, confidence, class_id in rows:
        detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id))
    return detections


def run_deepsort_realtime(
    video_path: Path,
    detections_by_frame: dict[int, list[DetectionRow]],
    max_age: int,
    n_init: int,
    max_iou_distance: float,
    max_cosine_distance: float,
    embedder: str,
    embedder_gpu: bool,
) -> list[list[float | int | str]]:
    if not detections_by_frame:
        return []

    tracker = DeepSort(
        max_iou_distance=max_iou_distance,
        max_age=max_age,
        n_init=n_init,
        max_cosine_distance=max_cosine_distance,
        embedder=embedder,
        embedder_gpu=embedder_gpu,
        bgr=True,
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    first_frame = min(detections_by_frame)
    last_frame = max(detections_by_frame)
    rows: list[list[float | int | str]] = []
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

        detections = to_deepsort_detections(detections_by_frame.get(frame_number, []))
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            if track.time_since_update != 0:
                continue
            x1, y1, x2, y2 = track.to_ltrb()
            confidence = getattr(track, "det_conf", None)
            if confidence is None:
                confidence = 1.0
            class_id = getattr(track, "det_class", None)
            if class_id is None:
                class_id = 0
            rows.append(
                [
                    frame_number,
                    track.track_id,
                    round(float(x1), 3),
                    round(float(y1), 3),
                    round(float(x2 - x1), 3),
                    round(float(y2 - y1), 3),
                    round(float(confidence), 6),
                    int(class_id),
                    -1,
                ]
            )

    capture.release()
    return rows


def write_tracks(path: Path, rows: list[list[float | int | str]]) -> None:
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
    rows = run_deepsort_realtime(
        video_path=repo_root / video["video_path"],
        detections_by_frame=detections_by_frame,
        max_age=args.max_age,
        n_init=args.n_init,
        max_iou_distance=args.max_iou_distance,
        max_cosine_distance=args.max_cosine_distance,
        embedder=args.embedder,
        embedder_gpu=args.embedder_gpu,
    )
    output_path = output_dir / f"{video_id}.txt"
    write_tracks(output_path, rows)
    print(f"Wrote {len(rows)} track rows to {output_path}")
    return len(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_metadata = repo_root / "data" / "metadata" / "videos.json"
    default_detections_dir = repo_root / "outputs" / "detections" / "yolo_conf035"
    default_output_dir = repo_root / "outputs" / "tracks" / "deepsort_realtime_conf035"

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=str(default_metadata))
    parser.add_argument("--detections-dir", default=str(default_detections_dir))
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--video-id", action="append", help="Track one video id. Repeatable.")
    parser.add_argument("--min-conf", type=float, default=0.0)
    parser.add_argument("--max-age", type=int, default=10)
    parser.add_argument("--n-init", type=int, default=1)
    parser.add_argument("--max-iou-distance", type=float, default=0.7)
    parser.add_argument("--max-cosine-distance", type=float, default=0.4)
    parser.add_argument("--embedder", default="mobilenet")
    parser.add_argument("--embedder-gpu", action="store_true")
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
