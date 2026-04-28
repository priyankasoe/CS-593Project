#!/usr/bin/env python3
"""Draw MOT-style tracks on a video."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    metadata = json.loads(path.read_text())
    return {video["id"]: video for video in metadata["videos"]}


def load_tracks(path: Path) -> dict[int, list[tuple[int, float, float, float, float, float]]]:
    tracks_by_frame: dict[int, list[tuple[int, float, float, float, float, float]]] = defaultdict(list)
    with path.open() as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            frame = int(float(row[0]))
            track_id = int(float(row[1]))
            x, y, w, h = map(float, row[2:6])
            confidence = float(row[6])
            tracks_by_frame[frame].append((track_id, x, y, w, h, confidence))
    return tracks_by_frame


def color_for_id(track_id: int) -> tuple[int, int, int]:
    # BGR colors generated deterministically from the track id.
    return (
        int((37 * track_id) % 255),
        int((17 * track_id + 120) % 255),
        int((97 * track_id + 60) % 255),
    )


def draw_tracks(
    video_path: Path,
    tracks_path: Path,
    output_path: Path,
    max_frames: int | None,
    trail_length: int,
    draw_trails: bool,
) -> None:
    tracks_by_frame = load_tracks(tracks_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=trail_length))
    zero_based_frame = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if max_frames is not None and zero_based_frame >= max_frames:
            break

        frame_number = zero_based_frame + 1
        for track_id, x, y, w, h, confidence in tracks_by_frame.get(frame_number, []):
            color = color_for_id(track_id)
            x1, y1 = int(round(x)), int(round(y))
            x2, y2 = int(round(x + w)), int(round(y + h))
            center = (int(round(x + w / 2)), int(round(y + h / 2)))
            if draw_trails:
                trails[track_id].append(center)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track_id} {confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        if draw_trails:
            for track_id, points in trails.items():
                color = color_for_id(track_id)
                for index in range(1, len(points)):
                    cv2.line(frame, points[index - 1], points[index], color, 2)

        cv2.putText(
            frame,
            f"Frame {frame_number}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        zero_based_frame += 1

    capture.release()
    writer.release()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_metadata = repo_root / "data" / "metadata" / "videos.json"
    default_tracks_dir = repo_root / "outputs" / "tracks" / "centroid"
    default_output_dir = repo_root / "outputs" / "videos" / "centroid"

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=str(default_metadata))
    parser.add_argument("--tracks-dir", default=str(default_tracks_dir))
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--trail-length", type=int, default=30)
    parser.add_argument("--draw-trails", action="store_true")
    parser.add_argument("--tracker-name", default="tracks")
    args = parser.parse_args()

    videos = load_metadata(Path(args.metadata))
    if args.video_id not in videos:
        raise ValueError(f"Unknown video id: {args.video_id}")

    video = videos[args.video_id]
    video_path = repo_root / video["video_path"]
    tracks_path = Path(args.tracks_dir) / f"{args.video_id}.txt"
    output_path = Path(args.output_dir) / f"{args.video_id}_{args.tracker_name}.mp4"

    draw_tracks(
        video_path=video_path,
        tracks_path=tracks_path,
        output_path=output_path,
        max_frames=args.max_frames,
        trail_length=args.trail_length,
        draw_trails=args.draw_trails,
    )
    print(f"Wrote visualization to {output_path}")


if __name__ == "__main__":
    main()
