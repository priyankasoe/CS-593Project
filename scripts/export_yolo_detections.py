#!/usr/bin/env python3
"""Export YOLO detections to a tracker-friendly text file.

Output rows use:
frame,x1,y1,x2,y2,confidence,class

Frames are 1-based by default so they line up with MOT ground-truth files.
When a metadata video has annotation_scope = partial_video, --video-id exports
only the annotated YOLO frame range unless --full-video is set.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_video(metadata_path: Path, video_id: str) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text())
    for video in metadata["videos"]:
        if video["id"] == video_id:
            return video
    raise ValueError(f"Unknown video id: {video_id}")


def default_output_path(repo_root: Path, source: Path, video_id: str | None) -> Path:
    name = video_id or source.stem
    return repo_root / "outputs" / "detections" / "yolo" / f"{name}.txt"


def frame_limits(video: dict[str, Any] | None, full_video: bool) -> tuple[int, int] | None:
    if video is None or full_video or video.get("annotation_scope") != "partial_video":
        return None

    frame_range = video["annotated_frame_range"]
    return int(frame_range["yolo_start"]), int(frame_range["yolo_end"])


def export_detections(
    model_path: Path,
    source: Path,
    output_path: Path,
    imgsz: int,
    conf: float,
    frame_base: int,
    limits: tuple[int, int] | None,
    vid_stride: int,
) -> int:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    first_frame = limits[0] if limits else 0
    last_frame = limits[1] if limits else None

    results = model.predict(
        source=str(source),
        imgsz=imgsz,
        conf=conf,
        stream=True,
        verbose=False,
        vid_stride=vid_stride,
    )

    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        for result_index, result in enumerate(results):
            zero_based_frame = result_index * vid_stride
            if zero_based_frame < first_frame:
                continue
            if last_frame is not None and zero_based_frame > last_frame:
                break

            frame_number = zero_based_frame + frame_base
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                writer.writerow(
                    [
                        frame_number,
                        f"{x1:.3f}",
                        f"{y1:.3f}",
                        f"{x2:.3f}",
                        f"{y2:.3f}",
                        f"{confidence:.6f}",
                        class_id,
                    ]
                )
                rows_written += 1

    return rows_written


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_model = repo_root / "runs" / "detect" / "train-2" / "weights" / "best.pt"
    default_metadata = repo_root / "data" / "metadata" / "videos.json"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(default_model))
    parser.add_argument("--metadata", default=str(default_metadata))
    parser.add_argument("--video-id", help="Video id from data/metadata/videos.json")
    parser.add_argument("--source", help="Direct video/image source path")
    parser.add_argument("--output", help="Output detections .txt path")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--frame-base", type=int, choices=[0, 1], default=1)
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument(
        "--full-video",
        action="store_true",
        help="Ignore partial annotation frame limits when using --video-id.",
    )
    args = parser.parse_args()

    if not args.video_id and not args.source:
        parser.error("Provide either --video-id or --source.")

    video = None
    if args.video_id:
        video = load_video(Path(args.metadata), args.video_id)
        source = repo_root / video["video_path"]
    else:
        source = Path(args.source)
        if not source.is_absolute():
            source = repo_root / source

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = repo_root / model_path

    output_path = Path(args.output) if args.output else default_output_path(
        repo_root, source, args.video_id
    )
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    limits = frame_limits(video, args.full_video)
    rows = export_detections(
        model_path=model_path,
        source=source,
        output_path=output_path,
        imgsz=args.imgsz,
        conf=args.conf,
        frame_base=args.frame_base,
        limits=limits,
        vid_stride=args.vid_stride,
    )

    limit_text = ""
    if limits:
        limit_text = f" from YOLO frames {limits[0]}-{limits[1]}"
    print(f"Wrote {rows} detections{limit_text} to {output_path}")


if __name__ == "__main__":
    main()
