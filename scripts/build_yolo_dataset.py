#!/usr/bin/env python3
"""Build a YOLO image/label dataset from annotated videos.

The script reads data/metadata/videos.json, extracts the labeled frame range
from each video with ffmpeg, and copies matching CVAT YOLO labels into the
dataset split folders with video-prefixed filenames.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def frame_range(video: dict) -> tuple[int, int]:
    if video.get("annotation_scope") == "partial_video":
        annotated = video["annotated_frame_range"]
        return int(annotated["yolo_start"]), int(annotated["yolo_end"])
    return 0, int(video["frame_count"]) - 1


def extract_frames(video: dict, output_dir: Path, start: int, end: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / f"{video['id']}_frame_%06d.jpg"
    frame_count = end - start + 1

    if start == 0:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            video["video_path"],
            "-frames:v",
            str(frame_count),
            "-start_number",
            "0",
            str(pattern),
        ]
    else:
        # Present for completeness; current metadata uses yolo_start = 0.
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            video["video_path"],
            "-vf",
            f"select='between(n,{start},{end})'",
            "-vsync",
            "0",
            "-start_number",
            str(start),
            str(pattern),
        ]

    subprocess.run(command, check=True)


def copy_labels(video: dict, output_dir: Path, start: int, end: int) -> None:
    source_dir = Path(video["annotations"]["yolo_labels"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in range(start, end + 1):
        source = source_dir / f"frame_{frame_idx:06d}.txt"
        target = output_dir / f"{video['id']}_frame_{frame_idx:06d}.txt"
        if not source.exists():
            raise FileNotFoundError(f"Missing label file: {source}")
        shutil.copy2(source, target)


def write_dataset_yaml(dataset_dir: Path) -> None:
    dataset_yaml = dataset_dir / "dataset.yaml"
    dataset_path = dataset_dir.resolve()
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_path}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "names:",
                "  0: robot",
                "",
            ]
        )
    )


def remove_existing_video_files(dataset_dir: Path, video_id: str) -> None:
    for split in ["train", "val", "test"]:
        for folder, suffix in [("images", ".jpg"), ("labels", ".txt")]:
            split_dir = dataset_dir / folder / split
            for path in split_dir.glob(f"{video_id}_frame_*{suffix}"):
                path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/metadata/videos.json")
    parser.add_argument("--dataset-dir", default="data/yolo_dataset")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not re-extract a video if all expected output frames exist.",
    )
    args = parser.parse_args()

    metadata = json.loads(Path(args.metadata).read_text())
    dataset_dir = Path(args.dataset_dir)

    for split in ["train", "val", "test"]:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for video in metadata["videos"]:
        split = video["dataset_split"]
        if split not in {"train", "val", "test"}:
            continue

        start, end = frame_range(video)
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        expected_first = image_dir / f"{video['id']}_frame_{start:06d}.jpg"
        expected_last = image_dir / f"{video['id']}_frame_{end:06d}.jpg"

        print(
            f"{video['id']}: split={split}, frames={start}-{end}, "
            f"count={end - start + 1}"
        )
        use_existing = (
            args.skip_existing and expected_first.exists() and expected_last.exists()
        )
        if not use_existing:
            remove_existing_video_files(dataset_dir, video["id"])
            extract_frames(video, image_dir, start, end)
        copy_labels(video, label_dir, start, end)

    write_dataset_yaml(dataset_dir)


if __name__ == "__main__":
    main()
