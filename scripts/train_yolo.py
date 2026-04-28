#!/usr/bin/env python3
"""Train YOLO with outputs anchored in this repository."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_data = repo_root / "data" / "yolo_dataset" / "dataset.yaml"
    default_project = repo_root / "runs" / "detect"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(default_data))
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", default="5")
    parser.add_argument("--imgsz", default="320")
    parser.add_argument("--batch", default="4")
    parser.add_argument("--workers", default="0")
    parser.add_argument("--name", default="train")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    args = parser.parse_args()

    command = [
        "yolo",
        "detect",
        "train",
        f"model={args.model}",
    ]
    if args.resume:
        command.append("resume=True")
    else:
        command.extend(
            [
                f"data={args.data}",
                f"epochs={args.epochs}",
                f"imgsz={args.imgsz}",
                f"batch={args.batch}",
                f"workers={args.workers}",
                f"project={default_project}",
                f"name={args.name}",
                f"exist_ok={str(args.exist_ok)}",
            ]
        )

    print("Running:", " ".join(command))
    subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
