#!/usr/bin/env python3
"""Run YOLO prediction with outputs anchored in this repository."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_model = repo_root / "runs" / "detect" / "train-2" / "weights" / "best.pt"
    default_project = repo_root / "runs" / "detect"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(default_model))
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", default="320")
    parser.add_argument("--conf", default="0.25")
    parser.add_argument("--name", default="predict")
    args = parser.parse_args()

    command = [
        "yolo",
        "detect",
        "predict",
        f"model={args.model}",
        f"source={args.source}",
        f"imgsz={args.imgsz}",
        f"conf={args.conf}",
        "save=True",
        f"project={default_project}",
        f"name={args.name}",
    ]

    print("Running:", " ".join(command))
    subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
