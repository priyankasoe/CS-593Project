#!/usr/bin/env python3
"""Evaluate MOT-style tracker outputs against MOT ground truth.

This computes a practical subset of MOT metrics:
- MOTA = 1 - (FN + FP + IDSW) / GT
- MOTP as mean IoU over matched boxes
- ID switches
- precision, recall, F1

Tracker rows are expected as:
frame,id,x,y,w,h,confidence,class,visibility
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.optimize import linear_sum_assignment


Box = tuple[float, float, float, float]
ObjectBox = tuple[int, Box]


def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Box:
    return x, y, x + w, y + h


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


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    metadata = json.loads(path.read_text())
    return {video["id"]: video for video in metadata["videos"]}


def mot_frame_range(video: dict[str, Any]) -> tuple[int, int] | None:
    if video.get("annotation_scope") != "partial_video":
        return None
    frame_range = video["annotated_frame_range"]
    return int(frame_range["mot_start"]), int(frame_range["mot_end"])


def load_mot_boxes(path: Path, frame_range: tuple[int, int] | None) -> dict[int, list[ObjectBox]]:
    boxes_by_frame: dict[int, list[ObjectBox]] = defaultdict(list)
    with path.open() as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            frame = int(float(row[0]))
            if frame_range and not (frame_range[0] <= frame <= frame_range[1]):
                continue
            object_id = int(float(row[1]))
            x, y, w, h = map(float, row[2:6])
            boxes_by_frame[frame].append((object_id, xywh_to_xyxy(x, y, w, h)))
    return boxes_by_frame


def evaluate_video(
    gt_by_frame: dict[int, list[ObjectBox]],
    pred_by_frame: dict[int, list[ObjectBox]],
    iou_threshold: float,
) -> dict[str, float]:
    false_positive = 0
    false_negative = 0
    true_positive = 0
    id_switches = 0
    ious: list[float] = []
    gt_count = 0

    # dict to store gt_ idx, prev index it was assigned to 
    previous_matches: dict[int, int] = {}

    # same as detection, get frames 
    frames = sorted(set(gt_by_frame) | set(pred_by_frame))

    for frame in frames:

        # get objects gt and detected per frame 
        gt_objects = gt_by_frame.get(frame, [])
        pred_objects = pred_by_frame.get(frame, [])
        gt_count += len(gt_objects)

        # handle no detections in both cases 
        if not gt_objects:
            false_positive += len(pred_objects)
            continue
        if not pred_objects:
            false_negative += len(gt_objects)
            continue

        # compute cost matrix to determine matches and solve 
        cost_matrix = [
            [-iou(gt_box, pred_box) for _pred_id, pred_box in pred_objects]
            for _gt_id, gt_box in gt_objects
        ]
        gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

        matched_gt_indices: set[int] = set()
        matched_pred_indices: set[int] = set()
        current_matches: dict[int, int] = {}

        for gt_index, pred_index in zip(gt_indices, pred_indices):
            overlap = -cost_matrix[gt_index][pred_index]

            # stage 1: check iou threshold 
            if overlap < iou_threshold:
                continue

            # now we have matched, but have to check if id lines up 
            # this still counts as a match 

            # stage 2: check id 
            gt_id = gt_objects[gt_index][0]
            pred_id = pred_objects[pred_index][0]
            previous_pred_id = previous_matches.get(gt_id)
            if previous_pred_id is not None and previous_pred_id != pred_id:
                id_switches += 1

            true_positive += 1
            ious.append(overlap)
            matched_gt_indices.add(gt_index)
            matched_pred_indices.add(pred_index)
            current_matches[gt_id] = pred_id

        false_negative += len(gt_objects) - len(matched_gt_indices)
        false_positive += len(pred_objects) - len(matched_pred_indices)

        # update previous matches 
        for gt_id, pred_id in current_matches.items():
            previous_matches[gt_id] = pred_id

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    mota = (
        1.0 - (false_negative + false_positive + id_switches) / gt_count
        if gt_count
        else 0.0
    )
    motp_iou = sum(ious) / len(ious) if ious else 0.0

    return {
        "frames": len(frames),
        "gt": gt_count,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "id_switches": id_switches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mota": mota,
        "motp_iou": motp_iou,
    }


def format_float(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_metadata = repo_root / "data" / "metadata" / "videos.json"
    default_tracks_dir = repo_root / "outputs" / "tracks" / "centroid"
    default_output = repo_root / "outputs" / "metrics" / "centroid_tracking_metrics.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=str(default_metadata))
    parser.add_argument("--tracks-dir", default=str(default_tracks_dir))
    parser.add_argument("--video-id", action="append", help="Evaluate one video id. Repeatable.")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", default=str(default_output))
    args = parser.parse_args()

    videos = load_metadata(Path(args.metadata))
    tracks_dir = Path(args.tracks_dir)
    video_ids = args.video_id or sorted(path.stem for path in tracks_dir.glob("*.txt"))

    rows: list[dict[str, float | int | str]] = []
    totals = {
        "gt": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "id_switches": 0,
        "weighted_iou_sum": 0.0,
    }

    print("video_id,frames,gt,tp,fp,fn,id_switches,precision,recall,f1,mota,motp_iou")
    for video_id in video_ids:
        if video_id not in videos:
            print(f"Skipping unknown video id: {video_id}")
            continue

        video = videos[video_id]
        frame_range = mot_frame_range(video)
        gt_path = repo_root / video["annotations"]["mot_gt"]
        track_path = tracks_dir / f"{video_id}.txt"
        if not track_path.exists():
            print(f"Skipping {video_id}: missing tracks {track_path}")
            continue

        gt_by_frame = load_mot_boxes(gt_path, frame_range)
        pred_by_frame = load_mot_boxes(track_path, frame_range)
        metrics = evaluate_video(gt_by_frame, pred_by_frame, args.iou_threshold)
        rows.append({"video_id": video_id, **metrics})

        totals["gt"] += int(metrics["gt"])
        totals["tp"] += int(metrics["tp"])
        totals["fp"] += int(metrics["fp"])
        totals["fn"] += int(metrics["fn"])
        totals["id_switches"] += int(metrics["id_switches"])
        totals["weighted_iou_sum"] += metrics["motp_iou"] * metrics["tp"]

        print(
            ",".join(
                [
                    video_id,
                    str(int(metrics["frames"])),
                    str(int(metrics["gt"])),
                    str(int(metrics["tp"])),
                    str(int(metrics["fp"])),
                    str(int(metrics["fn"])),
                    str(int(metrics["id_switches"])),
                    format_float(metrics["precision"]),
                    format_float(metrics["recall"]),
                    format_float(metrics["f1"]),
                    format_float(metrics["mota"]),
                    format_float(metrics["motp_iou"]),
                ]
            )
        )

    total_precision = (
        totals["tp"] / (totals["tp"] + totals["fp"])
        if totals["tp"] + totals["fp"]
        else 0.0
    )
    total_recall = (
        totals["tp"] / (totals["tp"] + totals["fn"])
        if totals["tp"] + totals["fn"]
        else 0.0
    )
    total_f1 = (
        2 * total_precision * total_recall / (total_precision + total_recall)
        if total_precision + total_recall
        else 0.0
    )
    total_mota = (
        1.0 - (totals["fn"] + totals["fp"] + totals["id_switches"]) / totals["gt"]
        if totals["gt"]
        else 0.0
    )
    total_motp = (
        totals["weighted_iou_sum"] / totals["tp"] if totals["tp"] else 0.0
    )

    print(
        ",".join(
            [
                "TOTAL",
                "",
                str(totals["gt"]),
                str(totals["tp"]),
                str(totals["fp"]),
                str(totals["fn"]),
                str(totals["id_switches"]),
                format_float(total_precision),
                format_float(total_recall),
                format_float(total_f1),
                format_float(total_mota),
                format_float(total_motp),
            ]
        )
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        fieldnames = [
            "video_id",
            "frames",
            "gt",
            "tp",
            "fp",
            "fn",
            "id_switches",
            "precision",
            "recall",
            "f1",
            "mota",
            "motp_iou",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(
            {
                "video_id": "TOTAL",
                "frames": "",
                "gt": totals["gt"],
                "tp": totals["tp"],
                "fp": totals["fp"],
                "fn": totals["fn"],
                "id_switches": totals["id_switches"],
                "precision": total_precision,
                "recall": total_recall,
                "f1": total_f1,
                "mota": total_mota,
                "motp_iou": total_motp,
            }
        )
    print(f"Wrote metrics to {output_path}")


if __name__ == "__main__":
    main()
