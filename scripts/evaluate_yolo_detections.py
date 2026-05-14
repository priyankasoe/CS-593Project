#!/usr/bin/env python3
"""Evaluate exported YOLO detections against MOT ground truth.

Reports frame-wise detection precision, recall, F1, and localization quality
using one-to-one IoU matching. This evaluates detection only, not identity.
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
    if union <= 0:
        return 0.0
    return intersection / union


def load_metadata(metadata_path: Path) -> dict[str, dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text())
    return {video["id"]: video for video in metadata["videos"]}


def load_ground_truth(path: Path, frame_range: tuple[int, int] | None) -> dict[int, list[Box]]:
    boxes_by_frame: dict[int, list[Box]] = defaultdict(list)
    with path.open() as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            frame = int(float(row[0]))
            if frame_range and not (frame_range[0] <= frame <= frame_range[1]):
                continue
            x, y, w, h = map(float, row[2:6])
            boxes_by_frame[frame].append(xywh_to_xyxy(x, y, w, h))
    return boxes_by_frame


def load_detections(
    path: Path,
    frame_range: tuple[int, int] | None,
    min_conf: float,
) -> dict[int, list[tuple[Box, float]]]:
    boxes_by_frame: dict[int, list[tuple[Box, float]]] = defaultdict(list)
    with path.open() as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            frame = int(float(row[0]))
            if frame_range and not (frame_range[0] <= frame <= frame_range[1]):
                continue
            confidence = float(row[5])
            if confidence < min_conf:
                continue
            x1, y1, x2, y2 = map(float, row[1:5])
            boxes_by_frame[frame].append(((x1, y1, x2, y2), confidence))
    return boxes_by_frame


def mot_frame_range(video: dict[str, Any]) -> tuple[int, int] | None:
    if video.get("annotation_scope") != "partial_video":
        return None
    frame_range = video["annotated_frame_range"]
    return int(frame_range["mot_start"]), int(frame_range["mot_end"])


def evaluate_video(
    gt_by_frame: dict[int, list[Box]],
    det_by_frame: dict[int, list[tuple[Box, float]]],
    iou_threshold: float,
) -> dict[str, float]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    matched_ious: list[float] = []

    # extract frames 
    frames = sorted(set(gt_by_frame) | set(det_by_frame))
    for frame in frames:

         # set of bounding boxes of ground truth vs. detections

        gt_boxes = gt_by_frame.get(frame, [])
        detections = det_by_frame.get(frame, [])
        det_boxes = [box for box, _confidence in detections]

        if not gt_boxes:   # no boxes in gt but boxes in detected -> false positive for all boxes detected 
            false_positive += len(det_boxes)
            continue
        if not det_boxes:   # boxes in gt but no boxes in detected -> false negative for all boxes not detected 
            false_negative += len(gt_boxes)
            continue

        # compute cost matrix (iou of all pairs of possible box matches)
        cost_matrix = [
            [-iou(gt_box, det_box) for det_box in det_boxes]
            for gt_box in gt_boxes
        ]

        # hungarian assignment 

        # solve system 
        gt_indices, det_indices = linear_sum_assignment(cost_matrix)

        matched_gt = set()
        matched_det = set()
        for gt_index, det_index in zip(gt_indices, det_indices):

            # get detections 
            overlap = -cost_matrix[gt_index][det_index]

            # detection if greater than threshold of 0.5 
            if overlap >= iou_threshold:
                true_positive += 1
                matched_ious.append(overlap)
                matched_gt.add(gt_index)
                matched_det.add(det_index)

        # calculate reminder 
        # will always be postitive or 0 since there can't be more matched than boxes (we only match boxes) 

        # false negative is when not enough boxes were detected (more gt boxes than matched)
        # false positive is when too many boxes were detected (more detected than matched ) 

        false_negative += len(gt_boxes) - len(matched_gt)
        false_positive += len(det_boxes) - len(matched_det)

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
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "frames": len(frames),
    }


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_metadata = repo_root / "data" / "metadata" / "videos.json"
    default_detection_dir = repo_root / "outputs" / "detections" / "yolo"
    default_output = repo_root / "outputs" / "metrics" / "yolo_detection_metrics.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=str(default_metadata))
    parser.add_argument("--detections-dir", default=str(default_detection_dir))
    parser.add_argument("--video-id", action="append", help="Evaluate one video id. Repeatable.")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-conf", type=float, default=0.0)
    parser.add_argument("--output", default=str(default_output))
    args = parser.parse_args()

    videos = load_metadata(Path(args.metadata))
    detection_dir = Path(args.detections_dir)

    if args.video_id:
        video_ids = args.video_id
    else:
        video_ids = sorted(path.stem for path in detection_dir.glob("*.txt"))

    rows: list[dict[str, float | str]] = []
    totals = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "weighted_iou_sum": 0.0,
        "matched": 0,
    }

    print(
        "video_id,frames,tp,fp,fn,precision,recall,f1,mean_iou"
    )
    for video_id in video_ids:
        if video_id not in videos:
            print(f"Skipping unknown video id: {video_id}")
            continue

        video = videos[video_id]
        gt_path = repo_root / video["annotations"]["mot_gt"]
        det_path = detection_dir / f"{video_id}.txt"
        if not det_path.exists():
            print(f"Skipping {video_id}: missing detections {det_path}")
            continue

        frame_range = mot_frame_range(video)
        gt_by_frame = load_ground_truth(gt_path, frame_range)
        det_by_frame = load_detections(det_path, frame_range, args.min_conf)
        metrics = evaluate_video(gt_by_frame, det_by_frame, args.iou_threshold)

        print(
            ",".join(
                [
                    video_id,
                    str(int(metrics["frames"])),
                    str(int(metrics["tp"])),
                    str(int(metrics["fp"])),
                    str(int(metrics["fn"])),
                    format_metric(metrics["precision"]),
                    format_metric(metrics["recall"]),
                    format_metric(metrics["f1"]),
                    format_metric(metrics["mean_iou"]),
                ]
            )
        )

        rows.append({"video_id": video_id, **metrics})
        totals["tp"] += int(metrics["tp"])
        totals["fp"] += int(metrics["fp"])
        totals["fn"] += int(metrics["fn"])
        totals["weighted_iou_sum"] += metrics["mean_iou"] * metrics["tp"]
        totals["matched"] += int(metrics["tp"])

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
    total_mean_iou = (
        totals["weighted_iou_sum"] / totals["matched"]
        if totals["matched"]
        else 0.0
    )

    print(
        ",".join(
            [
                "TOTAL",
                "",
                str(totals["tp"]),
                str(totals["fp"]),
                str(totals["fn"]),
                format_metric(total_precision),
                format_metric(total_recall),
                format_metric(total_f1),
                format_metric(total_mean_iou),
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
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "mean_iou",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(
            {
                "video_id": "TOTAL",
                "frames": "",
                "tp": totals["tp"],
                "fp": totals["fp"],
                "fn": totals["fn"],
                "precision": total_precision,
                "recall": total_recall,
                "f1": total_f1,
                "mean_iou": total_mean_iou,
            }
        )
    print(f"Wrote metrics to {output_path}")


if __name__ == "__main__":
    main()
