# CS-593 Project Handoff Context

This file summarizes the relevant project context and decisions from the current chat so another assistant/chat can continue without needing the original conversation.

## Project Goal

Project proposal title: **Robust Multi-Object Tracking for Identical Swarm Robots**.

Goal: build a vision-based system for detecting and tracking multiple visually similar Sphero BOLT robots under occlusion, motion blur, lighting variation, and increasing robot counts.

Planned comparison:
- Marker-based detection: AprilTags when markers are visible.
- Learning-based detection: YOLO trained/fine-tuned to detect Sphero robots.
- Tracking: centroid tracker baseline, SORT, and DeepSORT.
- Evaluation: detection precision/recall and tracking metrics such as MOTA, MOTP, ID switches, IDF1.

The user is currently prioritizing **Part 4B: learning-based detection** because marker videos are limited.

## Current Repository

Repo path:

```text
/Users/prisoe/projects/CS-593Project
```

Important files:

```text
data/metadata/videos.json
data/yolo_dataset/dataset.yaml
scripts/build_yolo_dataset.py
scripts/train_yolo.py
scripts/predict_yolo.py
scripts/export_yolo_detections.py
scripts/evaluate_yolo_detections.py
scripts/track_centroid.py
scripts/track_sort.py
scripts/track_deepsort.py
scripts/evaluate_tracks.py
scripts/visualize_tracks.py
runs/detect/train-2/weights/best.pt
```

## Annotation Formats

The user exported annotations from CVAT in two formats:

- **MOT 1.1** for tracking ground truth and identity evaluation.
- **YOLO** for detector training.

MOT files are used for tracking evaluation:

```text
data/annotations/mot/<video>MOT/gt.txt
data/annotations/mot/<video>MOT/labels.txt
```

YOLO labels are used for detector training:

```text
data/annotations/yolo/<video>YOLO/obj_train_data/frame_000000.txt
```

Images were not exported from CVAT. Frames are extracted locally from the video files.

## Dataset Metadata

Metadata is stored in:

```text
data/metadata/videos.json
```

It includes video paths, annotation paths, robot count, scenario, lighting, occlusion level, split, and annotation scope.

The current annotated videos are:

```text
random_sim_3bot
straight_line_3botSLOW
straight_line_6bots
straight_line_2move_1still
random_sim_5bot_dark
random_sim_6bot_naturallight_occlusion1
random_sim_6bot_naturallight_occlusion2
random_sim_6bot_naturallight_occlusion3
random_sim_6bot_naturallight_occlusion4
straight_line_6bot_dark
```

Current split:

```text
train:
  random_sim_3bot
  straight_line_3botSLOW
  straight_line_2move_1still
  random_sim_6bot_naturallight_occlusion1
  random_sim_6bot_naturallight_occlusion2
  straight_line_6bot_dark

val:
  straight_line_6bots
  random_sim_6bot_naturallight_occlusion4

test:
  random_sim_5bot_dark
  random_sim_6bot_naturallight_occlusion3
```

Current generated YOLO dataset counts:

```text
train: 5349 images / labels
val:    446 images / labels
test:  1970 images / labels
```

All generated images have matching labels.

## Important Annotation Scope Caveat

Most annotated videos are full-video annotations:

```json
"annotation_scope": "full_video"
```

Exception:

```text
random_sim_5bot_dark
```

This video is a partial annotation. The user annotated only the beginning portion.

Use:

```json
"annotation_scope": "partial_video",
"annotated_frame_range": {
  "yolo_start": 0,
  "yolo_end": 1670,
  "mot_start": 1,
  "mot_end": 1671
}
```

YOLO export for `random_sim_5bot_dark` contains empty label files after `frame_001670.txt`. These should **not** be used for training because robots are still visible later, and empty labels would teach YOLO that those robot frames contain no objects.

## Natural-Light Occlusion Videos

The user added four natural-light occlusion videos:

```text
random_sim_6bot_naturallight_occlusion1
random_sim_6bot_naturallight_occlusion2
random_sim_6bot_naturallight_occlusion3
random_sim_6bot_naturallight_occlusion4
```

These have documented occlusions and full obstruction events. The user included IDs in the MOT annotations.

Split decision:

```text
occlusion1 -> train
occlusion2 -> train
occlusion3 -> test
occlusion4 -> val
```

Video metadata:

```text
occlusion1: 300 frames, 1524x1144, 30 fps
occlusion2: 307 frames, 1524x1144, 30 fps
occlusion3: 299 frames, 1524x1144, 30 fps
occlusion4: 303 frames, 1524x1144, 30 fps
```

Note: `random_sim_6bot_naturallight_occlusion2MOT` currently has 12 track IDs for a 6-robot video. That is okay for YOLO detection training, but for tracking evaluation later, verify whether IDs are intended to continue through full obstruction or restart after disappearance.

## Dark Straight-Line Training Video

The user added annotations for:

```text
straight_line_6bot_dark
```

It is in the training split. Metadata:

```text
straight_line_6bot_dark: 381 frames, 1440x1920, 30 fps, dark lighting
```

Important caveat: the video/folder name says `6bot`, but the current MOT annotation contains 5 IDs and 1905 boxes over 381 frames. Metadata uses `robot_count: 5` and notes the mismatch.

## MOV vs MP4 Annotation Issue

The user accidentally annotated a `.mov` for the first natural-light occlusion video but has a matching `.mp4`.

The MOV and MP4 were compared:

```text
MOV:
width=1524
height=1144
avg_frame_rate=180000/6001
duration=10.001667
nb_frames=300

MP4:
width=1524
height=1144
avg_frame_rate=30/1
duration=10.000000
nb_frames=300
```

Decision: okay to use the `.mp4` because width, height, duration, and frame count match. The frame-rate difference is negligible and both have 300 frames.

## YOLO Dataset Builder

Script:

```text
scripts/build_yolo_dataset.py
```

Purpose:
- Reads `data/metadata/videos.json`.
- Extracts labeled frames from videos using `ffmpeg`.
- Copies matching YOLO labels.
- Adds video ID prefixes to frame/label filenames to avoid collisions.
- Writes `data/yolo_dataset/dataset.yaml`.

Generated structure:

```text
data/yolo_dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  dataset.yaml
```

The generated `dataset.yaml` uses an absolute dataset path so YOLO can find it from any working directory:

```yaml
path: /Users/prisoe/projects/CS-593Project/data/yolo_dataset
train: images/train
val: images/val
test: images/test

names:
  0: robot
```

To rebuild:

```bash
cd /Users/prisoe/projects/CS-593Project
python3 scripts/build_yolo_dataset.py
```

## YOLO Runs Location Fix

The user accidentally ran YOLO from another repo path, so training outputs were written into:

```text
sphero-swarm-CS593Project
```

The user manually moved the `runs/` folder back into the current repo.

To prevent future outputs from landing elsewhere, wrapper scripts were added:

```text
scripts/train_yolo.py
scripts/predict_yolo.py
```

These anchor outputs under:

```text
/Users/prisoe/projects/CS-593Project/runs/detect
```

## Current Model / Next Training

Existing trained model:

```text
runs/detect/train-2/weights/best.pt
```

This came from a short YOLO training run. It is usable as a starting point.

The user asked what fine-tuning means. Explanation:

- `model=yolov8n.pt` starts from generic pretrained YOLO.
- `model=runs/detect/train-2/weights/best.pt` starts from the already-trained Sphero detector and continues training on the expanded dataset.

Recommendation: fine-tune from the existing Sphero model because the user has limited time.

The user plans to train a 25-epoch model after adding `straight_line_6bot_dark` and `occlusion4`.

## Recommended Retraining Command

From repo root:

```bash
cd /Users/prisoe/projects/CS-593Project
./env/bin/python scripts/train_yolo.py \
  --model runs/detect/train_occlusion_finetune/weights/best.pt \
  --epochs 25 \
  --imgsz 320 \
  --batch 4 \
  --workers 0 \
  --name train_occlusion_25epoch
```

If `runs/detect/train_occlusion_finetune/weights/best.pt` does not exist, use:

```bash
./env/bin/python scripts/train_yolo.py \
  --model runs/detect/train-2/weights/best.pt \
  --epochs 25 \
  --imgsz 320 \
  --batch 4 \
  --workers 0 \
  --name train_occlusion_25epoch
```

The machine appeared to be training on CPU:

```text
GPU_mem 0G
```

So `imgsz=320`, small batch, and few epochs are recommended for speed.

## Detection Export and Prediction Commands

Visual YOLO prediction videos can be made with `scripts/predict_yolo.py`, but tracking uses structured detection exports from:

```text
scripts/export_yolo_detections.py
```

Export detections from the 25-epoch model after training:

```bash
./env/bin/python scripts/export_yolo_detections.py \
  --model runs/detect/train_occlusion_25epoch/weights/best.pt \
  --video-id random_sim_5bot_dark
```

```bash
./env/bin/python scripts/export_yolo_detections.py \
  --model runs/detect/train_occlusion_25epoch/weights/best.pt \
  --video-id random_sim_6bot_naturallight_occlusion3
```

```bash
./env/bin/python scripts/export_yolo_detections.py \
  --model runs/detect/train_occlusion_25epoch/weights/best.pt \
  --video-id straight_line_6bots
```

Prediction labels show confidence scores. Example:

```text
robot 0.87
```

means YOLO is 87% confident the detected object is a robot.

Use `--conf` to adjust display threshold:

```bash
./env/bin/python scripts/export_yolo_detections.py \
  --model runs/detect/train_occlusion_25epoch/weights/best.pt \
  --video-id random_sim_6bot_naturallight_occlusion3 \
  --conf 0.4 \
  --output outputs/detections/yolo/random_sim_6bot_naturallight_occlusion3_conf04.txt
```

Higher confidence reduces false positives but may miss robots.

## Marker-Based Detection / AprilTags

The user has a separate video:

```text
/Users/prisoe/Downloads/grid_test.mp4
```

It shows Sphero detection using AprilTags from other code, with a grid overlay.

Decision:
- It can be used as qualitative evidence for marker-based detection / Part 4A.
- Do not mix it into YOLO test metrics unless it is annotated or the other code exports structured detections.

Best framing:

```text
Marker-based detection prototype using AprilTags and grid overlay.
Quantitative comparison depends on collecting or annotating marker-visible sequences.
```

If the AprilTag code can export:

```text
frame,id,x,y,w,h,confidence
```

then it can be converted into the same tracking/evaluation format later.

## Slide Deadline Context

Slides are due tonight at 11:59 PM. Report is due Friday. Presentation is Thursday.

Recommended slide focus:
- Problem statement.
- Dataset and annotation summary.
- Pipeline architecture.
- YOLO detector implementation.
- Preliminary predictions / screenshots.
- Tracking and evaluation as next steps.

Useful dataset slide numbers after adding the latest annotations:

```text
10 annotated videos
5349 training frames
446 validation frames
1970 test frames
Conditions: straight-line motion, random motion, stationary robot, dark lighting, natural lighting, occlusions, full obstruction, 3-6 robots
```

## Implemented Detection and Tracking Pipeline

Current implemented flow:

```text
YOLO model -> exported detections -> centroid/SORT/DeepSORT-lite tracks -> metrics -> visualizations
```

Scripts:

```text
scripts/export_yolo_detections.py
scripts/evaluate_yolo_detections.py
scripts/track_centroid.py
scripts/track_sort.py
scripts/track_deepsort.py
scripts/evaluate_tracks.py
scripts/visualize_tracks.py
```

Tracker outputs:

```text
outputs/tracks/centroid/
outputs/tracks/sort/
outputs/tracks/deepsort/
```

Metrics outputs:

```text
outputs/metrics/yolo_detection_metrics.csv
outputs/metrics/centroid_tracking_metrics.csv
outputs/metrics/sort_tracking_metrics.csv
outputs/metrics/deepsort_tracking_metrics.csv
```

DeepSORT implementation note: `deep-sort-realtime` was not installed, so the repo currently uses a dependency-light `DeepSORT-lite`: Kalman motion prediction plus HSV histogram appearance matching from detected crops. This is valid to describe as appearance-augmented tracking, but not as the exact external DeepSORT package.

## Current Baseline Metrics

These metrics were generated before the planned 25-epoch retraining. Re-run them after exporting detections from the new model.

YOLO detection metrics:

```text
random_sim_5bot_dark                      F1 0.6922  mean IoU 0.6716
random_sim_6bot_naturallight_occlusion3   F1 0.8492  mean IoU 0.7466
straight_line_6bots                       F1 0.9789  mean IoU 0.8052
TOTAL                                     F1 0.7415  mean IoU 0.7001
```

Tracking comparison:

```text
Tracker    TOTAL MOTA   TOTAL ID switches
Centroid   0.4704       118
SORT       0.4722       98
DeepSORT   0.4720       101
```

Interpretation: detection quality is the main bottleneck, and SORT/DeepSORT-lite reduce ID switches compared with centroid. DeepSORT-lite does not clearly outperform SORT, which is plausible because the robots are visually very similar and appearance cues are weak.

## Post-Training Steps

After the 25-epoch model finishes:

1. Export detections:

   ```bash
   ./env/bin/python scripts/export_yolo_detections.py --model runs/detect/train_occlusion_25epoch/weights/best.pt --video-id random_sim_5bot_dark
   ./env/bin/python scripts/export_yolo_detections.py --model runs/detect/train_occlusion_25epoch/weights/best.pt --video-id random_sim_6bot_naturallight_occlusion3
   ./env/bin/python scripts/export_yolo_detections.py --model runs/detect/train_occlusion_25epoch/weights/best.pt --video-id straight_line_6bots
   ```

2. Evaluate detection:

   ```bash
   ./env/bin/python scripts/evaluate_yolo_detections.py
   ```

3. Re-run trackers:

   ```bash
   ./env/bin/python scripts/track_centroid.py
   ./env/bin/python scripts/track_sort.py
   ./env/bin/python scripts/track_deepsort.py
   ```

4. Re-run tracking metrics:

   ```bash
   ./env/bin/python scripts/evaluate_tracks.py --tracks-dir outputs/tracks/centroid --output outputs/metrics/centroid_tracking_metrics.csv
   ./env/bin/python scripts/evaluate_tracks.py --tracks-dir outputs/tracks/sort --output outputs/metrics/sort_tracking_metrics.csv
   ./env/bin/python scripts/evaluate_tracks.py --tracks-dir outputs/tracks/deepsort --output outputs/metrics/deepsort_tracking_metrics.csv
   ```

5. Generate visualizations as needed:

   ```bash
   ./env/bin/python scripts/visualize_tracks.py --tracks-dir outputs/tracks/sort --output-dir outputs/videos/sort --tracker-name sort_tracks --video-id random_sim_6bot_naturallight_occlusion3 --max-frames 120
   ```

## Key Cautions

- Do not include empty `random_sim_5bot_dark` YOLO labels after `frame_001670.txt` in training.
- Use wrapper scripts so YOLO outputs land in the current repo.
- For tracking evaluation, verify whether track IDs remain consistent through full occlusion, especially in `occlusion2`.
- Unannotated videos are fine for qualitative demos but not for quantitative metrics.
- YOLO training is slow on CPU; use `yolov8n`, `imgsz=320`, low epochs, and `batch=4`.
