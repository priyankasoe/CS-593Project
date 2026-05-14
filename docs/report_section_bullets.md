# Report Section Bullet Points

## Abstract

- This project develops a vision-based multi-object tracking pipeline for visually similar Sphero BOLT swarm robots.
- The main challenge is preserving robot identity across frames when robots are nearly identical, move close together, become occluded, or appear under poor lighting.
- The system uses a fine-tuned YOLO detector to localize robots in overhead video frames.
- YOLO detections are passed into four tracking methods:
  - centroid tracking,
  - SORT,
  - DeepSORT-lite,
  - real DeepSORT with neural appearance embeddings.
- The dataset contains annotated videos covering straight-line motion, random motion, mixed stationary/moving behavior, natural lighting, dark lighting, and occlusion.
- The fine-tuned detector is compared against a pretrained YOLO baseline; the pretrained model produced no valid Sphero detections on the validation video.
- Results show that detection is strong in clear natural/straight-line settings but degrades in dark random-motion scenes.
- Real DeepSORT reduced ID switches compared with simpler trackers, especially in occlusion scenarios.

## Related Works

- Multi-object tracking combines detection and temporal association to maintain object identities across video frames.
- SORT is a common baseline that uses Kalman filtering for motion prediction and IoU-based Hungarian matching for data association.
- DeepSORT extends SORT by adding appearance embeddings, allowing the tracker to use visual similarity as well as motion.
- YOLO-style detectors provide real-time object detection and are useful as front-end detectors for tracking pipelines.
- AprilTag or fiducial-marker systems can provide precise robot localization when markers are visible, but performance depends on marker visibility and can degrade under occlusion or blur.
- Prior Sphero localization work used a two-stage CNN framework:
  - first detect robot bounding boxes and robot type,
  - then crop each robot and use separate CNNs for instance identification and orientation estimation.
- That prior work directly estimates robot identity and orientation from appearance, while this project focuses on temporal identity preservation through multi-object tracking.
- The prior two-stage CNN approach was limited by small robot size after image downscaling and difficulty separating nearby robots.
- This project addresses similar challenges by evaluating YOLO-based detection and multiple trackers under occlusion, lighting variation, and increasing robot counts.

## Approach

- Data was collected from overhead videos of multiple Sphero BOLT robots.
- Annotations were exported from CVAT in:
  - YOLO format for detector training,
  - MOT format for tracking evaluation.
- The current dataset split is:
  - train: 6 videos,
  - validation: 2 videos,
  - test: 2 videos.
- The generated YOLO dataset contains:
  - 5,349 training frames,
  - 446 validation frames,
  - 1,970 test frames.
- Videos include:
  - straight-line motion,
  - random swarm motion,
  - two-moving/one-stationary motion,
  - natural-light occlusion,
  - dark lighting.
- A YOLOv8n detector was fine-tuned on the Sphero annotations.
- Detections are exported in the format:
  - `frame,x1,y1,x2,y2,confidence,class`.
- Tracking outputs use MOT-style rows:
  - `frame,id,x,y,w,h,confidence,class,visibility`.
- Centroid tracker:
  - computes each detection center,
  - matches detections by nearest center distance,
  - creates/deletes tracks based on unmatched detections.
- SORT:
  - predicts box locations with a Kalman filter,
  - matches predictions to detections using IoU,
  - maintains tracks through short missed detections.
- DeepSORT-lite:
  - adds simple HSV color-histogram appearance features to SORT-style tracking.
- Real DeepSORT:
  - uses the `deep-sort-realtime` package,
  - adds neural appearance embeddings from detection crops,
  - matches detections using motion and learned appearance similarity.
- Evaluation metrics include:
  - precision,
  - recall,
  - F1,
  - mean IoU,
  - MOTA,
  - MOTP IoU,
  - ID switches.

## Results

- A pretrained COCO YOLOv8n model produced zero valid detections on `straight_line_6bots`.
- This shows that task-specific fine-tuning is necessary for detecting Sphero robots.
- The fine-tuned detector performed best on structured or well-lit videos:
  - `straight_line_6bots`: F1 = 99.5%.
  - `random_sim_6bot_naturallight_occlusion4`: F1 = 90.7%.
  - `random_sim_6bot_naturallight_occlusion3`: F1 = 88.2%.
- The hardest condition was the dark random-motion video:
  - `random_sim_5bot_dark`: F1 = 52.5%.
- This suggests that lighting variation is a major source of detector failure.
- Raising the detection confidence threshold from 0.25 to 0.35:
  - improved precision,
  - reduced false positives,
  - reduced ID switches,
  - slightly reduced recall.
- At confidence 0.35, SORT improved tracking stability compared with centroid tracking:
  - centroid total ID switches: 90,
  - SORT total ID switches: 81.
- Real DeepSORT gave the best identity preservation on the videos where it was evaluated:
  - straight-line 6 robots: 3 ID switches,
  - natural-light occlusion 3: 12 ID switches,
  - natural-light occlusion 4: 12 ID switches.
- For natural-light occlusion 3:
  - SORT had 24 ID switches,
  - real DeepSORT had 12 ID switches.
- Real DeepSORT reduced identity switches substantially because neural appearance embeddings helped distinguish robots after close interactions.
- However, all trackers were still limited by detection quality; missed detections and false positives directly reduced MOTA.
- The dark 5-robot video had low MOTA for all trackers because detector recall was poor in that condition.

## Conclusions

- The project successfully implemented an end-to-end robot detection and tracking pipeline:
  - video input,
  - YOLO detection,
  - detection export,
  - multi-object tracking,
  - MOT-style evaluation,
  - visualized tracking outputs.
- Fine-tuning YOLO was necessary because a generic pretrained YOLO model did not detect the Sphero robots.
- The trained detector generalized well to clear and natural-light videos but struggled in dark random-motion scenes.
- Tracking performance depended strongly on detection quality.
- SORT improved over centroid tracking by adding motion prediction.
- Real DeepSORT produced the strongest identity preservation by using neural appearance embeddings.
- DeepSORT-lite was less effective because simple color histograms are weak features for visually similar Sphero robots.
- The results support the main project claim: identity preservation becomes harder under occlusion, crowding, and lighting variation.
- Future work could include:
  - collecting more dark-lighting training data,
  - adding AprilTag marker-based comparison,
  - training a Sphero-specific re-identification model,
  - estimating robot orientation,
  - testing larger YOLO models,
  - deploying the pipeline in real time.

