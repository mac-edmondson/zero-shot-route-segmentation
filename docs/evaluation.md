# Evaluation Suite

This document describes the evaluation pipeline implemented under
`src/pipeline/evaluation/`. It evaluates detector-to-route combinations, keeps clean and distorted conditions separate,
and writes machine-readable results for tables, plots, and error analysis.

## Pipeline overview

The benchmark flow is:

1. Load Roboflow COCO annotations and source images.
2. Convert each image into the shared `ImageRecord` model.
3. Normalize Roboflow route labels into `Hold.attributes["route_id"]`.
4. Run configured hold detectors.
5. Match detector predictions to annotations at IoU 0.50 and transfer route IDs.
6. Run every configured route discriminator on the matched detector predictions.
7. Serialize one JSON artifact per combination plus a manifest.

The matrix evaluates genuine detector-to-route combinations. The existing
`evaluate_route_discriminator()` API remains available for isolated grouping
quality checks using annotated holds directly.

## Data loading

`src/pipeline/utility/ground_truth_loader.py` loads the restructured Roboflow
COCO export. It reads image entries and polygon segmentations and returns holds
grouped in COCO image order.

`run_all_eval.py` searches the supplied dataset root recursively for
`*_annotations.coco.json` files. It resolves referenced images, loads polygon
annotations into `Hold` values, preserves route labels, converts labels such as
`route_3` to numeric route IDs, and excludes unknown route labels from route
ground truth. Files under `distorted` or `augmented` directories are marked as
distorted.

Expected input layout:

```text
data/roboflow/
  train/images/ *_annotations.coco.json
  valid/images/ *_annotations.coco.json
  test/images/  *_annotations.coco.json
```

The runner selects the requested split, defaulting to `test`.

## Hold detector evaluation

`evaluate_hold_detector()` accepts any implementation satisfying the shared
`HoldDetector` protocol. Current factory options include Mask R-CNN, YOLOv8,
SAM3, and Mock; future detectors use the same interface.

Ground truth excludes annotations with `hold_type == "volume"`. Volume
predictions remain predictions and count as false positives.

The evaluator reports COCO-style AP at IoU thresholds `0.50, 0.55, ..., 0.95`:

- `AP50` and `AP75`;
- AP for every configured threshold;
- `mAP`, averaged over configured thresholds;
- prediction and ground-truth counts;
- mean inference time.

Predictions are ranked by `Hold.attributes["confidence"]`. Missing confidence
uses a recorded fallback of `1.0`. A valid binary `attributes["mask"]` is used
for geometry when available; polygon rasterization is the fallback.

## Route-discriminator evaluation

`evaluate_route_discriminator()` accepts any implementation satisfying the
shared `RouteDiscriminator` protocol, including Triplet MLP, DINO clustering,
DINO learning, and future CIELAB/DINOv3 implementations.

The evaluator supplies ground-truth holds directly. Ground-truth routes are
formed from numeric `route_id` values; volumes and unknown/unlabelled holds are
excluded.

Reported metrics include pairwise precision, recall, and F1 for same-route hold
pairs, cluster purity, adjusted Rand index, normalized mutual information,
route counts, and mean inference time.

## Independent evaluation

Use `--independent` to produce detector-only and route-only reports. Detectors
are evaluated against the annotated holds in the evaluation set. Route
discriminators receive those annotated holds directly, so their scores measure
route grouping without detector recall or localization affecting the result.
Reports are written to `results/independent/detectors.json` and
`results/independent/route_discriminators.json`.

## Detector-to-route matrix

`run_evaluation_matrix()` runs Mask R-CNN, YOLOv8, and SAM3 against the
registered route discriminators. Detector predictions are inferred once per
image, matched to unused annotated holds at IoU `0.50`, and passed to every
route discriminator. Unmatched predictions are retained for detector metrics
but excluded from route metrics.

The default matrix includes Triplet MLP, Color Only, Ground Truth and Mock
baselines, DINO clustering with HDBSCAN/DBSCAN/agglomerative methods with
`color_weight` 0.0 and 1.0, and learned DINO with weighted and mean pooling.
Each combination is saved separately and indexed by `manifest.json`.

## Results and execution

Each run produces an `EvaluationReport` with run identity, dataset metadata,
implementation configuration, per-image cases, aggregate metrics, and separate
clean/distorted blocks. Cases are `evaluated`, `unevaluable`, or `error`.
Reports are JSON-safe; detector masks are represented by availability and shape
metadata. `metric_rows(report)` produces flat records for tables and plots.

## Metric reference

For quality metrics, higher is better unless noted otherwise.

### Hold detection metrics

| Metric | Direction | Meaning and calculation |
|---|---|---|
| Precision | Higher | Correct detections divided by all detections: `TP / (TP + FP)`. It measures how many predictions are correct. |
| Recall | Higher | Correct detections divided by all labelled holds: `TP / (TP + FN)`. It measures how many holds were found. |
| F1 | Higher | Harmonic mean of precision and recall: `2PR / (P + R)`. It balances missed holds and false detections. |
| AP50 | Higher | Average precision at IoU threshold `0.50`, using confidence-ranked predictions. |
| AP75 | Higher | Average precision at the stricter IoU threshold `0.75`. |
| AP50–AP95 | Higher | Average precision at each IoU threshold from `0.50` to `0.95` in steps of `0.05`. |
| mAP | Higher | Mean of AP across all configured IoU thresholds. This is the primary overall detector score. |
| Mean matched IoU | Higher | Average intersection-over-union of matched prediction/ground-truth shapes. `IoU = intersection / union`. |
| Prediction count | Context only | Number of holds returned by the detector; it is not a quality score by itself. |
| Ground-truth count | Context only | Number of labelled positive holds used for scoring. |
| Mean inference time | Lower | Average seconds required to process one image. |

`TP` is a true positive, `FP` is a false positive, and `FN` is a missed ground
truth hold. AP additionally uses confidence ranking, so it evaluates the quality
of the detector across score cutoffs rather than at only one threshold.

### Route-discriminator metrics

These metrics evaluate grouping using the same ground-truth holds as input; hold
detection quality is therefore excluded.

| Metric | Direction | Meaning and calculation |
|---|---|---|
| Pairwise precision | Higher | Of all hold pairs placed in the same predicted route, the fraction that belong to the same true route. |
| Pairwise recall | Higher | Of all hold pairs that belong to the same true route, the fraction grouped together by the discriminator. |
| Pairwise F1 | Higher | Harmonic mean of pairwise precision and recall; the primary route-grouping balance score. |
| Purity | Higher | For each predicted route, the fraction belonging to its most common true route, averaged over holds. `1.0` is perfectly pure. |
| Adjusted Rand Index (ARI) | Higher | Agreement between predicted and true pair assignments, adjusted for agreement expected by chance. `1.0` is perfect; `0` is chance-level; it can be negative when worse than chance. |
| Normalized Mutual Information (NMI) | Higher | Shared information between predicted and true route labels, normalized to a comparable scale. `1.0` is perfect and `0` means no shared information. |
| Predicted route count | Context only | Number of route groups produced by the discriminator. It should be interpreted alongside the ground-truth route count. |
| Ground-truth route count | Context only | Number of labelled routes available in the image. |
| Mean inference time | Lower | Average seconds required to group the holds for one image. |

For all metrics, compare models on the same images, annotations, condition, and
configuration. A high count or low runtime alone does not imply better model
quality.

Run the configured suite with:

```bash
PYTHONPATH=src python -m pipeline.evaluation.run_all_eval \
  --dataset-root data/roboflow \
  --split test \
  --output results/evaluation_matrix/manifest.json
```

Defaults are `Mask-RCNN`, `YOLO`, and `SAM 3`, plus `Triplet MLP`, `DINO
Clustering`, and `DINO Learning`. Repeat `--detector` or
`--route-discriminator` to replace the defaults. Successful reports and
component construction/evaluation errors are retained together.

## Current status

The result contract, metric utilities, hold detector evaluator, route
discriminator evaluator, Roboflow loader integration, and consolidated runner
are implemented. Regression tests, smoke tests, compilation, and Ruff checks
pass. A full Roboflow benchmark artifact requires the dataset, model weights,
and optional runtime dependencies.
