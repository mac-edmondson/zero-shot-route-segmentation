# EvaluationSuite Interface

**Depends on:** [RouteDiscriminatorPipeline](route-discriminator-pipeline.md), [HoldDetector](hold-detector.md), [RouteDiscriminator](route-discriminator.md), [DataPreprocessingPipeline](data-preprocessing-pipeline.md)
**Consumed by:** offline experiment tooling and [Dashboard Backend](dashboard-backend.md)

## 1. Responsibility

Run reproducible evaluations and gather results for:

- full route-discriminator pipelines;
- hold detectors in isolation;
- route discriminators in isolation;
- clean vs distorted inputs;
- discriminator/configuration variants such as color-only, color+spatial, DINO-only, and color+spatial+DINO.

## 2. Specification

![Evaluation Spec. Diagram](/docs/diagrams/spec_experiment_pipeline.drawio.svg)

## 3. Result contract

The first implementation increment is the contract and pure reporting utilities in `src/pipeline/evaluation/`. It does not run models or persist files.

`EvaluationReport` contains run identity, target, dataset/model descriptors, one required `clean` block, and an optional separate `distorted` block. Each block contains `EvaluationCaseResult` values preserving stable image/source IDs, split, augmentation metadata, status, timing, structured predictions and annotations, and scalar metrics.

Statuses are `evaluated`, `unevaluable`, and `error`. Missing ground truth is unevaluable; failed cases are recorded as errors. Correctness aggregates exclude both statuses while retaining their counts. All result types expose `to_dict()`, and `metric_rows(report)` returns flat JSON-safe records for tables and plots. PIL images are deliberately not serializable.

## 4. Metrics

Hold evaluation uses deterministic one-to-one polygon matching with pixel IoU and a default threshold of `0.50`, reporting TP, FP, FN, precision, recall, F1, and mean matched IoU. Route evaluation uses `Hold.attributes["route_id"]` for annotations; predicted holds are linked by hold matches, then routes are matched one-to-one by maximum hold-membership F1.

### 4. Provenance

Evaluation requires comparison between clean and distorted inputs. Therefore all dataset/image envelopes SHOULD preserve:

- stable source image ID;
- source URI/path when safe to expose;
- split (`train`, `val`, `test`, or custom);
- augmentation recipe/seed;
- parent/source ID for derived images;
- pipeline/configuration IDs used to generate results.

This provenance is formalized in [data-models.md](interfaces/data-models.md) and consumed by [evaluation-suite.md](interfaces/evaluation-suite.md).

### 5. Configuration identity

Every swappable model or algorithm SHOULD expose a stable `implementation_id` and a serializable configuration. Evaluation reports MUST record both, so a result can be tied to the exact detector/discriminator variant, including board-mentioned conditions such as color-only, color+spatial, DINO-only, and color+spatial+DINO.
