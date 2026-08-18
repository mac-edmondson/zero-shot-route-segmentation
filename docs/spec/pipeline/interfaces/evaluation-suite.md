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

## TODO: This needs actually implemented and some details in this spec filled in

Some general ideas are given in the below sections but these should not be taken as gospel by any stretch.

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
