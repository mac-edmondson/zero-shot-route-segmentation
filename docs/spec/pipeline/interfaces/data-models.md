# Shared Data Models

**Used by:** [HoldDetector](hold-detector.md), [RouteDiscriminator](route-discriminator.md), [RouteDetectionPipeline](route-detection-pipeline.md), [AugmentationSuite](augmentation-suite.md), [DataPreprocessingPipeline](data-preprocessing-pipeline.md), [EvaluationSuite](evaluation-suite.md), and the [Dashboard Backend](dashboard-backend.md).

## 1. Purpose

Provide stable, implementation-neutral structures shared by all components. The image representation is intentionally abstract so model implementations can use PIL, NumPy, tensors, or another internal type behind an adapter.

## 2. Interface Contract, self-documenting

Implementation: [/src/pipeline/interfaces/data_models.py](/src/pipeline/interfaces/data_models.py)
