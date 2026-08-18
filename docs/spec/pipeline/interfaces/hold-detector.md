# HoldDetector Interface

**Depends on:** [Shared Data Models](data-models.md)
**Created by:** [HoldDetectorFactory](hold-detector-factory.md)
**Consumed by:** [RouteDiscriminatorPipeline](route-discriminator-pipeline.md), [EvaluationSuite](evaluation-suite.md)

## 1. Responsibility

Detect all climbing holds visible in each input image and return their geometry (and optional metadata) without deciding which route they belong to.

## 2. Interface Contract, self-documenting

Implementation: [/src/pipeline/hold_detector/hold_detector.py](/src/pipeline/hold_detector/hold_detector.py)
