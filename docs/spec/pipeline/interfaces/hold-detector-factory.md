# HoldDetectorFactory Interface

**Produces:** [HoldDetector](hold-detector.md)
**Used by:** [RouteDiscriminatorPipeline](route-discriminator-pipeline.md), CLI/configuration code, [EvaluationSuite](evaluation-suite.md)

## 1. Responsibility

Create a concrete hold detector by stable method name while preventing higher-level code from importing model-specific classes.

## 2. Interface Contract, self-documenting

Implementation: [/src/pipeline/hold_detector/hold_detector_factory.py](/src/pipeline/hold_detector/hold_detector_factory.py)
