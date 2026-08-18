# RouteDiscriminator Interface

**Depends on:** [Shared Data Models](data-models.md), detections from [HoldDetector](hold-detector.md)
**Created by:** [RouteDiscriminatorFactory](route-discriminator-factory.md)
**Consumed by:** [RouteDiscriminatorPipeline](route-discriminator-pipeline.md), [EvaluationSuite](evaluation-suite.md)

## 1. Responsibility

Given images and already-detected holds, group those holds into climbing routes. It MUST NOT be responsible for detecting hold geometry from scratch.

## 2. Interface Contract, self-documenting

Implementation: [/src/pipeline/route_discriminator/route_discriminator.py](/src/pipeline/route_discriminator/route_discriminator.py)
