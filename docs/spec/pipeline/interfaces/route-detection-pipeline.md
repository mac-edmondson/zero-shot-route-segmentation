# RouteDetectionPipeline Interface

TODO: Port to code and clean

**Composes:** [HoldDetector](hold-detector.md) + [RouteClassifier](route-classifier.md)
**Consumed by:** [EvaluationSuite](evaluation-suite.md), [Dashboard Backend](dashboard-backend.md)

## 1. Responsibility

Provide the canonical end-to-end inference entry point while retaining direct access to the two component stages for individual evaluation.

## 2. Construction

```python
from dataclasses import dataclass
from typing import Sequence
from .data_models import Image, PipelineResult
from .hold_detector import HoldDetector
from .route_classifier import RouteClassifier

@dataclass
class RouteDetectionPipeline:
    hold_detector: HoldDetector
    route_classifier: RouteClassifier

    def run(self, images: Sequence[Image]) -> PipelineResult: ...
```

A convenience constructor MAY use [HoldDetectorFactory](hold-detector-factory.md) and [RouteClassifierFactory](route-classifier-factory.md), but dependency injection of already-created components SHOULD remain supported for testing and evaluation.

## 3. Inference flow

```mermaid
sequenceDiagram
    participant C as Caller
    participant P as RouteDetectionPipeline
    participant H as HoldDetector
    participant R as RouteClassifier

    C->>P: get_routes(images)
    P->>H: get_holds(images)
    H-->>P: holds_by_image
    P->>R: get_routes(images, holds_by_image)
    R-->>P: routes_by_image
    P-->>C: PipelineResult
```

## 4. Pipeline identity

The pipeline SHOULD expose a serializable descriptor.

[EvaluationSuite](evaluation-suite.md) MUST persist this descriptor with results.
