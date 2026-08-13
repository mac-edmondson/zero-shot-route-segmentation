# HoldDetector Interface

TODO: Port to code and clean

**Depends on:** [Shared Data Models](data-models.md)
**Created by:** [HoldDetectorFactory](hold-detector-factory.md)
**Consumed by:** [RouteDetectionPipeline](route-detection-pipeline.md), [EvaluationSuite](evaluation-suite.md)

## 1. Responsibility

Detect all climbing holds visible in each input image and return their geometry (and optional metadata) without deciding which route they belong to.

## 2. Protocol

```python
from typing import Protocol, Sequence
from .data_models import Hold, Image

class HoldDetector(Protocol):
    @property
    def implementation_id(self) -> str: ...

    def get_holds(
        self,
        images: Sequence[Image],
    ) -> list[list[Hold]]:
        """Return one list of detected holds per input image."""

    @staticmethod
    def mark_holds(
        images: Sequence[Image],
        holds: Sequence[Sequence[Hold]],
    ) -> list[Image]:
        """Return visualization images with hold detections overlaid."""
```

### Board compatibility

- `get_holds` corresponds to `getHolds(imgs: List[images]) -> List[List[Holds]]`.
- `mark_holds` corresponds to the static helper described as marking/segmenting holds.

## 3. Contract

`get_holds` MUST:

- return exactly one result list for each input image;
- preserve input image order;
- return an empty list when no holds are detected in a valid image;
- produce [Hold](data-models.md#2-canonical-python-facing-contracts) objects whose polygons use the image's coordinate system;
- avoid assigning route membership. Route membership belongs to [RouteClassifier](route-classifier.md).

An implementation SHOULD attach confidence or method-specific values through `Hold.attributes` rather than changing the public protocol.

## 4. Visualization helper

`mark_holds` MUST NOT change detection semantics. It is a presentation/debugging function used by the [Dashboard](dashboard-frontend.md) and evaluation artifacts.

`mark_holds(images, holds)` MUST require matching batch lengths. It SHOULD draw at least the hold polygon and MAY draw centroid/confidence information.

## 5. Errors

Recommended error classes:

```python
class HoldDetectionError(RuntimeError): ...
class InvalidImageError(HoldDetectionError): ...
class BatchAlignmentError(HoldDetectionError): ...
```

Implementations MUST document whether one invalid image fails the entire batch or is returned as an item-level error envelope.

## 6. Evaluation hooks

[EvaluationSuite](evaluation-suite.md) needs direct access to a detector rather than only a composed pipeline. Therefore detector implementations SHOULD expose:

- `implementation_id`;
- serializable configuration;
- optional timing/diagnostic metadata;
- deterministic inference where possible for a fixed model/configuration.
