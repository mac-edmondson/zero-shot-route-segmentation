# RouteDiscriminator Interface

TODO: Port to code and clean

**Depends on:** [Shared Data Models](data-models.md), detections from [HoldDetector](hold-detector.md)
**Created by:** [RouteDiscriminatorFactory](route-discriminator-factory.md)
**Consumed by:** [RouteDetectionPipeline](route-detection-pipeline.md), [EvaluationSuite](evaluation-suite.md)

## 1. Responsibility

Given images and already-detected holds, group those holds into climbing routes. It MUST NOT be responsible for detecting hold geometry from scratch.

## 2. Protocol

```python
from typing import Protocol, Sequence
from .data_models import Hold, Image, Route

class RouteDiscriminator(Protocol):
    @property
    def implementation_id(self) -> str: ...

    def get_routes(
        self,
        images: Sequence[Image],
        holds: Sequence[Sequence[Hold]],
    ) -> list[list[Route]]:
        """Return one route list per input image."""

    @staticmethod
    def mark_routes(
        images: Sequence[Image],
        routes: Sequence[Sequence[Route]],
    ) -> list[Image]:
        """Return visualization images with routes overlaid."""
```

### Board compatibility

- `get_routes` corresponds to `getRoutes(imgs, holds) -> List[List[Route]]`.
- `mark_routes` corresponds to `@staticmethod markRoutes(...) -> List[Image]`.

## 3. Contract

`get_routes` MUST:

- require `len(images) == len(holds)`;
- preserve image order;
- return one route list per image;
- construct each [Route](data-models.md#2-canonical-python-facing-contracts) from hold objects originating from the corresponding image;
- permit zero routes for a valid image;
- define or document whether a hold MAY belong to more than one route.

The last point is unresolved in the source board and is tracked in [Open Questions](../decisions/open-questions.md#route-semantics).

## 4. Feature variants

The board calls out evaluation variants such as:

- color-only;
- color + spatial;
- DINO-only;
- color + spatial + DINO.

These SHOULD be represented as classifier configurations behind the same `RouteDiscriminator` protocol, not as separate downstream APIs. This lets [EvaluationSuite](evaluation-suite.md) compare variants uniformly.

## 5. Visualization

`mark_routes` SHOULD make distinct routes visually separable and SHOULD call `Route.mark_route()` or an equivalent rendering primitive rather than reimplementing route geometry rules.
